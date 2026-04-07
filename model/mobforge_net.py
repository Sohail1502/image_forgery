"""
MobForge-Net: Lightweight Dual-Stream Image Forgery Detection & Localization
============================================================================
Novel contributions over base paper (UFG-Net, Neurocomputing 2025):
  1. Dual-stream encoder: RGB + SRM noise residual (base papers use RGB only)
  2. MobileNetV3-Small backbone (vs heavy PVTv2 transformer in UFG-Net)
  3. Boundary-aware combined loss (Edge-BCE + Dice + boundary weight map)
  4. Inference speed logging displayed in output (novel UI contribution)
  5. Lightweight channel attention fusion instead of redundant parallel fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ─────────────────────────────────────────────
# SRM Noise Residual Filters
# ─────────────────────────────────────────────
class SRMFilter(nn.Module):
    """
    Spatial Rich Model filters (3 kernels).
    These amplify invisible camera noise; when a region is copy-pasted,
    its noise pattern breaks — SRM makes that break visible to the model.
    Novel: we feed this as a parallel stream alongside RGB.
    """
    def __init__(self):
        super().__init__()
        # Three classic SRM high-pass kernels
        f1 = np.array([[ 0,  0,  0,  0,  0],
                       [ 0, -1,  2, -1,  0],
                       [ 0,  2, -4,  2,  0],
                       [ 0, -1,  2, -1,  0],
                       [ 0,  0,  0,  0,  0]], dtype=np.float32) / 4.0

        f2 = np.array([[-1,  2, -2,  2, -1],
                       [ 2, -6,  8, -6,  2],
                       [-2,  8,-12,  8, -2],
                       [ 2, -6,  8, -6,  2],
                       [-1,  2, -2,  2, -1]], dtype=np.float32) / 12.0

        f3 = np.array([[ 0,  0,  0,  0,  0],
                       [ 0,  0,  0,  0,  0],
                       [ 0,  1, -2,  1,  0],
                       [ 0,  0,  0,  0,  0],
                       [ 0,  0,  0,  0,  0]], dtype=np.float32) / 2.0

        # Stack into conv weights: shape [3, 1, 5, 5]
        kernels = np.stack([f1, f2, f3], axis=0)[:, np.newaxis]
        self.conv = nn.Conv2d(1, 3, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(kernels))
        for p in self.conv.parameters():
            p.requires_grad = False  # fixed filters, not learned

    def forward(self, x):
        # x: [B, 3, H, W] — convert to grayscale first, then filter
        gray = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        noise = self.conv(gray)  # [B, 3, H, W]
        return torch.clamp(noise, -3.0, 3.0)


# ─────────────────────────────────────────────
# Lightweight Channel Attention Fusion
# ─────────────────────────────────────────────
class ChannelAttentionFusion(nn.Module):
    """
    Fuse RGB and SRM feature maps with channel attention.
    Novel vs UFG-Net: no redundant parallel domain processing;
    we merge early with learned importance weights per channel.
    """
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels * 2),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, feat_rgb, feat_srm):
        cat = torch.cat([feat_rgb, feat_srm], dim=1)  # [B, 2C, H, W]
        w = self.fc(self.gap(cat)).unsqueeze(-1).unsqueeze(-1)
        attended = cat * w
        return F.relu(self.bn(self.fuse_conv(attended)))


# ─────────────────────────────────────────────
# U-Net Decoder Block
# ─────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Pad if sizes don't match exactly
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Main MobForge-Net
# ─────────────────────────────────────────────
class MobForgeNet(nn.Module):
    """
    MobForge-Net Architecture:
    ─────────────────────────
    Input (H×W×3)
      ├─ Stream 1: MobileNetV3-Small encoder  → [C, H/32, W/32]
      └─ Stream 2: SRM filter → MobileNetV3-Small encoder → [C, H/32, W/32]
    Channel Attention Fusion
    U-Net Decoder (4 upsampling stages with skip connections)
    Output: [1, H, W] — forgery probability map (0=authentic, 1=forged)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # SRM branch
        self.srm = SRMFilter()

        # Two MobileNetV3-Small encoders
        def make_encoder(pretrained):
            mob = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            return mob.features

        self.encoder_rgb = make_encoder(pretrained)
        self.encoder_srm = make_encoder(False)  # SRM stream: no pretrain (noise domain)

        # Channel sizes at each stage of MobileNetV3-Small features
        # Layers 1,2,4,9 give us 4 multi-scale feature maps; layer 12 is bottleneck
        self.skip_channels = [16, 24, 40, 96]  # MV3-Small feature channels at skips (not bottleneck)
        self.bottleneck_channels = 576  # Output of layer 12

        # Fusion at bottleneck
        self.fuse = ChannelAttentionFusion(self.bottleneck_channels)

        # U-Net Decoder: dec4 expects bottleneck + skip[3]
        self.dec4 = DecoderBlock(self.bottleneck_channels, self.skip_channels[-1], 256)
        self.dec3 = DecoderBlock(256, self.skip_channels[-2], 128)
        self.dec2 = DecoderBlock(128, self.skip_channels[-3], 64)
        self.dec1 = DecoderBlock(64,  self.skip_channels[-4], 32)
        self.dec0 = DecoderBlock(32,  0, 16)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def _get_skips(self, encoder, x):
        """Extract multi-scale skip features from MobileNetV3 encoder."""
        skips = []
        skip_indices = [1, 2, 4, 9]  # Only 4 skips; layer 12 is bottleneck
        for i, layer in enumerate(encoder):
            x = layer(x)
            if i in skip_indices:
                skips.append(x)
        return x, skips  # (bottleneck at layer 12, [skip1..skip4])

    def forward(self, x):
        # Stream 1: RGB
        noise_map = self.srm(x)           # [B, 3, H, W]
        bottleneck_rgb, skips_rgb = self._get_skips(self.encoder_rgb, x)

        # Stream 2: SRM noise residual
        bottleneck_srm, _ = self._get_skips(self.encoder_srm, noise_map)

        # Fuse bottlenecks
        fused = self.fuse(bottleneck_rgb, bottleneck_srm)

        # U-Net decode with RGB skip connections (reversed order)
        d = self.dec4(fused,  skips_rgb[3])
        d = self.dec3(d,      skips_rgb[2])
        d = self.dec2(d,      skips_rgb[1])
        d = self.dec1(d,      skips_rgb[0])
        d = self.dec0(d,      None)

        # Upsample to input resolution
        d = F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return torch.sigmoid(self.head(d))   # [B, 1, H, W], values in [0,1]


# ─────────────────────────────────────────────
# Boundary-Aware Combined Loss
# ─────────────────────────────────────────────
class BoundaryAwareLoss(nn.Module):
    """
    Novel loss = λ1 * EdgeWeightedBCE + λ2 * DiceLoss + λ3 * BoundaryDice
    - Edge-weighted BCE: pixels near forged region boundaries get higher weight
    - Dice: handles class imbalance (most pixels are authentic)
    - BoundaryDice: specifically penalizes blurry edges
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.5):
        super().__init__()
        self.l1, self.l2, self.l3 = lambda1, lambda2, lambda3

    def _boundary_map(self, mask):
        """Extract boundary pixels via max-pool dilation."""
        dilated  = F.max_pool2d(mask,  kernel_size=5, stride=1, padding=2)
        eroded   = -F.max_pool2d(-mask, kernel_size=5, stride=1, padding=2)
        boundary = (dilated - eroded).clamp(0, 1)
        return boundary

    def forward(self, pred, target):
        target = target.float()

        # 1. Edge-weighted BCE
        boundary = self._boundary_map(target)
        weight = 1.0 + 4.0 * boundary  # boundary pixels weighted 5x
        bce = F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')

        # 2. Dice loss
        smooth = 1e-5
        p, t = pred.view(-1), target.view(-1)
        dice_loss = 1.0 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

        # 3. Boundary Dice
        pred_b  = self._boundary_map(pred)
        targ_b  = self._boundary_map(target)
        pb, tb  = pred_b.view(-1), targ_b.view(-1)
        bdice   = 1.0 - (2 * (pb * tb).sum() + smooth) / (pb.sum() + tb.sum() + smooth)

        return self.l1 * bce + self.l2 * dice_loss + self.l3 * bdice


if __name__ == "__main__":
    model = MobForgeNet(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")
