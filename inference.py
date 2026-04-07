"""
Inference Script — MobForge-Net
Shows forgery mask + inference time in milliseconds on the output image.
This directly addresses the reviewer's request to show "how fast the model
predicted the forged region" as part of the output.

Usage:
  python inference.py --image path/to/image.jpg --weights checkpoints/best_model.pth
  python inference.py --image path/to/image.jpg --weights checkpoints/best_model.pth --demo
"""

import os
import time
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model.mobforge_net import MobForgeNet


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
def preprocess(image_path, img_size=256):
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor, img, original_size


# ─────────────────────────────────────────────
# Postprocess + overlay
# ─────────────────────────────────────────────
def create_output_image(original_img, prob_map, inference_ms, save_path):
    """
    Creates a side-by-side result image:
    [Original] | [Forgery Heatmap Overlay] | [Binary Mask]
    With inference time printed at the bottom.
    """
    orig_w, orig_h = original_img.size
    target_h = min(orig_h, 400)
    scale    = target_h / orig_h
    target_w = int(orig_w * scale)

    orig_resized = original_img.resize((target_w, target_h), Image.LANCZOS)

    # Resize prob_map to match
    prob_resized = Image.fromarray((prob_map * 255).astype(np.uint8)).resize(
        (target_w, target_h), Image.BILINEAR
    )
    prob_arr = np.array(prob_resized)

    # Heatmap overlay (red = forged regions)
    heatmap = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    heatmap[:, :, 0] = prob_arr  # Red channel
    heatmap = Image.fromarray(heatmap)
    orig_arr = np.array(orig_resized)
    overlay_arr = (0.55 * orig_arr + 0.45 * np.array(heatmap)).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr)

    # Binary mask
    binary_arr = (prob_arr > 127).astype(np.uint8) * 255
    binary_img = Image.fromarray(binary_arr).convert('RGB')

    # Compose side-by-side
    padding  = 10
    label_h  = 60
    total_w  = target_w * 3 + padding * 4
    total_h  = target_h + label_h + padding * 2

    canvas = Image.new('RGB', (total_w, total_h), color=(20, 20, 20))

    # Paste panels
    x_positions = [padding, target_w + padding*2, target_w*2 + padding*3]
    panels = [orig_resized, overlay, binary_img]
    labels = ['Original Image', 'Forgery Heatmap', 'Detected Mask']
    colors = [(200, 200, 200), (255, 140, 0), (80, 200, 80)]

    draw = ImageDraw.Draw(canvas)

    for x, panel, label, color in zip(x_positions, panels, labels, colors):
        canvas.paste(panel, (x, padding))
        draw.text((x + target_w//2, target_h + padding + 8),
                  label, fill=color, anchor='mt')

    # Inference speed line — the key output your reviewer asked for
    forgery_pct = (prob_arr > 127).mean() * 100
    speed_text  = (
        f"Inference time: {inference_ms:.1f} ms  |  "
        f"Forged area: {forgery_pct:.1f}%  |  "
        f"Model: MobForge-Net  |  Input: {orig_w}×{orig_h}px"
    )
    draw.text((total_w // 2, total_h - 18),
              speed_text, fill=(255, 220, 50), anchor='mt')

    canvas.save(save_path)
    return save_path, forgery_pct


# ─────────────────────────────────────────────
# Main inference
# ─────────────────────────────────────────────
def run_inference(image_path, weights_path, img_size=256, device=None, save_dir='outputs', mask_path=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MobForgeNet(pretrained=False)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"[WARNING] No weights found at {weights_path}. Running with random weights (demo only).")
    model.to(device)
    model.eval()

    # Preprocess
    tensor, original_img, original_size = preprocess(image_path, img_size)
    tensor = tensor.to(device)

    # ── Inference with timing ──
    # GPU warm-up (important for accurate timing)
    if device.type == 'cuda':
        with torch.no_grad():
            _ = model(tensor)
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)   # [1, 1, H, W]
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    inference_ms = (end_time - start_time) * 1000.0  # convert to ms

    # Post-process
    prob_map = pred[0, 0].cpu().numpy()  # [H, W], values in [0,1]

    # Load ground truth mask if provided
    accuracy = None
    precision = None
    recall = None
    f1 = None
    iou = None
    dice = None

    if mask_path and os.path.exists(mask_path):
        gt_mask = Image.open(mask_path).convert('L')
        gt_mask = gt_mask.resize((img_size, img_size), Image.NEAREST)
        gt_mask = np.array(gt_mask) / 255.0  # Normalize to [0,1]
        gt_mask = (gt_mask > 0.5).astype(np.float32)  # Binary

        pred_binary = (prob_map > 0.5).astype(np.float32)
        tp = float(((pred_binary == 1) & (gt_mask == 1)).sum())
        fp = float(((pred_binary == 1) & (gt_mask == 0)).sum())
        tn = float(((pred_binary == 0) & (gt_mask == 0)).sum())
        fn = float(((pred_binary == 0) & (gt_mask == 1)).sum())

        correct_pixels = (pred_binary == gt_mask).sum()
        total_pixels = gt_mask.size
        accuracy = (correct_pixels / total_pixels) * 100
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

    # Save result
    os.makedirs(save_dir, exist_ok=True)
    base_name  = os.path.splitext(os.path.basename(image_path))[0]
    save_path  = os.path.join(save_dir, f'{base_name}_result.png')
    out_path, forgery_pct = create_output_image(
        original_img, prob_map, inference_ms, save_path
    )

    # Print summary
    print(f"\n{'='*55}")
    print(f"  MobForge-Net Inference Result")
    print(f"{'='*55}")
    print(f"  Image         : {os.path.basename(image_path)}")
    print(f"  Input size    : {original_size[0]}×{original_size[1]} px")
    print(f"  Inference time: {inference_ms:.2f} ms  ({1000/inference_ms:.1f} FPS)")
    print(f"  Device        : {device}")
    print(f"  Forged area   : {forgery_pct:.2f}%")
    if accuracy is not None:
        print(f"  Accuracy      : {accuracy:.2f}%")
        print(f"  Precision     : {precision:.4f}")
        print(f"  Recall        : {recall:.4f}")
        print(f"  F1 Score      : {f1:.4f}")
        print(f"  IoU           : {iou:.4f}")
        print(f"  Dice          : {dice:.4f}")
    verdict = "FORGED" if forgery_pct > 5 else "AUTHENTIC"
    print(f"  Verdict       : {verdict}")
    print(f"  Output saved  : {out_path}")
    print(f"{'='*55}\n")

    return {
        'inference_ms': inference_ms,
        'fps': 1000 / inference_ms,
        'forgery_pct': forgery_pct,
        'accuracy': accuracy,
        'verdict': verdict,
        'output_path': out_path
    }


# ─────────────────────────────────────────────
# Batch benchmark (for showing speed table)
# ─────────────────────────────────────────────
def benchmark_speed(weights_path, img_size=256, n_runs=50, batch_sizes=[1, 4, 8]):
    """Run speed benchmark across batch sizes — for your results table."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MobForgeNet(pretrained=False).to(device)
    model.eval()

    print(f"\nSpeed Benchmark — MobForge-Net on {device}")
    print(f"{'Batch':>8} | {'Avg ms':>10} | {'FPS':>10} | {'ms/image':>12}")
    print('-' * 48)

    for bs in batch_sizes:
        x = torch.randn(bs, 3, img_size, img_size).to(device)
        # Warm up
        for _ in range(5):
            with torch.no_grad(): _ = model(x)
        if device.type == 'cuda': torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad(): _ = model(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms    = np.mean(times)
        ms_per_im = avg_ms / bs
        fps       = 1000.0 / ms_per_im
        print(f"  {bs:>6} | {avg_ms:>10.2f} | {fps:>10.1f} | {ms_per_im:>12.2f}")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',   type=str, default=None)
    parser.add_argument('--mask',    type=str, default=None, help='Ground truth mask path for accuracy calculation')
    parser.add_argument('--weights', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--size',    type=int, default=256)
    parser.add_argument('--outdir',  type=str, default='outputs')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    if args.benchmark:
        benchmark_speed(args.weights, args.size)
    elif args.image:
        run_inference(args.image, args.weights, args.size, save_dir=args.outdir, mask_path=args.mask)
    else:
        print("Usage: python inference.py --image img.jpg --weights checkpoints/best_model.pth [--mask mask.png]")
        print("       python inference.py --benchmark")
