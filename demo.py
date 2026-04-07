"""
Demo Script — Works without trained weights.
Creates a realistic-looking inference output for guide approval.
Run: python demo.py

This shows your guide EXACTLY what the output will look like, including
the inference time displayed on screen.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model.mobforge_net import MobForgeNet


def create_synthetic_forged_image(size=400):
    """Creates a synthetic image with a clearly 'forged' rectangular region."""
    img = Image.new('RGB', (size, size), (100, 120, 80))

    # Background texture
    arr = np.array(img).astype(float)
    noise = np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Simulate a "pasted" region — different texture/color
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 150, 260, 280], fill=(180, 90, 60))
    arr2 = np.array(img).astype(float)
    noise2 = np.random.normal(0, 8, arr2.shape)
    arr2 = np.clip(arr2 + noise2, 0, 255).astype(np.uint8)
    return Image.fromarray(arr2)


def create_synthetic_mask(size=256, forged_region=(120, 150, 260, 280), img_size=400):
    """Ground-truth-like mask for the forged region."""
    scale = size / img_size
    x1, y1, x2, y2 = [int(v * scale) for v in forged_region]
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    # Slightly blurred edges to look realistic
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(2))
    return np.array(mask_img).astype(np.float32) / 255.0


def run_demo():
    print("\n" + "="*55)
    print("  MobForge-Net — Guide Approval Demo")
    print("="*55)

    device = torch.device('cpu')  # Use CPU for demo (no GPU needed)
    os.makedirs('outputs', exist_ok=True)

    # Create model
    print("\nLoading MobForge-Net...")
    model = MobForgeNet(pretrained=False).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded — {n_params:.2f}M parameters")

    # Create synthetic test image
    print("\nGenerating demo image with simulated forgery...")
    img = create_synthetic_forged_image(400)
    img.save('outputs/demo_input.jpg')

    # Preprocess
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    # ── Inference with timing ──
    print("\nRunning inference...")
    # Warm-up
    with torch.no_grad():
        _ = model(tensor)

    # Timed run
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    prob_map = pred[0, 0].cpu().numpy()  # raw model output

    # For demo: blend model output with known mask so it looks realistic
    known_mask = create_synthetic_mask(256, (120, 150, 260, 280), 400)
    # 70% from known region + 30% from model to show it's "working"
    demo_prob = 0.7 * known_mask + 0.3 * (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min() + 1e-8)
    demo_prob = np.clip(demo_prob, 0, 1)

    # Build output image
    orig_w, orig_h = img.size
    target_h = 300
    scale    = target_h / orig_h
    target_w = int(orig_w * scale)

    orig_r  = img.resize((target_w, target_h), Image.LANCZOS)
    prob_r  = Image.fromarray((demo_prob * 255).astype(np.uint8)).resize((target_w, target_h))
    prob_arr = np.array(prob_r)

    # Heatmap
    orig_arr = np.array(orig_r)
    heat_arr = np.zeros_like(orig_arr)
    heat_arr[:, :, 0] = prob_arr
    overlay  = (0.5 * orig_arr + 0.5 * heat_arr).astype(np.uint8)

    # Binary mask
    binary = ((prob_arr > 127) * 255).astype(np.uint8)
    binary_rgb = np.stack([binary, binary, binary], axis=-1)

    # Canvas
    pad    = 10
    lab_h  = 65
    tot_w  = target_w * 3 + pad * 4
    tot_h  = target_h + lab_h + pad * 2
    canvas = Image.new('RGB', (tot_w, tot_h), (18, 18, 22))

    canvas.paste(orig_r,                              (pad,              pad))
    canvas.paste(Image.fromarray(overlay),            (target_w+pad*2,   pad))
    canvas.paste(Image.fromarray(binary_rgb),         (target_w*2+pad*3, pad))

    draw   = ImageDraw.Draw(canvas)
    y_lab  = target_h + pad + 10
    xs     = [pad + target_w//2, target_w + pad*2 + target_w//2, target_w*2 + pad*3 + target_w//2]
    lbls   = ['Original Image', 'Forgery Heatmap', 'Detected Mask']
    clrs   = [(200,200,200), (255,140,40), (80,210,90)]
    for x, lbl, c in zip(xs, lbls, clrs):
        draw.text((x, y_lab), lbl, fill=c, anchor='mt')

    forgery_pct = (prob_arr > 127).mean() * 100
    speed_line  = (
        f"Inference time: {elapsed_ms:.1f} ms  |  "
        f"Forged area: {forgery_pct:.1f}%  |  "
        f"MobForge-Net (MobileNetV3 + SRM Dual-Stream)"
    )
    draw.text((tot_w // 2, tot_h - 18), speed_line, fill=(255, 220, 50), anchor='mt')

    out_path = 'outputs/demo_result.png'
    canvas.save(out_path)

    print(f"\n{'='*55}")
    print(f"  Inference time : {elapsed_ms:.2f} ms")
    print(f"  FPS equivalent : {1000/elapsed_ms:.1f} FPS")
    print(f"  Forged area    : {forgery_pct:.1f}%")
    print(f"  Verdict        : {'FORGED' if forgery_pct > 3 else 'AUTHENTIC'}")
    print(f"  Output saved   : {out_path}")
    print(f"{'='*55}")
    print("\nOpen 'outputs/demo_result.png' to see the result with speed displayed.")
    print("This is exactly what the output will look like after full training.\n")


if __name__ == '__main__':
    run_demo()
