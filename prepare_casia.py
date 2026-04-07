"""
CASIA 2.0 Dataset Preparation Script
Downloads and organizes CASIA dataset into train/val split with images and masks.

CASIA 2.0 is available at: http://forensics.idealtest.org/

After downloading CASIA 2.0:
  1. Extract to ~/CASIA_2.0_Full/
  2. Run: python prepare_casia.py --dataset_path ~/CASIA_2.0_Full --output_dir data --train_ratio 0.7

Structure:
  CASIA_2.0_Full/
    Sp/             (spliced image)
    Tp/             (copy-move image)
    Cm/             (copy-move mask)
    NIST/           (authentic images)
    Nd/             (authentic mask - empty/all black)

Output creates:
  data/
    train/images/  and  train/masks/
    val/images/    and  val/masks/
"""

import os
import sys
import argparse
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict
import random


def load_mask_from_casia(mask_path):
    """Load CASIA ground truth mask (3-channel or 1-channel)."""
    mask = Image.open(mask_path)
    if mask.mode == 'RGB':
        # Convert RGB to grayscale, any non-black pixel = forged
        mask_arr = np.array(mask)
        mask_arr = (mask_arr.sum(axis=2) > 0).astype(np.uint8) * 255
    else:
        mask_arr = np.array(mask.convert('L'))
    return Image.fromarray(mask_arr)


def get_mask_file_casia(img_name, mask_dir, img_relpath):
    """
    Find the ground truth mask for a CASIA image.
    CASIA naming: if image is Sp/101_1.tif, mask is Sp/101_1_gt.png or similar.
    """
    stem = os.path.splitext(img_name)[0]
    
    # Try common CASIA mask naming patterns
    candidates = [
        os.path.join(mask_dir, img_name.replace('.tif', '.png')),
        os.path.join(mask_dir, f"{stem}_gt.png"),
        os.path.join(mask_dir, f"{stem}_gt.tif"),
        os.path.join(mask_dir, f"{stem}_mask.png"),
    ]
    
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    
    return None


def prepare_casia(dataset_path, output_dir, train_ratio=0.7):
    """
    Organize CASIA 2.0 dataset into train/val with images and masks.
    """
    print("\n" + "="*70)
    print("CASIA 2.0 Dataset Preparation")
    print("="*70)
    
    if not os.path.isdir(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("   Download CASIA 2.0 from: http://forensics.idealtest.org/")
        sys.exit(1)
    
    # CASIA structure - support multiple naming conventions
    sp_dir = os.path.join(dataset_path, 'Sp')      # Spliced images
    tp_dir = os.path.join(dataset_path, 'Tp')      # Copy-move images
    
    # Try different names for masks
    cm_dir = os.path.join(dataset_path, 'Cm')
    if not os.path.isdir(cm_dir):
        cm_dir = os.path.join(dataset_path, 'CASIA 2 Groundtruth')
    
    # Try different names for authentic images
    nist_dir = os.path.join(dataset_path, 'NIST')
    if not os.path.isdir(nist_dir):
        nist_dir = os.path.join(dataset_path, 'Au')
    
    nd_dir = os.path.join(dataset_path, 'Nd')      # Authentic masks (empty)
    
    # Collect all samples with their masks
    samples = []  # (image_path, mask_path, is_authentic)
    
    print("\n📂 Scanning Spliced (Sp) images...")
    sp_count = 0
    if os.path.isdir(sp_dir):
        for fname in os.listdir(sp_dir):
            if fname.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp')):
                img_path = os.path.join(sp_dir, fname)
                # Look for mask with common CASIA naming patterns
                stem = os.path.splitext(fname)[0]
                candidates = [
                    os.path.join(sp_dir, f"{stem}_gt.png"),
                    os.path.join(sp_dir, f"{stem}_gt.tif"),
                    os.path.join(sp_dir, f"{stem}_mask.png"),
                    os.path.join(sp_dir, fname.replace('.tif', '.png')),
                    os.path.join(sp_dir, fname),
                ]
                mask_path = None
                for cand in candidates:
                    if os.path.exists(cand):
                        mask_path = cand
                        break
                
                if mask_path:
                    samples.append((img_path, mask_path, False))
                    sp_count += 1
                    print(f"  ✓ {fname}")
        print(f"  Total: {sp_count} spliced images")
    
    print("\n📂 Scanning Copy-move (Tp) images...")
    tp_count = 0
    if os.path.isdir(tp_dir) and os.path.isdir(cm_dir):
        for fname in os.listdir(tp_dir):
            if fname.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp')):
                img_path = os.path.join(tp_dir, fname)
                # Look for mask in Cm/ directory with _gt suffix
                stem = os.path.splitext(fname)[0]
                # Try common CASIA mask naming patterns
                candidates = [
                    os.path.join(cm_dir, f"{stem}_gt.png"),
                    os.path.join(cm_dir, f"{stem}_gt.tif"),
                    os.path.join(cm_dir, fname.replace('.tif', '.png')),
                    os.path.join(cm_dir, fname),
                ]
                mask_path = None
                for cand in candidates:
                    if os.path.exists(cand):
                        mask_path = cand
                        break
                
                if mask_path:
                    samples.append((img_path, mask_path, False))
                    tp_count += 1
        print(f"  Total: {tp_count} copy-move images")
    
    print("\n📂 Scanning Authentic (NIST) images...")
    if os.path.isdir(nist_dir):
        for fname in os.listdir(nist_dir):
            if fname.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp')):
                img_path = os.path.join(nist_dir, fname)
                # Create empty mask (all black = authentic)
                samples.append((img_path, None, True))
        auth_count = len([s for s in samples if s[2]])
        print(f"  Total: {auth_count} authentic images")
    
    total_samples = len(samples)
    print(f"\n✓ Total samples found: {total_samples}")
    
    if total_samples == 0:
        print("❌ No images/masks found. Check your dataset structure.")
        sys.exit(1)
    
    # Create output directories
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_mask_dir = os.path.join(output_dir, 'train', 'masks')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_mask_dir = os.path.join(output_dir, 'val', 'masks')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    
    # Split into train/val (stratified by authentic vs forged)
    authentic_samples = [s for s in samples if s[2]]
    forged_samples = [s for s in samples if not s[2]]
    
    random.shuffle(authentic_samples)
    random.shuffle(forged_samples)
    
    auth_split = int(len(authentic_samples) * train_ratio)
    forge_split = int(len(forged_samples) * train_ratio)
    
    train_samples = authentic_samples[:auth_split] + forged_samples[:forge_split]
    val_samples = authentic_samples[auth_split:] + forged_samples[forge_split:]
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    print(f"\n📊 Train: {len(train_samples)} | Val: {len(val_samples)}")
    
    # Copy images and masks
    print("\n📋 Copying to train set...")
    for i, (img_path, mask_path, is_auth) in enumerate(train_samples):
        try:
            # Image
            fname = os.path.basename(img_path)
            fname = f"train_{i:06d}_{fname}"  # Rename to avoid conflicts
            shutil.copy2(img_path, os.path.join(train_img_dir, fname))
            
            # Mask
            if is_auth:
                # Create empty mask
                img = Image.open(img_path).convert('RGB')
                empty_mask = Image.new('L', img.size, 0)
                empty_mask.save(os.path.join(train_mask_dir, fname.replace('.tif', '.png').replace('.jpg', '.png').replace('.bmp', '.png').replace('.jpeg', '.png')))
            else:
                # Copy mask
                mask = load_mask_from_casia(mask_path)
                mask_fname = fname.replace('.tif', '.png').replace('.jpg', '.png').replace('.bmp', '.png').replace('.jpeg', '.png')
                mask.save(os.path.join(train_mask_dir, mask_fname))
        except Exception as e:
            print(f"  ⚠️  Skipped {fname}: {e}")
    print(f"  ✓ {len(train_samples)} samples copied")
    
    print("\n📋 Copying to val set...")
    for i, (img_path, mask_path, is_auth) in enumerate(val_samples):
        try:
            fname = os.path.basename(img_path)
            fname = f"val_{i:06d}_{fname}"
            shutil.copy2(img_path, os.path.join(val_img_dir, fname))
            
            if is_auth:
                img = Image.open(img_path).convert('RGB')
                empty_mask = Image.new('L', img.size, 0)
                empty_mask.save(os.path.join(val_mask_dir, fname.replace('.tif', '.png').replace('.jpg', '.png').replace('.bmp', '.png').replace('.jpeg', '.png')))
            else:
                mask = load_mask_from_casia(mask_path)
                mask_fname = fname.replace('.tif', '.png').replace('.jpg', '.png').replace('.bmp', '.png').replace('.jpeg', '.png')
                mask.save(os.path.join(val_mask_dir, mask_fname))
        except Exception as e:
            print(f"  ⚠️  Skipped {fname}: {e}")
    print(f"  ✓ {len(val_samples)} samples copied")
    
    print("\n" + "="*70)
    print("✓ Dataset preparation complete!")
    print(f"   Train images: {train_img_dir}")
    print(f"   Val images:   {val_img_dir}")
    print("\n🚀 Next step: python train.py --data_dir data --epochs 50")
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CASIA 2.0 dataset')
    parser.add_argument('--dataset_path', required=True,
                        help='Path to extracted CASIA_2.0_Full directory')
    parser.add_argument('--output_dir', default='data',
                        help='Output directory for train/val split')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Fraction of data for training (default: 0.7)')
    args = parser.parse_args()
    
    prepare_casia(args.dataset_path, args.output_dir, args.train_ratio)
