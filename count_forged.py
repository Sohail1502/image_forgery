import os
from PIL import Image
import numpy as np

train_mask_dir = r'data\train\masks'
val_mask_dir = r'data\val\masks'

def count_forged_masks(mask_dir):
    forged = 0
    authentic = 0
    for fname in os.listdir(mask_dir):
        fpath = os.path.join(mask_dir, fname)
        if os.path.isfile(fpath):
            try:
                mask = Image.open(fpath).convert('L')
                mask_arr = np.array(mask)
                if np.any(mask_arr > 0):  # Has non-zero pixels = forged
                    forged += 1
                else:  # All black = authentic
                    authentic += 1
            except:
                pass  # Skip corrupted files
    return forged, authentic

train_forged, train_auth = count_forged_masks(train_mask_dir)
val_forged, val_auth = count_forged_masks(val_mask_dir)

print(f"TRAIN SET:")
print(f"  Forged images: {train_forged}")
print(f"  Authentic images: {train_auth}")
print(f"  Total: {train_forged + train_auth}")

print(f"\nVAL SET:")
print(f"  Forged images: {val_forged}")
print(f"  Authentic images: {val_auth}")
print(f"  Total: {val_forged + val_auth}")

print(f"\nOVERALL:")
print(f"  Training forged: {train_forged}")
print(f"  Validation forged: {val_forged}")
print(f"  Total forged: {train_forged + val_forged}")
