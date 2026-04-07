import os
from PIL import Image
import numpy as np

data_dirs = [
    'data/train',
    'data/val'
]
for data_dir in data_dirs:
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'masks')
    print('DATA DIR:', data_dir)
    print(' images:', len(os.listdir(img_dir)))
    print(' masks:', len(os.listdir(mask_dir)))
    missing = 0
    zero_masks = 0
    total = 0
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        total += 1
        stem, _ = os.path.splitext(fname)
        candidates = [f'{stem}.png', f'{stem}.jpg', f'{stem}.jpeg', f'{stem}_mask.png', f'{stem}_mask.jpg', f'{stem}_mask.jpeg']
        found = None
        for c in candidates:
            p = os.path.join(mask_dir, c)
            if os.path.exists(p):
                found = p
                break
        if found is None:
            missing += 1
        else:
            m = np.array(Image.open(found).convert('L'))
            if np.max(m) == 0:
                zero_masks += 1
    print(' total images:', total)
    print(' missing masks:', missing)
    print(' zero masks:', zero_masks)
    print()
