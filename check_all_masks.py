import os
from PIL import Image

mask_dir = r'data\train\masks'
corrupted = []
valid = 0

for fname in os.listdir(mask_dir):
    fpath = os.path.join(mask_dir, fname)
    if os.path.isfile(fpath):
        fsize = os.path.getsize(fpath)
        
        # PNG files should be > 100 bytes
        if fsize < 100:
            corrupted.append((fname, fsize))
        else:
            try:
                img = Image.open(fpath)
                img.verify()
                valid += 1
            except Exception as e:
                corrupted.append((fname, str(e)))

print(f"Valid masks: {valid}")
print(f"Corrupted/Invalid: {len(corrupted)}")
print(f"\nFirst 10 corrupted files:")
for fname, issue in corrupted[:10]:
    print(f"  {fname}: {issue}")
