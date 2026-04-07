import os

casia_path = r'data\CASIA2'
tp_dir = os.path.join(casia_path, 'Tp')
gt_dir = os.path.join(casia_path, 'CASIA 2 Groundtruth')

# Count real files (not subdirs)
au_real = len([f for f in os.listdir(os.path.join(casia_path, 'Au')) if os.path.isfile(os.path.join(casia_path, 'Au', f))])
tp_real = len([f for f in os.listdir(tp_dir) if os.path.isfile(os.path.join(tp_dir, f))])
gt_real = len([f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))])

# Count image files only
tp_images = [f for f in os.listdir(tp_dir) if f.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp')) and os.path.isfile(os.path.join(tp_dir, f))]
gt_images = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp')) and os.path.isfile(os.path.join(gt_dir, f))]

print("=== FILE COUNTS ===")
print(f"Au total files: {au_real}")
print(f"Tp total files: {tp_real}")
print(f"CASIA 2 Groundtruth total files: {gt_real}")
print(f"\nTp image files: {len(tp_images)}")
print(f"GT image files: {len(gt_images)}")
print(f"\nTp/GT ratio: {len(tp_images)}/{len(gt_images)} -> DIFFERENCE: {len(tp_images) - len(gt_images)}")
print(f"\nDataset prepared with: 4981 forged + 7491 authentic = 12472 total")
print(f"Expected with all files: {len(tp_images)} forged + {au_real} authentic = {len(tp_images) + au_real}")
