import os

casia_path = r'data\CASIA2'
tp_dir = os.path.join(casia_path, 'Tp')
gt_dir = os.path.join(casia_path, 'CASIA 2 Groundtruth')

# Get all files
tp_files = sorted([f for f in os.listdir(tp_dir) if f.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp'))])
gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp'))])

print(f"Total Tp images: {len(tp_files)}")
print(f"Total GT masks: {len(gt_files)}\n")

# Count matches
tp_with_mask = 0
no_mask_list = []

for tp_file in tp_files:
    stem = os.path.splitext(tp_file)[0]
    # Try candidates
    candidates = [
        f"{stem}_gt.png",
        f"{stem}_gt.tif",
        f"{stem}_mask.png",
        tp_file.replace('.tif', '.png').replace('.jpg', '.png').replace('.bmp', '.png'),
    ]
    
    found = False
    for cand in candidates:
        if cand in gt_files:
            tp_with_mask += 1
            found = True
            break
    
    if not found:
        no_mask_list.append((tp_file, candidates))

print(f"Tp images WITH matching mask: {tp_with_mask}")
print(f"Tp images WITHOUT matching mask: {len(no_mask_list)}\n")

if no_mask_list:
    print("First 10 Tp files without masks:")
    for tp_file, candidates in no_mask_list[:10]:
        print(f"  ✗ {tp_file}")
        print(f"      Looking for: {candidates}")
