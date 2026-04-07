import os

casia_path = r'data\CASIA2'
tp_dir = os.path.join(casia_path, 'Tp')
gt_dir = os.path.join(casia_path, 'CASIA 2 Groundtruth')

# Get all files
tp_files = sorted([f for f in os.listdir(tp_dir) if f.lower().endswith(('.tif', '.jpg', '.png', '.jpeg', '.bmp'))])
gt_files_all = sorted(os.listdir(gt_dir))  # All files, any extension
gt_stems = {}
for f in gt_files_all:
    if os.path.isfile(os.path.join(gt_dir, f)):
        stem = os.path.splitext(f)[0].replace('_gt', '')
        gt_stems[stem] = gt_stems.get(stem, 0) + 1

print(f"Total unique mask stems (without _gt): {len(gt_stems)}\n")

# Find Tp files without masks - check ALL files in GT dir
no_mask_list = []
for tp_file in tp_files:
    tp_stem = os.path.splitext(tp_file)[0]
    
    # Check if ANY file in GT matches this stem
    found = False
    for gt_file in gt_files_all:
        gt_stem = os.path.splitext(gt_file)[0].replace('_gt', '')
        if gt_stem == tp_stem:
            found = True
            break
    
    if not found:
        no_mask_list.append(tp_file)

print(f"Tp images without ANY matching file in GT: {len(no_mask_list)}\n")

if no_mask_list:
    print("First 10 unmatched Tp files:")
    for f in no_mask_list[:10]:
        print(f"  {f}")
        stem = os.path.splitext(f)[0]
        # Show what's in GT that's similar
        similar = [g for g in gt_files_all if stem[:20] in g]
        if similar:
            print(f"    Similar files in GT: {similar[:2]}")
