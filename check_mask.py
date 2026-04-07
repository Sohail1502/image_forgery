import os
from PIL import Image

mask_path = r'data\train\masks\train_002522_Tp_S_NNN_S_N_pla00099_pla00099_10618.png'

print(f"File exists: {os.path.exists(mask_path)}")
if os.path.exists(mask_path):
    fsize = os.path.getsize(mask_path)
    print(f"File size: {fsize} bytes")
    
    if fsize == 0:
        print("ERROR: File is empty (0 bytes)!")
    else:
        try:
            img = Image.open(mask_path)
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
        except Exception as e:
            print(f"Error opening: {e}")
            
            # Check file header
            with open(mask_path, 'rb') as f:
                header = f.read(20)
                print(f"Header: {header.hex()}")
else:
    print("File not found!")
