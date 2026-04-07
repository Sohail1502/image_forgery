

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from model.mobforge_net import MobForgeNet, BoundaryAwareLoss


# =========================
# Dataset
# =========================
class ForgeryDataset(Dataset):
    def __init__(self, data_dir, img_size=256, augment=True):
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.img_size = img_size

        self.files = [
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Image
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')

        # Mask path resolution: support different mask extensions and `_mask` suffixes.
        mask_path = os.path.join(self.mask_dir, fname)
        if not os.path.exists(mask_path):
            stem, _ = os.path.splitext(fname)
            candidates = [
                f'{stem}.png', f'{stem}.jpg', f'{stem}.jpeg',
                f'{stem}_mask.png', f'{stem}_mask.jpg', f'{stem}_mask.jpeg'
            ]
            mask_path = None
            for candidate in candidates:
                candidate_path = os.path.join(self.mask_dir, candidate)
                if os.path.exists(candidate_path):
                    mask_path = candidate_path
                    break

        if mask_path and os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')
                # Verify the mask is valid (not corrupted)
                mask.load()
            except Exception as e:
                # If mask is corrupted, create empty mask
                print(f"⚠️  Corrupted mask, skipping: {mask_path}")
                mask = Image.fromarray(
                    np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
                )
        else:
            mask = Image.fromarray(
                np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
            )

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.augment and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Transform
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return img, mask, fname


# =========================
# Train
# =========================
def train(args):
    print("🔥 Training started...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Model
    model = MobForgeNet(pretrained=True).to(device)
    print("Model loaded")

    # Dataset
    train_ds = ForgeryDataset(os.path.join(args.data_dir, 'train'))
    val_ds = ForgeryDataset(os.path.join(args.data_dir, 'val'), augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Loss + optimizer
    criterion = BoundaryAwareLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_metrics = {'f1': 0, 'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'accuracy': 0, 'train_loss': float('inf'), 'val_loss': float('inf')}
    best_epoch = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_pixels = 0

        for imgs, masks, _ in tqdm(train_dl):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            # Calculate accuracy
            pred_binary = (preds > 0.5).float()
            correct = (pred_binary == masks).sum().item()
            pixels = masks.numel()
            total_correct += correct
            total_pixels += pixels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dl)
        avg_train_acc = (total_correct / total_pixels) * 100 if total_pixels > 0 else 0

        # Validation
        model.eval()
        total_val_loss = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)
                total_val_loss += loss.item()

                pred_binary = (preds > 0.5).float()
                tp += ((pred_binary == 1) & (masks == 1)).sum().item()
                fp += ((pred_binary == 1) & (masks == 0)).sum().item()
                tn += ((pred_binary == 0) & (masks == 0)).sum().item()
                fn += ((pred_binary == 0) & (masks == 1)).sum().item()

        avg_val_loss = total_val_loss / len(val_dl)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
        dice = f1
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_acc:.2f}% - Val Loss: {avg_val_loss:.4f} - F1: {f1:.4f} - IoU: {iou:.4f}")

        # Save best model based on F1
        if f1 > best_metrics['f1']:
            best_metrics = {'f1': f1, 'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
            best_epoch = epoch + 1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("💾 Best model saved!")

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Best F1 Score: {best_metrics['f1']:.4f}")
    print(f"Best IoU:      {best_metrics['iou']:.4f}")
    print(f"Best Dice:     {best_metrics['dice']:.4f}")
    print(f"Best Precision: {best_metrics['precision']:.4f}")
    print(f"Best Recall:   {best_metrics['recall']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Final Train Loss: {best_metrics['train_loss']:.4f}")
    print(f"Final Val Loss:   {best_metrics['val_loss']:.4f}")
    print("="*60)
    print("Training complete.")


# =========================
# Evaluate
# =========================
def evaluate_model(args):
    print("🔍 Evaluating model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on:", device)

    # Model
    model = MobForgeNet(pretrained=False).to(device)
    weights_path = "checkpoints/best_model.pth"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights: {weights_path}")
    else:
        print(f"No weights found at {weights_path}. Exiting.")
        return
    model.eval()

    # Dataset
    val_ds = ForgeryDataset(os.path.join(args.data_dir, 'val'), augment=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Validation set: {len(val_ds)} samples")

    # Evaluation
    total_val_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    criterion = BoundaryAwareLoss()

    with torch.no_grad():
        for imgs, masks, _ in tqdm(val_dl):
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)
            total_val_loss += loss.item()

            pred_binary = (preds > 0.5).float()
            tp += ((pred_binary == 1) & (masks == 1)).sum().item()
            fp += ((pred_binary == 1) & (masks == 0)).sum().item()
            tn += ((pred_binary == 0) & (masks == 0)).sum().item()
            fn += ((pred_binary == 0) & (masks == 1)).sum().item()

    avg_val_loss = total_val_loss / len(val_dl)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
    dice = f1
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"IoU:             {iou:.4f}")
    print(f"Dice:            {dice:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print("="*60)
    print("Evaluation complete.")


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--load_model", action="store_true", help="Load existing model and evaluate on validation set")

    args = parser.parse_args()

    if args.load_model:
        evaluate_model(args)
    else:
        train(args)