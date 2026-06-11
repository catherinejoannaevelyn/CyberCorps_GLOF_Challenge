"""
Team Rashmi — Training DeepLabV3+ with 8-channel multispectral input.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.classes import NUM_CLASSES, CLASS_NAMES
from common.metrics import full_report, print_report

from deeplabv3plus import DeepLabV3Plus
from dataset import MultispectralDataset, CHANNEL_NAMES


# Reuse the loss from Team Goban — keeps both teams comparable.
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__(); self.smooth = smooth

    def forward(self, logits, targets):
        n = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        oh = nn.functional.one_hot(targets, n).permute(0, 3, 1, 2).float()
        inter = (probs * oh).sum(dim=(0, 2, 3))
        card  = probs.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.ce(logits, targets) + 0.5 * self.dice(logits, targets)


def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            yt.append(y.numpy()); yp.append(pred)
    yt = np.concatenate([a.flatten() for a in yt])
    yp = np.concatenate([a.flatten() for a in yp])
    return full_report(yt, yp, NUM_CLASSES, CLASS_NAMES)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  channels: {args.n_channels}")

    train_ds = MultispectralDataset(args.train_images, args.train_labels,
                                    augment=True, image_size=args.image_size,
                                    n_channels=args.n_channels)
    val_ds   = MultispectralDataset(args.val_images, args.val_labels,
                                    augment=False, image_size=args.image_size,
                                    n_channels=args.n_channels)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    model = DeepLabV3Plus(n_classes=NUM_CLASSES,
                          in_channels=args.n_channels,
                          dropout=args.dropout,
                          pretrained=True).to(device)

    cw = torch.tensor([0.5, 3.0, 1.5, 1.5, 1.5, 1.0], dtype=torch.float32).to(device)
    criterion = CombinedLoss(class_weights=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / max(len(train_ds), 1)

        report = evaluate(model, val_loader, device)
        miou = report["mIoU"]
        print(f"epoch {epoch:3d} | train_loss={train_loss:.4f} "
              f"| val_mIoU={miou:.4f} | val_kappa={report['cohens_kappa']:.4f}")

        if miou > best_miou:
            best_miou = miou
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "n_channels": args.n_channels,
                "n_classes": NUM_CLASSES,
                "channel_names": CHANNEL_NAMES[:args.n_channels] if args.n_channels in (3, 8)
                                 else CHANNEL_NAMES[:3] + ["ndwi"],
                "best_miou": best_miou,
                "dropout": args.dropout,
            }, out_dir / "best.pt")
            print(f"  -> saved best (mIoU={best_miou:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping (best mIoU={best_miou:.4f})")
                break

    print("\nFinal validation metrics with best checkpoint:")
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print_report(evaluate(model, val_loader, device))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-channels", type=int, default=8, choices=[3, 4, 8],
                    help="3=RGB only, 4=RGB+NDWI, 8=full")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
