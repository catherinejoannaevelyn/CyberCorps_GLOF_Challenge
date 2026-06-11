"""
Team Rashmi — Ablation runner.

A: DeepLabV3+ with RGB only (3 channels)
B: DeepLabV3+ with RGB + NDWI (4 channels)
C: DeepLabV3+ with full 8 channels
D: Full 8 channels + Monte Carlo Dropout at inference (use inference.py --mc-passes 30)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "A_rgb":      "3",
    "B_rgb_ndwi": "4",
    "C_full_8ch": "8",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", default="./ablation_runs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--train-images", args.train_images,
        "--train-labels", args.train_labels,
        "--val-images",   args.val_images,
        "--val-labels",   args.val_labels,
        "--epochs",       str(args.epochs),
        "--batch-size",   str(args.batch_size),
    ]
    for name, n_ch in VARIANTS.items():
        out_dir = out_root / name
        cmd = [sys.executable, "train.py", "--out", str(out_dir),
               "--n-channels", n_ch] + common
        print(f"\n[ABLATION] {name}\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    # Summary
    summary = {}
    for name in VARIANTS:
        p = out_root / name / "best.pt"
        if p.exists():
            import torch
            d = torch.load(p, map_location="cpu")
            summary[name] = {"best_miou": float(d.get("best_miou", -1))}
    with open(out_root / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== ABLATION SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:<14} mIoU = {v['best_miou']:.4f}")
    print("\nFor variant D (MC Dropout), use the C checkpoint with inference.py --mc-passes 30.")


if __name__ == "__main__":
    main()
