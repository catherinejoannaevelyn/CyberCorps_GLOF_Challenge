"""
Team Rashmi — Monte Carlo Dropout inference and GLOF risk scoring.

The MC trick:
  * Dropout layers are randomly disabled during training (regularization).
  * Normally turned OFF at inference time.
  * Here we KEEP THEM ON at inference and run T forward passes per image.
  * Average predictions = final answer.
  * Variance across T runs = uncertainty.

Cyber analogy: run the same alert past 30 different analysts and check whether
they agree. Disagreement = flag for human review.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.classes import NUM_CLASSES, colorize_mask
from common.data_loading import load_multispectral, load_rgb_image

from deeplabv3plus import DeepLabV3Plus, enable_mc_dropout
from dataset import build_stack


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DeepLabV3Plus(
        n_classes=ckpt["n_classes"],
        in_channels=ckpt["n_channels"],
        dropout=ckpt.get("dropout", 0.1),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


@torch.no_grad()
def mc_predict(model, x: torch.Tensor, n_passes: int = 30):
    """Run T stochastic forward passes. Returns (mean_probs, var_probs)."""
    enable_mc_dropout(model)
    probs = []
    for _ in range(n_passes):
        logits = model(x)
        probs.append(F.softmax(logits, dim=1))
    stacked = torch.stack(probs, dim=0)         # (T, B, C, H, W)
    return stacked.mean(dim=0), stacked.var(dim=0)


# ---------- GLOF risk scoring ----------

def compute_lake_risk(lake_mask: np.ndarray, glacier_mask: np.ndarray | None,
                      pixel_size_m: float = 30.0,
                      area_norm: float = 100_000.0,        # 0.1 km² normalisation
                      growth_norm: float = 50_000.0,       # m²/yr normalisation
                      growth_rate_m2_per_yr: float = 0.0) -> dict:
    """
    Return per-lake risk scores and a labeled mask.

    Risk = 0.4*norm_area + 0.3*norm_growth + 0.3*inv_distance_to_glacier
    """
    from scipy import ndimage as ndi

    labeled, n = ndi.label(lake_mask)
    if n == 0:
        return {"per_lake": [], "risk_mask": np.zeros_like(lake_mask, dtype=np.uint8),
                "labeled": labeled}

    # Distance from each pixel to nearest glacier pixel (in pixels)
    if glacier_mask is not None and glacier_mask.any():
        dist_px = ndi.distance_transform_edt(~glacier_mask)
    else:
        dist_px = np.full(lake_mask.shape, fill_value=1000.0)
    dist_m = dist_px * pixel_size_m

    px_area = pixel_size_m ** 2
    risk_mask = np.zeros_like(lake_mask, dtype=np.uint8)
    per_lake = []

    for i in range(1, n + 1):
        lake_i = (labeled == i)
        area_m2 = lake_i.sum() * px_area
        mean_dist = float(dist_m[lake_i].mean())

        norm_area    = min(area_m2 / area_norm, 1.0)
        norm_growth  = min(max(growth_rate_m2_per_yr, 0) / growth_norm, 1.0)
        inv_dist     = 1.0 / (1.0 + mean_dist / 1000.0)  # km

        score = 0.4 * norm_area + 0.3 * norm_growth + 0.3 * inv_dist
        if score < 0.25:    risk_level, risk_id = "LOW", 1
        elif score < 0.50:  risk_level, risk_id = "MEDIUM", 2
        elif score < 0.75:  risk_level, risk_id = "HIGH", 3
        else:               risk_level, risk_id = "CRITICAL", 4

        risk_mask[lake_i] = risk_id
        per_lake.append({
            "lake_id": int(i),
            "area_m2": float(area_m2),
            "area_km2": float(area_m2 / 1e6),
            "mean_distance_to_glacier_m": mean_dist,
            "growth_rate_m2_per_yr": float(growth_rate_m2_per_yr),
            "risk_score": float(score),
            "risk_level": risk_level,
        })
    return {"per_lake": per_lake, "risk_mask": risk_mask, "labeled": labeled}


# ---------- Pipeline ----------

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.checkpoint, device)
    n_channels = ckpt["n_channels"]

    out_dir = Path(args.out)
    for sub in ("predictions", "visualizations", "uncertainty", "risk", "reports"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.images)
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})

    from PIL import Image
    for ip in paths:
        try:
            bands = load_multispectral(ip) if ip.suffix.lower() in {".tif", ".tiff"} else load_rgb_image(ip)
        except Exception:
            bands = load_rgb_image(ip)
        stack = build_stack(bands, n_channels=n_channels)
        h, w = stack.shape[1], stack.shape[2]

        # Resize to model input size
        x_t = torch.from_numpy(stack).unsqueeze(0)
        x_t = F.interpolate(x_t, size=(args.input_size, args.input_size),
                            mode="bilinear", align_corners=False).to(device)

        if args.mc_passes > 1:
            mean_probs, var_probs = mc_predict(model, x_t, n_passes=args.mc_passes)
            uncertainty = var_probs.sum(dim=1)  # total variance
        else:
            with torch.no_grad():
                model.eval()
                logits = model(x_t)
                mean_probs = F.softmax(logits, dim=1)
                uncertainty = 1.0 - mean_probs.max(dim=1).values

        pred_small = mean_probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        unc_small  = uncertainty[0].cpu().numpy()
        unc_small  = (unc_small - unc_small.min()) / (unc_small.max() - unc_small.min() + 1e-7)

        # Resize back to original
        pred = np.array(Image.fromarray(pred_small).resize((w, h), Image.NEAREST))
        unc  = np.array(Image.fromarray((unc_small * 255).astype(np.uint8))
                        .resize((w, h), Image.BILINEAR))

        # Save predictions / viz
        Image.fromarray(pred).save(out_dir / "predictions" / f"{ip.stem}.png")
        Image.fromarray(colorize_mask(pred)).save(out_dir / "visualizations" / f"{ip.stem}.png")
        Image.fromarray(unc).save(out_dir / "uncertainty" / f"{ip.stem}.png")

        # GLOF risk
        lake_mask = (pred == 1)
        glacier_mask = np.isin(pred, [3, 4])  # clean ice + debris ice as the glacier
        risk = compute_lake_risk(lake_mask, glacier_mask)
        Image.fromarray(_risk_to_rgb(risk["risk_mask"])).save(out_dir / "risk" / f"{ip.stem}.png")
        with open(out_dir / "reports" / f"{ip.stem}.json", "w") as f:
            json.dump(risk["per_lake"], f, indent=2)

    print(f"Wrote {len(paths)} predictions to {out_dir}")


def _risk_to_rgb(risk_mask: np.ndarray) -> np.ndarray:
    """Color the 4-level risk mask: green/yellow/orange/red."""
    palette = {
        0: (0, 0, 0),         # not a lake
        1: (0, 200, 0),       # LOW
        2: (255, 220, 0),     # MEDIUM
        3: (255, 140, 0),     # HIGH
        4: (220, 0, 0),       # CRITICAL
    }
    out = np.zeros((*risk_mask.shape, 3), dtype=np.uint8)
    for k, c in palette.items():
        out[risk_mask == k] = c
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="./inference_out")
    ap.add_argument("--mc-passes", type=int, default=30,
                    help="Number of Monte Carlo forward passes (>=1; 1 = no MC)")
    ap.add_argument("--input-size", type=int, default=512)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
