"""
Team Rashmi — Multi-temporal lake area tracking.

Given predictions for the same area at multiple dates, compute lake area over time
and growth rate per lake. Pairs cleanly with the GLOF risk score in inference.py.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def lake_area_per_image(pred_mask: np.ndarray, pixel_size_m: float = 30.0):
    """Total lake area (m²) and number of distinct lake instances."""
    lake = (pred_mask == 1)
    _, n = ndi.label(lake)
    area_m2 = lake.sum() * (pixel_size_m ** 2)
    return area_m2, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-by-date", nargs="+", required=True,
                    help='space-separated entries of form "YYYY-MM-DD:/path/to/pred.png"')
    ap.add_argument("--pixel-size-m", type=float, default=30.0)
    ap.add_argument("--out", default="temporal_report.json")
    args = ap.parse_args()

    entries = []
    for item in args.predictions_by_date:
        date, path = item.split(":", 1)
        pred = np.array(Image.open(path), dtype=np.int64)
        area, n = lake_area_per_image(pred, args.pixel_size_m)
        entries.append({"date": date, "path": path,
                        "lake_area_m2": float(area), "lake_count": int(n)})
    entries.sort(key=lambda e: e["date"])

    # Compute area growth rates between consecutive dates
    from datetime import datetime
    for i in range(1, len(entries)):
        d1 = datetime.fromisoformat(entries[i - 1]["date"])
        d2 = datetime.fromisoformat(entries[i]["date"])
        years = max((d2 - d1).days / 365.25, 1e-6)
        delta = entries[i]["lake_area_m2"] - entries[i - 1]["lake_area_m2"]
        entries[i]["growth_rate_m2_per_yr"] = delta / years

    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2)
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
