"""
Team Rashmi — Dataset for DeepLabV3+ with 8-channel multispectral input.

Channels: R, G, B, NIR, SWIR, NDWI, MNDWI, NDSI
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_loading import load_multispectral, load_rgb_image
from common.spectral_indices import ndwi, mndwi, ndsi


CHANNEL_NAMES = ["red", "green", "blue", "nir", "swir", "ndwi", "mndwi", "ndsi"]


def build_stack(bands: dict, n_channels: int = 8) -> np.ndarray:
    """
    Build a (C, H, W) float32 stack.
    n_channels selects the slice for ablation:
        3 -> RGB only
        4 -> RGB + NDWI
        8 -> all 8 channels (full)
    """
    if n_channels == 3:
        layers = [bands["red"], bands["green"], bands["blue"]]
    elif n_channels == 4:
        layers = [bands["red"], bands["green"], bands["blue"],
                  ndwi(bands["green"], bands["nir"])]
    elif n_channels == 8:
        layers = [bands["red"], bands["green"], bands["blue"],
                  bands["nir"], bands["swir"],
                  ndwi(bands["green"], bands["nir"]),
                  mndwi(bands["green"], bands["swir"]),
                  ndsi(bands["green"], bands["swir"])]
    else:
        raise ValueError(f"n_channels must be 3, 4, or 8 (got {n_channels})")
    return np.stack(layers, axis=0).astype(np.float32)


class MultispectralDataset(Dataset):
    def __init__(self, image_dir, label_dir, augment: bool = False,
                 image_size: int = 512, n_channels: int = 8):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.augment   = augment
        self.image_size = image_size
        self.n_channels = n_channels

        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        self.image_paths = sorted(p for p in self.image_dir.iterdir()
                                  if p.suffix.lower() in exts)
        self.image_paths = [p for p in self.image_paths
                            if (self.label_dir / f"{p.stem}.png").exists()]

    def __len__(self):
        return len(self.image_paths)

    def _load_bands(self, path: Path) -> dict:
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                return load_multispectral(path)
            except Exception:
                pass
        return load_rgb_image(path)

    def __getitem__(self, idx):
        ip = self.image_paths[idx]
        lp = self.label_dir / f"{ip.stem}.png"

        bands = self._load_bands(ip)
        from PIL import Image
        label = np.array(Image.open(lp), dtype=np.int64)
        label[label == 255] = 1

        stack = build_stack(bands, self.n_channels)
        stack, label = _resize(stack, label, self.image_size)

        if self.augment:
            stack, label = _augment(stack, label)

        return torch.from_numpy(stack), torch.from_numpy(label).long()


def _resize(stack, label, size):
    from PIL import Image
    c = stack.shape[0]
    out = np.empty((c, size, size), dtype=np.float32)
    for i in range(c):
        out[i] = np.array(
            Image.fromarray(stack[i]).resize((size, size), Image.BILINEAR),
            dtype=np.float32,
        )
    label_resized = np.array(
        Image.fromarray(label.astype(np.int32), mode="I").resize((size, size), Image.NEAREST),
        dtype=np.int64,
    )
    return out, label_resized


def _augment(stack, label):
    if np.random.rand() < 0.5:
        stack = stack[:, :, ::-1].copy(); label = label[:, ::-1].copy()
    if np.random.rand() < 0.5:
        stack = stack[:, ::-1, :].copy(); label = label[::-1, :].copy()
    k = np.random.randint(0, 4)
    if k:
        stack = np.rot90(stack, k=k, axes=(1, 2)).copy()
        label = np.rot90(label, k=k, axes=(0, 1)).copy()
    if np.random.rand() < 0.5:
        f = np.random.uniform(0.85, 1.15)
        n_raw = min(stack.shape[0], 5)
        stack[:n_raw] = np.clip(stack[:n_raw] * f, 0.0, 1.0)
    if np.random.rand() < 0.3:
        stack = stack + np.random.normal(0, 0.01, size=stack.shape).astype(np.float32)
        stack = np.clip(stack, 0.0, 1.0)
    return stack, label
