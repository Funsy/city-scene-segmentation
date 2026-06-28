from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

PALETTE = {
    "Sky": (128, 128, 128),
    "Building": (128, 0, 0),
    "Road": (128, 64, 128),
    "Car": (64, 0, 128),
}


def make_sample(idx: int, width: int = 256, height: int = 256):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky
    image[: height // 2] = (140 + idx % 30, 180, 220)
    mask[: height // 2] = PALETTE["Sky"]

    # Building blocks
    image[40:150, 25:95] = (90, 90, 100)
    image[30:150, 150:230] = (100, 95, 90)
    mask[40:150, 25:95] = PALETTE["Building"]
    mask[30:150, 150:230] = PALETTE["Building"]

    # Road
    image[height // 2 :] = (70, 70, 70)
    mask[height // 2 :] = PALETTE["Road"]

    # Car rectangle
    x = 30 + (idx * 13) % 120
    y = 165 + (idx * 7) % 45
    image[y : y + 30, x : x + 65] = (180, 30 + idx % 100, 40)
    mask[y : y + 30, x : x + 65] = PALETTE["Car"]

    # Lane line
    cv2.line(image, (120, 255), (150, 130), (230, 230, 230), 3)
    return image, mask


def main(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "class_dict.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "r", "g", "b"])
        for name, rgb in PALETTE.items():
            writer.writerow([name, *rgb])

    counts = {"train": 10, "val": 4, "test": 4}
    offset = 0
    for split, count in counts.items():
        image_dir = root / split
        mask_dir = root / f"{split}_labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            image, mask = make_sample(offset + i)
            cv2.imwrite(str(image_dir / f"sample_{i:03d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_dir / f"sample_{i:03d}_L.png"), cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        offset += count
    print(f"Sample dataset created at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/sample_city")
    args = parser.parse_args()
    main(Path(args.root))
