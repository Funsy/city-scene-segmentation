from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


DEFAULT_CAMVID_CLASSES = [
    ("Animal", (64, 128, 64)),
    ("Archway", (192, 0, 128)),
    ("Bicyclist", (0, 128, 192)),
    ("Bridge", (0, 128, 64)),
    ("Building", (128, 0, 0)),
    ("Car", (64, 0, 128)),
    ("CartLuggagePram", (64, 0, 192)),
    ("Child", (192, 128, 64)),
    ("Column_Pole", (192, 192, 128)),
    ("Fence", (64, 64, 128)),
    ("LaneMkgsDriv", (128, 0, 192)),
    ("LaneMkgsNonDriv", (192, 0, 64)),
    ("Misc_Text", (128, 128, 64)),
    ("MotorcycleScooter", (192, 0, 192)),
    ("OtherMoving", (128, 64, 64)),
    ("ParkingBlock", (64, 192, 128)),
    ("Pedestrian", (64, 64, 0)),
    ("Road", (128, 64, 128)),
    ("RoadShoulder", (128, 128, 192)),
    ("Sidewalk", (0, 0, 192)),
    ("SignSymbol", (192, 128, 128)),
    ("Sky", (128, 128, 128)),
    ("SUVPickupTruck", (64, 128, 192)),
    ("TrafficCone", (0, 0, 64)),
    ("TrafficLight", (0, 64, 64)),
    ("Train", (192, 64, 128)),
    ("Tree", (128, 128, 0)),
    ("Truck_Bus", (192, 128, 192)),
    ("Tunnel", (64, 0, 64)),
    ("VegetationMisc", (192, 192, 0)),
    ("Void", (0, 0, 0)),
    ("Wall", (64, 192, 0)),
]


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def _find_existing_folder(root: Path, candidates: Iterable[str]) -> Path | None:
    for item in candidates:
        path = root / item
        if path.exists() and path.is_dir():
            return path
    return None


def find_split_dirs(root: str | Path, split: str) -> tuple[Path, Path]:
    """Find image and mask folders for common CamVid layouts.

    Supported examples:
    - data/camvid/train and data/camvid/train_labels
    - data/camvid/images/train and data/camvid/masks/train
    - data/camvid/images/train and data/camvid/annotations/train
    """
    root = Path(root)
    aliases = {
        "val": ["val", "valid", "validation"],
        "valid": ["val", "valid", "validation"],
        "validation": ["val", "valid", "validation"],
        "train": ["train"],
        "test": ["test"],
    }
    split_names = aliases.get(split, [split])

    image_candidates: list[str] = []
    mask_candidates: list[str] = []
    for s in split_names:
        image_candidates.extend([
            s,
            f"images/{s}",
            f"imgs/{s}",
            f"image/{s}",
            f"data/{s}",
        ])
        mask_candidates.extend([
            f"{s}_labels",
            f"{s}_label",
            f"{s}annot",
            f"{s}_annot",
            f"labels/{s}",
            f"masks/{s}",
            f"annotations/{s}",
            f"ann/{s}",
            f"mask/{s}",
        ])

    image_dir = _find_existing_folder(root, image_candidates)
    mask_dir = _find_existing_folder(root, mask_candidates)

    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(
            "Could not find image/mask folders for split "
            f"'{split}' under {root}. See README.md for supported layouts."
        )
    return image_dir, mask_dir


def normalize_stem(stem: str) -> str:
    for suffix in ["_L", "_label", "_labels", "_mask", "_annot", "_gt"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def build_pairs(root: str | Path, split: str, limit: int | None = None) -> list[tuple[Path, Path]]:
    image_dir, mask_dir = find_split_dirs(root, split)
    images = _list_images(image_dir)
    masks = _list_images(mask_dir)

    mask_by_stem = {normalize_stem(m.stem): m for m in masks}
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for img in images:
        key = normalize_stem(img.stem)
        mask = mask_by_stem.get(key)
        if mask is None:
            missing.append(img.name)
            continue
        pairs.append((img, mask))

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found for split '{split}'. "
            f"Images: {image_dir}, masks: {mask_dir}."
        )
    if missing:
        print(f"Warning: {len(missing)} images without masks were skipped for split '{split}'.")
    if limit is not None:
        pairs = pairs[: int(limit)]
    return pairs


def load_class_dict(root: str | Path) -> tuple[list[str], np.ndarray]:
    """Load class names and RGB colors.

    CamVid Kaggle usually contains class_dict.csv with columns name,r,g,b.
    If it is absent, fallback to the standard 32-class CamVid palette.
    """
    root = Path(root)
    csv_path = root / "class_dict.csv"
    names: list[str] = []
    colors: list[tuple[int, int, int]] = []

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            normalized = [{k.strip().lower(): v for k, v in row.items()} for row in reader]
        for row in normalized:
            name = row.get("name") or row.get("class") or row.get("classname")
            if name is None:
                raise ValueError("class_dict.csv must contain a name/class column")
            r = int(row.get("r", row.get("red", 0)))
            g = int(row.get("g", row.get("green", 0)))
            b = int(row.get("b", row.get("blue", 0)))
            names.append(str(name))
            colors.append((r, g, b))
    else:
        names = [name for name, _ in DEFAULT_CAMVID_CLASSES]
        colors = [rgb for _, rgb in DEFAULT_CAMVID_CLASSES]

    return names, np.array(colors, dtype=np.uint8)


def rgb_mask_to_class(mask_rgb: np.ndarray, palette: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert an RGB segmentation mask to class-index mask."""
    h, w, _ = mask_rgb.shape
    result = np.full((h, w), ignore_index, dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        matches = np.all(mask_rgb == color.reshape(1, 1, 3), axis=-1)
        result[matches] = class_idx
    return result


def class_mask_to_rgb(mask: np.ndarray, palette: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert class-index mask to RGB image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        out[mask == idx] = color
    out[mask == ignore_index] = (0, 0, 0)
    return out


def get_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                shear=0,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=255,
                p=0.3,
            ),
        ]
    )


def get_eval_transform(image_size: int) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)])


class CamVidSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        class_palette: np.ndarray,
        ignore_index: int = 255,
        train: bool = False,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.pairs = build_pairs(self.root, split, limit)
        self.palette = class_palette
        self.ignore_index = ignore_index
        self.transform = get_train_transform(image_size) if train else get_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask_bgr is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class(mask_rgb, self.palette, self.ignore_index)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.int64)

        # ImageNet normalization.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
