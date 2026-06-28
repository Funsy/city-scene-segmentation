from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_device, load_config
from src.data import CamVidSegmentationDataset, class_mask_to_rgb
from src.metrics import ConfusionMatrix
from src.models import create_model
from src.utils import checkpoint_size_mb, ensure_dir, save_json


@torch.no_grad()
def evaluate_checkpoint(config_path: str, checkpoint_path: str, split: str = "test") -> dict:
    cfg = load_config(config_path)
    device = get_device(str(cfg.get("device", "auto")))
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    palette = torch.tensor(checkpoint["palette"], dtype=torch.uint8).numpy()
    num_classes = int(checkpoint["num_classes"])
    ignore_index = int(checkpoint.get("ignore_index", cfg["data"].get("ignore_index", 255)))

    model = create_model(checkpoint["model_cfg"], num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    data_cfg = cfg["data"]
    limit_key = f"limit_{split}_samples"
    ds = CamVidSegmentationDataset(
        root=data_cfg["root"],
        split=split,
        image_size=int(checkpoint.get("image_size", data_cfg["image_size"])),
        class_palette=palette,
        ignore_index=ignore_index,
        train=False,
        limit=data_cfg.get(limit_key),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    times = []
    total_images = 0
    for batch in tqdm(loader, desc=f"evaluate {checkpoint['model_name']} on {split}"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(images)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        times.append(elapsed / images.shape[0])
        preds = logits.argmax(dim=1)
        cm.update(preds, masks)
        total_images += images.shape[0]

    scores = cm.compute().as_dict()
    avg_time = sum(times) / max(1, len(times))
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    result = {
        "model_name": checkpoint["model_name"],
        "split": split,
        "num_images": total_images,
        **scores,
        "avg_inference_time_sec": avg_time,
        "fps": fps,
        "checkpoint_mb": checkpoint_size_mb(checkpoint_path),
        "device": device,
        "image_size": int(checkpoint.get("image_size", data_cfg["image_size"])),
    }
    result["per_class_iou"] = cm.per_class_iou(class_names)

    run_dir = checkpoint_path.parent
    save_json(result, run_dir / f"{split}_metrics.json")
    pd.DataFrame(result["per_class_iou"]).to_csv(run_dir / f"{split}_per_class_iou.csv", index=False)

    # Update summary table.
    summary_path = Path(cfg["outputs"]["run_dir"]) / "summary.csv"
    flat = {k: v for k, v in result.items() if k != "per_class_iou"}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[~((summary["model_name"] == flat["model_name"]) & (summary["split"] == flat["split"]))]
        summary = pd.concat([summary, pd.DataFrame([flat])], ignore_index=True)
    else:
        summary = pd.DataFrame([flat])
    summary.to_csv(summary_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a semantic segmentation checkpoint.")
    parser.add_argument("--config", default="configs/camvid.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "validation", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_checkpoint(args.config, args.checkpoint, args.split)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class_iou"}, ensure_ascii=False, indent=2))
