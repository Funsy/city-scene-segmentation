from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_device, load_config
from src.data import CamVidSegmentationDataset, load_class_dict
from src.metrics import ConfusionMatrix
from src.models import create_model
from src.utils import checkpoint_size_mb, ensure_dir, save_json, set_seed


def make_loaders(cfg: dict, palette):
    data_cfg = cfg["data"]
    train_ds = CamVidSegmentationDataset(
        root=data_cfg["root"],
        split="train",
        image_size=int(data_cfg["image_size"]),
        class_palette=palette,
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        train=True,
        limit=data_cfg.get("limit_train_samples"),
    )
    val_ds = CamVidSegmentationDataset(
        root=data_cfg["root"],
        split="val",
        image_size=int(data_cfg["image_size"]),
        class_palette=palette,
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        train=False,
        limit=data_cfg.get("limit_val_samples"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def validate(model, loader, criterion, device: str, num_classes: int, ignore_index: int) -> dict[str, float]:
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    losses = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        cm.update(preds, masks)
    scores = cm.compute().as_dict()
    scores["loss"] = sum(losses) / max(1, len(losses))
    return scores


def train_one_model(config_path: str, model_name: str) -> Path:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "auto")))
    class_names, palette = load_class_dict(cfg["data"]["root"])
    num_classes = len(class_names)
    ignore_index = int(cfg["data"].get("ignore_index", 255))

    if model_name not in cfg["models"]:
        raise KeyError(f"Unknown model '{model_name}'. Available: {', '.join(cfg['models'].keys())}")

    run_dir = ensure_dir(Path(cfg["outputs"]["run_dir"]) / model_name)
    save_json({"class_names": class_names, "palette": palette.tolist()}, run_dir / "classes.json")

    train_loader, val_loader = make_loaders(cfg, palette)
    model = create_model(cfg["models"][model_name], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["training"].get("amp", True)) and device == "cuda")

    best_miou = -1.0
    bad_epochs = 0
    rows: list[dict[str, float | int | str]] = []
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("early_stopping_patience", 5))

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{sum(train_losses) / len(train_losses):.4f}")

        val_scores = validate(model, val_loader, criterion, device, num_classes, ignore_index)
        train_loss = sum(train_losses) / max(1, len(train_losses))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_scores.items()},
            "epoch_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(run_dir / "history.csv", index=False)
        print(row)

        if val_scores["mean_iou"] > best_miou:
            best_miou = val_scores["mean_iou"]
            bad_epochs = 0
            checkpoint = {
                "model_name": model_name,
                "model_cfg": cfg["models"][model_name],
                "num_classes": num_classes,
                "class_names": class_names,
                "palette": palette.tolist(),
                "image_size": int(cfg["data"]["image_size"]),
                "ignore_index": ignore_index,
                "state_dict": model.state_dict(),
                "best_val_mean_iou": best_miou,
            }
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {bad_epochs} non-improving epochs.")
                break

    metrics = {
        "model_name": model_name,
        "best_val_mean_iou": best_miou,
        "checkpoint_mb": checkpoint_size_mb(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None,
        "device": device,
        "epochs_requested": epochs,
        "epochs_finished": len(rows),
        "image_size": int(cfg["data"]["image_size"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
    }
    save_json(metrics, run_dir / "train_metrics.json")
    return run_dir / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one semantic segmentation model.")
    parser.add_argument("--config", default="configs/camvid.yaml")
    parser.add_argument("--model", required=True, help="Model name from config, e.g. unet_resnet18")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_model(args.config, args.model)
