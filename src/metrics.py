from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SegmentationScores:
    pixel_accuracy: float
    mean_iou: float
    mean_dice: float
    mean_precision: float
    mean_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "pixel_accuracy": self.pixel_accuracy,
            "mean_iou": self.mean_iou,
            "mean_dice": self.mean_dice,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
        }


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().long().flatten()
        targets = targets.detach().cpu().long().flatten()
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]
        if targets.numel() == 0:
            return
        idx = targets * self.num_classes + preds
        cm = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self) -> SegmentationScores:
        cm = self.matrix.float()
        tp = torch.diag(cm)
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        union = support + predicted - tp

        iou = tp / union.clamp_min(1)
        dice = 2 * tp / (support + predicted).clamp_min(1)
        precision = tp / predicted.clamp_min(1)
        recall = tp / support.clamp_min(1)
        pixel_acc = tp.sum() / cm.sum().clamp_min(1)

        present = support > 0
        if present.any():
            mean_iou = iou[present].mean().item()
            mean_dice = dice[present].mean().item()
            mean_precision = precision[present].mean().item()
            mean_recall = recall[present].mean().item()
        else:
            mean_iou = mean_dice = mean_precision = mean_recall = 0.0

        return SegmentationScores(
            pixel_accuracy=float(pixel_acc.item()),
            mean_iou=float(mean_iou),
            mean_dice=float(mean_dice),
            mean_precision=float(mean_precision),
            mean_recall=float(mean_recall),
        )

    def per_class_iou(self, class_names: list[str]) -> list[dict[str, float | str]]:
        cm = self.matrix.float()
        tp = torch.diag(cm)
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        union = support + predicted - tp
        iou = tp / union.clamp_min(1)
        rows = []
        for idx, name in enumerate(class_names):
            rows.append({"class_id": idx, "class_name": name, "iou": float(iou[idx].item()), "support_pixels": int(support[idx].item())})
        return rows


def to_numpy_mask(logits: torch.Tensor) -> np.ndarray:
    return logits.argmax(dim=1).detach().cpu().numpy()
