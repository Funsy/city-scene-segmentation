from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp


def create_model(model_cfg: dict[str, Any], num_classes: int):
    architecture = model_cfg["architecture"]
    encoder_name = model_cfg.get("encoder_name", "resnet18")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    if encoder_weights in ["null", "None", "none", ""]:
        encoder_weights = None

    model_cls = getattr(smp, architecture, None)
    if model_cls is None:
        available = [name for name in dir(smp) if name and name[0].isupper()]
        raise ValueError(f"Unknown SMP architecture '{architecture}'. Available examples: {available[:30]}")

    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
