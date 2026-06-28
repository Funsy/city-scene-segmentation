from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import evaluate_checkpoint
from src.config import load_config
from train import train_one_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate all models from config.")
    parser.add_argument("--config", default="configs/camvid.yaml")
    parser.add_argument("--split", default="test", choices=["val", "valid", "validation", "test"])
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    model_names = args.models or list(cfg["models"].keys())
    for model_name in model_names:
        print(f"\n=== TRAIN {model_name} ===")
        checkpoint = train_one_model(args.config, model_name)
        print(f"\n=== EVALUATE {model_name} ===")
        evaluate_checkpoint(args.config, str(checkpoint), args.split)
    print(f"\nDone. Summary: {Path(cfg['outputs']['run_dir']) / 'summary.csv'}")
