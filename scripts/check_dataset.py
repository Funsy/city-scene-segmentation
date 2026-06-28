from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_pairs, load_class_dict


def main(root: str) -> None:
    class_names, palette = load_class_dict(root)
    print(f"Classes: {len(class_names)}")
    print(", ".join(class_names[:20]) + (" ..." if len(class_names) > 20 else ""))
    for split in ["train", "val", "test"]:
        pairs = build_pairs(root, split)
        print(f"{split}: {len(pairs)} pairs")
        for img, mask in pairs[:3]:
            print(f"  {img.name} -> {mask.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/camvid")
    args = parser.parse_args()
    main(args.root)
