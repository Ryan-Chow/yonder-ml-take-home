"""Train YOLOv8n for the ablation. All runs share configs/train.yaml `common`; only data and augmentation differ.

    baseline : grouped split, raw images,   Ultralytics default augmentation
    final    : grouped split, CLAHE images, configs/train.yaml `final_aug`      -> copied to weights/best.pt
    leak     : ORIGINAL Roboflow split, CLAHE images, `final_aug`  (only to measure leakage inflation)

    python scripts/train.py --run baseline [--device mps] [--epochs N] [--fraction F] [--name NAME]
    python scripts/train.py --run final --smoke        # 1 epoch on ~20 train images, before any full run

Writes runs/<name>/ (git-ignored). Run from the repo root.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "baseline": {"data": "data/split/data.yaml", "aug": False},
    "final": {"data": "data/split_clahe/data.yaml", "aug": True},
    "leak": {"data": "data/orig_clahe/data.yaml", "aug": True},
}


def build_args(run: str, device: str, epochs: int | None, fraction: float, name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    args = dict(cfg["common"])
    if RUNS[run]["aug"]:
        args.update(cfg["final_aug"])
    args.update(data=str(ROOT / RUNS[run]["data"]), device=device, project=str(ROOT / "runs"), name=name,
                exist_ok=True, fraction=fraction)
    if epochs is not None:
        args["epochs"] = epochs
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=RUNS, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the train set to use")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--name")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on ~20 train images")
    a = ap.parse_args()

    if a.smoke:
        a.epochs, a.fraction = 1, 20 / 800
    name = a.name or (f"smoke_{a.run}" if a.smoke else a.run)
    data = ROOT / RUNS[a.run]["data"]
    if not data.exists():
        sys.exit(f"{data} missing: run scripts/make_split.py and src/preprocess.py first")

    from ultralytics import YOLO

    args = build_args(a.run, a.device, a.epochs, a.fraction, name)
    if a.batch:
        args["batch"] = a.batch
    model_path = args.pop("model")
    pretrained = ROOT / "data" / "pretrained" / model_path
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    if not pretrained.exists():  # Ultralytics downloads the COCO checkpoint into the cwd; keep it out of git
        YOLO(model_path)
        shutil.move(model_path, pretrained)
    print("train args:", {k: v for k, v in args.items()})
    t0 = time.time()
    model = YOLO(str(pretrained))
    model.train(**args)
    dt = time.time() - t0
    best = ROOT / "runs" / name / "weights" / "best.pt"
    print(f"done in {dt / 60:.1f} min; best weights: {best}")
    if a.run == "final" and not a.smoke:
        (ROOT / "weights").mkdir(exist_ok=True)
        shutil.copy2(best, ROOT / "weights" / "best.pt")
        print("copied to weights/best.pt")


if __name__ == "__main__":
    main()
