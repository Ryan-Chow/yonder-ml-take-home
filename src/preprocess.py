"""The ONE preprocessing function shared by training-data prep and inference.

CLAHE (contrast-limited adaptive histogram equalisation) on the L channel in LAB space. Why (see results/audit.md):
image brightness spans almost the full range (mean L from 0 to 254; 15% of train images darker than L=80, 7%
brighter than L=170) and 13% are low-contrast (std L < 40). Outdoor shots of a small orange/blue object against
dirt and rocks lose local contrast in shadow and glare. CLAHE equalises *local* contrast while leaving hue (a/b
channels) untouched, so colour, the strongest cue for an orange mallet, is preserved. The clip limit stops it from
amplifying sensor noise in flat regions such as sky and the black rotation corners.

    python -m src.preprocess --src data/split --dst data/split_clahe
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

TRAIN_SIZE = 512  # Roboflow exported every training image as 512x512 "Stretch" (aspect ratio NOT preserved)
CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)  # 64x64-px tiles at 512x512: about the size of a median object box (~60 px)
_clahe = None


def stretch(img_bgr: np.ndarray) -> np.ndarray:
    """Resize to TRAIN_SIZE x TRAIN_SIZE ignoring aspect ratio, exactly like the Roboflow export (no-op at 512x512)."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[:2] != (TRAIN_SIZE, TRAIN_SIZE):
        img_bgr = cv2.resize(img_bgr, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_AREA)
    return img_bgr


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> BGR uint8 512x512x3: stretch-resize like the training export, then CLAHE on LAB L.

    The stretch matters for non-square inputs (e.g. 16:9 video frames): the model only ever saw stretched images,
    while Ultralytics would otherwise letterbox and preserve aspect. Normalised box coordinates are invariant to
    independent x/y scaling, so detections on the 512x512 image map 1:1 onto the original frame. For the 512x512
    dataset images the resize is a no-op.
    """
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID)
    lab = cv2.cvtColor(stretch(img_bgr), cv2.COLOR_BGR2LAB)
    lab[..., 0] = _clahe.apply(np.ascontiguousarray(lab[..., 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_dataset(src: Path, dst: Path) -> None:
    """Apply `preprocess` to every image of a YOLO dataset (src/data.yaml layout), save as lossless PNG so the
    training pixels are exactly what inference computes in memory, copy labels unchanged, write dst/data.yaml."""
    cfg = yaml.safe_load((src / "data.yaml").read_text())
    root = Path(cfg["path"])
    if dst.exists():
        shutil.rmtree(dst)
    for key, split in (("train", "train"), ("val", "valid")):
        img_dir = root / cfg[key]
        lbl_dir = img_dir.parent / "labels"
        (dst / split / "images").mkdir(parents=True)
        (dst / split / "labels").mkdir(parents=True)
        for p in sorted(img_dir.iterdir()):
            img = cv2.imread(str(p))
            if img is None:
                continue
            cv2.imwrite(str(dst / split / "images" / f"{p.stem}.png"), preprocess(img))
            shutil.copy2(lbl_dir / f"{p.stem}.txt", dst / split / "labels" / f"{p.stem}.txt")
    out = dict(cfg, path=str(dst.resolve()), train="train/images", val="valid/images")
    out.pop("test", None)
    (dst / "data.yaml").write_text(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="dataset dir containing data.yaml")
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    preprocess_dataset(a.src, a.dst)
    print(f"wrote {a.dst}")
