"""Run the trained detector on a folder of images.

    python predict.py <input_dir> <output_dir> [--weights weights/best.pt] [--conf T] [--save-vis] [--device cpu]

For every image in <input_dir> writes <output_dir>/<same_stem>.txt, one detection per line:

    class_id x_center y_center width height confidence

coordinates normalised to 0-1 like the dataset labels; class ids 0 = bottle, 1 = mallet. An image with no
detections gets an empty file. Images go through src/preprocess.py (CLAHE), the same function used to build the
training data. Default --conf is read from results/threshold.json (the max-mean-F1 threshold on val).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_utils import CLASS_NAMES, IMG_EXTS  # noqa: E402
from src.infer import detect, load_model  # noqa: E402

COLORS = {0: (214, 120, 42), 1: (52, 104, 235)}  # BGR: bottle blue, mallet orange


def default_conf() -> float:
    f = ROOT / "results" / "threshold.json"
    return float(json.loads(f.read_text())["conf"]) if f.exists() else 0.25


def draw(img, dets):
    h, w = img.shape[:2]
    for c, xc, yc, bw, bh, conf in dets:
        x1, y1, x2, y2 = int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[int(c)], 2)
        cv2.putText(img, f"{CLASS_NAMES[int(c)]} {conf:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS[int(c)], 2)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    ap.add_argument("--conf", type=float, default=None, help="confidence threshold (default: results/threshold.json)")
    ap.add_argument("--save-vis", action="store_true", help="also write <stem>.jpg with boxes drawn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)

    conf = a.conf if a.conf is not None else default_conf()
    images = sorted(p for p in a.input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"no images found in {a.input_dir}", file=sys.stderr)
        return 1
    stems = [p.stem for p in images]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:  # a.jpg and a.png would both write a.txt
        print(f"error: images share a stem, outputs would collide: {dupes[:5]}", file=sys.stderr)
        return 1
    a.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(a.weights)
    n_det = 0
    for k in range(0, len(images), a.batch):
        paths = images[k:k + a.batch]
        imgs, ok = [], []
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                print(f"warning: could not read {p.name}, writing empty file", file=sys.stderr)
                (a.output_dir / f"{p.stem}.txt").write_text("")
                continue
            imgs.append(im)
            ok.append(p)
        for p, im, dets in zip(ok, imgs, detect(model, imgs, conf=conf, clahe=True, device=a.device)):
            lines = [f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {s:.4f}" for c, xc, yc, w, h, s in dets]
            (a.output_dir / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_det += len(lines)
            if a.save_vis:
                cv2.imwrite(str(a.output_dir / f"{p.stem}.jpg"), draw(im, dets))
    print(f"{len(images)} images -> {a.output_dir} ({n_det} detections, conf >= {conf:.3f}, weights {a.weights})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
