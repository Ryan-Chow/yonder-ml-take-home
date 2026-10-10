"""Before/after evidence for the CLAHE preprocessing: results/clahe_examples.jpg + results/clahe_stats.md.

Picks the darkest, brightest and lowest-contrast train images, shows raw vs preprocessed with GT boxes, and reports
how the brightness/contrast distribution moves over the whole dataset.

    python scripts/show_preprocess.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_data import BRIGHT_L, DARK_L, LOW_CONTRAST, draw_boxes  # noqa: E402
from src.data_utils import RAW, ROOT, load_all  # noqa: E402
from src.preprocess import CLIP_LIMIT, TILE_GRID, preprocess  # noqa: E402


def lstats(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    return L.mean(), L.std()


def main():
    samples = load_all(RAW)
    raw = {s.name: cv2.imread(str(s.image)) for s in samples}
    before = np.array([lstats(raw[s.name]) for s in samples])
    after = np.array([lstats(preprocess(raw[s.name])) for s in samples])
    # hue preservation: mean absolute change of the a/b channels
    ab_shift = np.mean([np.abs(cv2.cvtColor(raw[s.name], cv2.COLOR_BGR2LAB)[..., 1:].astype(int)
                               - cv2.cvtColor(preprocess(raw[s.name]), cv2.COLOR_BGR2LAB)[..., 1:].astype(int)).mean()
                        for s in samples[::10]])

    def row(name, a):
        return (f"| {name} | {a[:, 0].mean():.0f} | {a[:, 0].std():.0f} | {a[:, 1].mean():.0f} | "
                f"{(a[:, 0] < DARK_L).mean() * 100:.0f}% | {(a[:, 0] > BRIGHT_L).mean() * 100:.0f}% | "
                f"{(a[:, 1] < LOW_CONTRAST).mean() * 100:.0f}% |")

    md = ["# CLAHE preprocessing: effect on the dataset", "",
          f"`src/preprocess.py`: CLAHE on LAB L, clipLimit={CLIP_LIMIT}, tileGridSize={TILE_GRID}. All 999 images.",
          "", "| | mean L | spread of mean L across images | mean contrast (std L) | dark | bright | low contrast |",
          "|---|---|---|---|---|---|---|", row("raw", before), row("CLAHE", after), "",
          f"- Mean absolute change of the LAB a/b (colour) channels: {ab_shift:.2f} / 255, i.e. hue is essentially "
          "untouched; only lightness is redistributed.",
          "- CLAHE is a *local* operation: it raises contrast inside each tile but deliberately does not normalise "
          "global exposure fully (the clip limit bounds the change). Overall exposure robustness is therefore "
          "left to the HSV-value augmentation in `configs/train.yaml`.",
          "", "`results/clahe_examples.jpg`: rows = darkest / brightest / lowest-contrast train images; "
          "left raw, right CLAHE (GT boxes drawn)."]
    (ROOT / "results" / "clahe_stats.md").write_text("\n".join(md) + "\n")

    train = [s for s in samples if s.split == "train"]
    tl = {s.name: lstats(raw[s.name]) for s in train}
    picks = [sorted(train, key=lambda s: tl[s.name][0])[:3],
             sorted(train, key=lambda s: -tl[s.name][0])[:3],
             sorted(train, key=lambda s: tl[s.name][1])[5:8]]  # skip the near-blank synthetic images
    rows = []
    for group in picks:
        tiles = []
        for s in group:
            a = draw_boxes(cv2.resize(raw[s.name], (220, 220)), s.boxes)
            b = draw_boxes(cv2.resize(preprocess(raw[s.name]), (220, 220)), s.boxes)
            tiles.append(np.hstack([a, b, np.full((220, 8, 3), 255, np.uint8)]))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(ROOT / "results" / "clahe_examples.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("\n".join(md))


if __name__ == "__main__":
    main()
