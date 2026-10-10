"""Error analysis of the shipped model (weights/best.pt) on the grouped val split at the deployed threshold.

Matches predictions to GT at IoU 0.5 (src/matching.py), saves every false positive, false negative and wrong-class
detection as an image in results/errors/ (GT green, prediction red, the error itself drawn thick), and writes
results/errors.md with counts AND rates by type, class, box size, brightness and contrast.

This file reports patterns as data only. Hypotheses about *why* go in DEV_NOTES.md (git-ignored).

    python scripts/error_analysis.py [--weights weights/best.pt] [--conf T]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_data import BRIGHT_L, DARK_L, LOW_CONTRAST  # noqa: E402
from src.data_utils import CLASS_NAMES, ROOT, load_split  # noqa: E402
from src.infer import detect, load_model, uses_clahe  # noqa: E402
from src.matching import iou_matrix, match_image  # noqa: E402

RESULTS = ROOT / "results"
GREEN, RED = (60, 180, 60), (40, 40, 220)


def size_bucket(w, h):
    s = np.sqrt(w * h) * 512
    return "small (<32px)" if s < 32 else ("medium (32-96px)" if s < 96 else "large (>=96px)")


def light_bucket(L):
    return f"dark (L<{DARK_L})" if L < DARK_L else (f"bright (L>{BRIGHT_L})" if L > BRIGHT_L else "normal")


def contrast_bucket(std):
    return f"low (std<{LOW_CONTRAST})" if std < LOW_CONTRAST else "normal"


def family(name):
    """Coarse filename family (prefix before the capture number), for spotting source-specific failures."""
    import re
    stem = name.split(".rf.")[0]
    m = re.match(r"^([A-Za-z_\-]*?)[_\-]?\d", stem)
    return (m.group(1) if m and m.group(1) else "numeric/other")[:24]


def box_px(b, w, h):
    xc, yc, bw, bh = b
    return int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)


def render(img, gts, preds, focus_gt=None, focus_pred=None, title=""):
    im = img.copy()
    h, w = im.shape[:2]
    for k, g in enumerate(gts):
        x1, y1, x2, y2 = box_px(g[1:5], w, h)
        cv2.rectangle(im, (x1, y1), (x2, y2), GREEN, 4 if k == focus_gt else 1)
        cv2.putText(im, f"GT {CLASS_NAMES[int(g[0])]}", (x1, min(y2 + 14, h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    GREEN, 1)
    for k, p in enumerate(preds):
        x1, y1, x2, y2 = box_px(p[1:5], w, h)
        cv2.rectangle(im, (x1, y1), (x2, y2), RED, 4 if k == focus_pred else 1)
        cv2.putText(im, f"{CLASS_NAMES[int(p[0])]} {p[5]:.2f}", (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, RED, 1)
    cv2.rectangle(im, (0, 0), (w, 20), (0, 0, 0), -1)
    cv2.putText(im, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return im


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "split")
    a = ap.parse_args()
    conf = a.conf if a.conf is not None else json.loads((RESULTS / "threshold.json").read_text())["conf"]

    samples = load_split(a.data, "valid")
    imgs = [cv2.imread(str(s.image)) for s in samples]
    preds = detect(load_model(a.weights), imgs, conf=conf, clahe=uses_clahe(a.weights), device="cpu")

    out_dir = RESULTS / "errors"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    errors = []          # one dict per error
    gt_rows = []         # one dict per GT box (denominators for rates)
    for s, img, P in zip(samples, imgs, preds):
        G = np.array([[b.cls, b.xc, b.yc, b.w, b.h] for b in s.boxes]).reshape(-1, 5)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        ctx = {"image": s.name, "light": light_bucket(lab.mean()), "contrast": contrast_bucket(lab.std()),
               "family": family(s.name), "n_gt": len(G)}
        ious = iou_matrix(P[:, 1:5], G[:, 1:5]) if len(P) and len(G) else np.zeros((len(P), len(G)))
        matches = match_image(P, G)
        matched_gt = {m.gt_idx for m in matches if m.kind == "tp"}
        hit = {m.gt_idx: m.kind for m in matches if m.gt_idx is not None}
        for g in range(len(G)):
            gt_rows.append(dict(ctx, cls=CLASS_NAMES[int(G[g, 0])], size=size_bucket(G[g, 3], G[g, 4]),
                                outcome=hit.get(g, "fn")))
        for m in matches:
            if m.kind == "tp":
                continue
            e = dict(ctx, type=m.kind)
            if m.kind == "fn":
                g = G[m.gt_idx]
                best = float(ious[:, m.gt_idx].max()) if len(P) else 0.0
                e.update(cls=CLASS_NAMES[int(g[0])], size=size_bucket(g[3], g[4]), conf=None,
                         subtype="poorly localised (IoU 0.1-0.5)" if best >= 0.1 else "no detection near it")
            elif m.kind == "fp":
                p = P[m.pred_idx]
                same = [g for g in range(len(G)) if int(G[g, 0]) == int(p[0])]
                best_same = max((ious[m.pred_idx, g] for g in same), default=0.0)
                best_any = float(ious[m.pred_idx].max()) if len(G) else 0.0
                if best_same >= 0.5 and any(g in matched_gt for g in same if ious[m.pred_idx, g] >= 0.5):
                    sub = "duplicate of a matched object"
                elif best_any >= 0.1:
                    sub = "poor localisation (IoU 0.1-0.5)"
                else:
                    sub = "background (no object there)"
                e.update(cls=CLASS_NAMES[int(p[0])], size=size_bucket(p[3], p[4]), conf=float(p[5]), subtype=sub)
            else:  # wrong class
                p, g = P[m.pred_idx], G[m.gt_idx]
                e.update(cls=f"{CLASS_NAMES[int(g[0])]}->{CLASS_NAMES[int(p[0])]}", size=size_bucket(g[3], g[4]),
                         conf=float(p[5]), subtype="class confusion")
            k = len(errors)
            title = f"{m.kind.upper()} {e['cls']} | {e['subtype']} | {s.name[:28]}"
            vis = render(img, G, P, focus_gt=m.gt_idx, focus_pred=m.pred_idx, title=title)
            fname = f"{k:03d}_{m.kind}_{e['cls'].replace('->', '_as_')}_{Path(s.name).stem[:40]}.jpg"
            cv2.imwrite(str(out_dir / fname), cv2.resize(vis, (384, 384)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            e["file"] = f"errors/{fname}"
            errors.append(e)

    # ---------------------------------------------------------------- tables
    n_gt = len(gt_rows)
    n_tp = sum(r["outcome"] == "tp" for r in gt_rows)
    by_type = Counter(e["type"] for e in errors)
    md = ["# Error analysis (shipped model, grouped val)", "",
          f"Generated by `scripts/error_analysis.py`. Threshold conf >= {conf:.3f} (results/threshold.json), match "
          f"IoU >= 0.5. {len(samples)} images, {n_gt} GT boxes, {n_tp} matched correctly.", "",
          "Every error is saved in `results/errors/` (green = GT, red = prediction, thick = the error). "
          "`results/errors_sheet.jpg` shows a sample.", "",
          "## Counts by type", "",
          md_table(["type", "count"], [["false negative (GT missed)", by_type["fn"]],
                                       ["false positive (spurious box)", by_type["fp"]],
                                       ["wrong class (right place, wrong label)", by_type["wrong_class"]]]), "",
          "## Subtypes", "",
          md_table(["type", "subtype", "count"],
                   sorted([[t, s, n] for (t, s), n in Counter((e["type"], e["subtype"]) for e in errors).items()])),
          "", "## By class", "",
          md_table(["class", "FN", "FP", "wrong class (true class->predicted)"],
                   [[c, sum(e["type"] == "fn" and e["cls"] == c for e in errors),
                     sum(e["type"] == "fp" and e["cls"] == c for e in errors),
                     sum(e["type"] == "wrong_class" and e["cls"].startswith(c) for e in errors)] for c in CLASS_NAMES]),
          ""]

    def rate_table(key, title):
        groups = defaultdict(lambda: [0, 0])
        for r in gt_rows:
            groups[r[key]][0] += 1
            groups[r[key]][1] += r["outcome"] != "tp"
        fp_counts = Counter(e[key] for e in errors if e["type"] == "fp")
        rows = [[k, n, miss, f"{miss / n * 100:.0f}%", fp_counts.get(k, 0)] for k, (n, miss) in sorted(groups.items())]
        return [f"## {title}", "", md_table([key, "GT boxes", "GT not matched (FN + wrong class)", "miss rate",
                                             "FP boxes"], rows), ""]

    md += rate_table("size", "By box size (miss rate = share of GT boxes in the bucket that were not detected)")
    md += rate_table("light", "By image brightness (raw image, LAB L mean)")
    md += rate_table("contrast", "By image contrast (raw image, std of L)")
    md += rate_table("cls", "Miss rate by class")
    fam = rate_table("family", "By filename family (source/session prefix)")
    md += fam
    fps = [e["conf"] for e in errors if e["type"] == "fp"]
    if fps:
        md += ["## False-positive confidence", "",
               f"{len(fps)} FPs, confidence median {np.median(fps):.2f}, max {max(fps):.2f}; "
               f"{sum(c >= 0.5 for c in fps)} at conf >= 0.5.", ""]
    md += ["## Worst images (most errors)", "",
           md_table(["image", "errors", "types"],
                    [[im, n, ", ".join(f"{t}x{c}" for t, c in Counter(e['type'] for e in errors
                                                                      if e['image'] == im).items())]
                     for im, n in Counter(e["image"] for e in errors).most_common(8)]), ""]
    (RESULTS / "errors.md").write_text("\n".join(md) + "\n")
    (RESULTS / "errors.json").write_text(json.dumps({"conf": conf, "errors": errors}, indent=2))

    files = sorted(out_dir.glob("*.jpg"))
    if files:
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(files), min(24, len(files)), replace=False))
        tiles = [cv2.resize(cv2.imread(str(files[i])), (256, 256)) for i in pick]
        while len(tiles) % 6:
            tiles.append(np.full((256, 256, 3), 255, np.uint8))
        sheet = np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)])
        cv2.imwrite(str(RESULTS / "errors_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("\n".join(md))


if __name__ == "__main__":
    main()
