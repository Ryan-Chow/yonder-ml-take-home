"""Hard-example mining on the TRAIN split (stretch goal: active learning).

For every training image the shipped model (weights/best.pt) is run at a very low threshold; each GT box gets a
"GT confidence" = the highest confidence of a same-class prediction overlapping it at IoU >= 0.5 (0 if none).
An image's hardness is its worst GT confidence, penalised by confident false alarms. Images are then grouped by failure type:

  missed            no prediction of any class near the object (best IoU < 0.1 at conf >= 0.05)
  low confidence    detected, but below the deployed threshold
  mislocalised      a box above threshold overlaps the object only at IoU 0.1-0.5
  class confusion   the other class is predicted on the object above threshold
  false alarm       a confident prediction (>= threshold) on no labelled object
  label suspect     the model is very confident (>= 0.7) about the OTHER class on a GT box, or about an object
                    that has no label at all -> check the label before collecting more data

Training images were seen during training, so anything still hard here is either genuinely hard for this model
capacity or mislabelled. Writes results/hard_examples.md, results/hard_examples.jpg (contact sheet) and
results/hard_examples.csv.

    python scripts/hard_examples.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.error_analysis import family, light_bucket, render, size_bucket  # noqa: E402
from src.data_utils import CLASS_NAMES, ROOT, load_split  # noqa: E402
from src.infer import detect, load_model, uses_clahe  # noqa: E402
from src.matching import iou_matrix  # noqa: E402

RESULTS = ROOT / "results"
SUSPECT_CONF = 0.7


def analyse(G, P, thr):
    """Return per-image dict with worst GT confidence, failure types and label-suspect reasons."""
    ious = iou_matrix(P[:, 1:5], G[:, 1:5]) if len(P) and len(G) else np.zeros((len(P), len(G)))
    types, gt_conf, suspects = [], [], []
    for g in range(len(G)):
        cls = int(G[g, 0])
        same = [p for p in range(len(P)) if int(P[p, 0]) == cls and ious[p, g] >= 0.5]
        conf = max((P[p, 5] for p in same), default=0.0)
        gt_conf.append(conf)
        if conf >= thr:
            continue
        other = [p for p in range(len(P)) if int(P[p, 0]) != cls and ious[p, g] >= 0.5 and P[p, 5] >= thr]
        loose = [p for p in range(len(P)) if 0.1 <= ious[p, g] < 0.5 and P[p, 5] >= thr]
        near = [p for p in range(len(P)) if ious[p, g] >= 0.1 and P[p, 5] >= 0.05]
        if other:
            types.append("class confusion")
            if max(P[p, 5] for p in other) >= SUSPECT_CONF:
                suspects.append(f"GT {CLASS_NAMES[cls]} predicted as {CLASS_NAMES[1 - cls]} "
                                f"{max(P[p, 5] for p in other):.2f}")
        elif loose:
            types.append("mislocalised")
        elif conf > 0 or near:
            types.append("low confidence")
        else:
            types.append("missed")
    for p in range(len(P)):
        if P[p, 5] >= thr and (len(G) == 0 or ious[p].max() < 0.1):
            types.append("false alarm")
            if P[p, 5] >= SUSPECT_CONF:
                suspects.append(f"unlabelled {CLASS_NAMES[int(P[p, 0])]}? conf {P[p, 5]:.2f}")
    n_fa = sum(t == "false alarm" for t in types)
    worst = min(gt_conf) if gt_conf else 1.0
    hardness = (1 - worst) + 0.5 * n_fa
    if suspects:
        types.append("label suspect")
    return {"worst_gt_conf": float(worst), "hardness": float(hardness), "types": sorted(set(types)),
            "suspects": suspects}


def _rate(rows, hard, key, value):
    tot = sum(r[key] == value for r in rows)
    return sum(r[key] == value for r in hard) / max(tot, 1), tot


def strategy(rows, hard, sus) -> list[str]:
    """Data-collection priorities, computed from the train hard examples and (if present) the val error analysis."""
    small, n_small = _rate(rows, hard, "size", "small (<32px)")
    med, _ = _rate(rows, hard, "size", "medium (32-96px)")
    both, n_both = _rate(rows, hard, "classes", "bottle+mallet")
    single = sum(r["classes"] != "bottle+mallet" for r in hard) / max(sum(r["classes"] != "bottle+mallet"
                                                                         for r in rows), 1)
    val = {}
    ef = RESULTS / "errors.json"
    if ef.exists():
        errs = json.loads(ef.read_text())["errors"]
        val["fn_by_light"] = Counter(e["light"] for e in errs if e["type"] == "fn")
        val["fn_by_family"] = Counter(e["family"] for e in errs if e["type"] == "fn")
        val["fp_by_family"] = Counter(e["family"] for e in errs if e["type"] == "fp")
    out = ["## Data-collection priorities", "",
           "Ordered by expected value per labelling hour, using the train hard examples above and the val errors in "
           "`results/errors.md`. The point is to collect *specific* data for the failure modes we can see, not more "
           "of what the model already handles (most medium-sized, single-object outdoor shots are already easy).", "",
           f"1. **Fix labels before collecting anything.** {len(sus)} train images have confident detections of "
           "objects that carry no label, and the val error analysis found mallets labelled as bottles "
           "(`t_bot_mal578`, `t_bot_mal581`). A wrong label teaches the wrong thing no matter how much new data "
           "is added. Cost: minutes. Re-check the label-suspect list and every wrong-class error image.",
           f"2. **Small, distant objects.** Images whose smallest object is under 32 px are hard "
           f"{small * 100:.0f}% of the time ({n_small} images) vs {med * 100:.0f}% for 32–96 px. On the rover, the "
           "mallet has to be found *before* the rover is close, so this is the operating regime that matters. "
           "Collect: the mallet and bottles placed 3–15 m from a rover-height camera, on the competition-like "
           "terrain (dirt, gravel, scrub), several headings. Consider also training at imgsz 640 for small "
           "objects (see `results/efficiency.md` for the latency cost).",
           f"3. **Scenes with several objects.** Images with both classes are hard {both * 100:.0f}% of the time "
           f"({n_both} images) vs {single * 100:.0f}% for single-class images. Part of this is missing labels "
           "(priority 1), the rest is crowding and small size. Collect: multi-object scenes with distractors "
           "(other bottles, orange tools, cans), and label *every* instance."]
    if val:
        dark = sum(v for k, v in val["fn_by_light"].items() if k.startswith("dark"))
        bright = sum(v for k, v in val["fn_by_light"].items() if k.startswith("bright"))
        fam_fn = ", ".join(f"`{k}` {v}" for k, v in val["fn_by_family"].most_common(3))
        fam_fp = ", ".join(f"`{k}` {v}" for k, v in val["fp_by_family"].most_common(3))
        out += [f"4. **Exposure extremes and unseen contexts (from the val errors).** {dark} of the val misses are "
                f"in dark images and {bright} in bright ones; by source the most misses come from {fam_fn}. "
                "Collect: dawn/dusk, harsh midday sun with hard shadows, backlit shots, and the objects partly "
                "occluded (held, half-buried, behind scrub), all at rover height.",
                f"5. **Hard negatives.** Most val false positives come from {fam_fp}. Collect rover-terrain images "
                "*without* the objects but with look-alikes (blue barrels/cans, orange debris, shadows) and add them "
                "as background images (empty label files) so the model learns what is *not* a bottle."]
    out += ["6. **De-prioritise / drop off-domain data.** Product-page screenshots, white-background collages "
            "(`Untitled-drawing`) and studio product shots are a web-scrape domain the rover will never see; their "
            "errors (tiny thumbnails, colour swatches, extreme close-ups) are not worth fixing with more of the "
            "same. Measure the val score without them before deciding to remove them from training.",
            "7. **Close the loop (active learning).** Run the model over unlabelled rover footage and send for "
            "labelling the frames where (a) max confidence falls in the uncertain band 0.2–0.6, or (b) a "
            "detection flickers on and off between neighbouring frames (`scripts/video_test.py` measures this). "
            "After each labelled batch, re-run `scripts/hard_examples.py` and `scripts/error_analysis.py` and "
            "re-rank these priorities. Always add new data by *session* to train or val, never split a session "
            "(the leakage lesson from `results/audit.md`).", ""]
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    ap.add_argument("--conf", type=float)
    a = ap.parse_args()
    thr = a.conf if a.conf is not None else json.loads((RESULTS / "threshold.json").read_text())["conf"]
    samples = load_split(ROOT / "data" / "split", "train")
    imgs = [cv2.imread(str(s.image)) for s in samples]
    preds = detect(load_model(a.weights), imgs, conf=0.01, clahe=uses_clahe(a.weights), device="cpu")

    rows = []
    for s, img, P in zip(samples, imgs, preds):
        G = np.array([[b.cls, b.xc, b.yc, b.w, b.h] for b in s.boxes]).reshape(-1, 5)
        a = analyse(G, P, thr)
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean()
        smallest = min(G, key=lambda g: g[3] * g[4])
        rows.append(dict(a, image=s.name, light=light_bucket(L), family=family(s.name),
                         size=size_bucket(smallest[3], smallest[4]), n_gt=len(G),
                         classes="+".join(sorted({CLASS_NAMES[int(g[0])] for g in G}))))
        rows[-1]["_vis"] = (img, G, P)
    rows.sort(key=lambda r: -r["hardness"])
    hard = [r for r in rows if r["types"]]

    with (RESULTS / "hard_examples.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "image", "hardness", "worst_gt_conf", "types", "suspects", "classes", "size", "light",
                    "family"])
        for k, r in enumerate(hard, 1):
            w.writerow([k, r["image"], f"{r['hardness']:.3f}", f"{r['worst_gt_conf']:.3f}", ";".join(r["types"]),
                        " | ".join(r["suspects"]), r["classes"], r["size"], r["light"], r["family"]])

    tiles = []
    for r in hard[:24]:
        img, G, P = r["_vis"]
        Pv = P[P[:, 5] >= 0.1]
        title = f"{r['worst_gt_conf']:.2f} {'/'.join(t[:12] for t in r['types'])}"
        tiles.append(cv2.resize(render(img, G, Pv, title=title), (256, 256)))
    while tiles and len(tiles) % 6:
        tiles.append(np.full((256, 256, 3), 255, np.uint8))
    if tiles:
        cv2.imwrite(str(RESULTS / "hard_examples.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])

    n = len(rows)
    tcount = Counter(t for r in hard for t in r["types"])
    base = {k: Counter(r[k] for r in rows) for k in ("size", "light", "family", "classes")}

    def over(key):
        hc = Counter(r[key] for r in hard)
        out = []
        for v, tot in base[key].most_common():
            if tot >= 5:
                out.append([v, tot, hc[v], f"{hc[v] / tot * 100:.0f}%"])
        return sorted(out, key=lambda x: -hc[x[0]] / x[1])

    def table(h, rws):
        return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)]
                         + ["| " + " | ".join(str(c) for c in r) + " |" for r in rws])

    md = ["# Hard examples (train split)", "",
          f"Generated by `scripts/hard_examples.py` with the shipped model (`weights/best.pt`) at the deployed threshold {thr:.3f}. "
          f"{n} train images; **{len(hard)} have at least one failure** even though the model trained on them.", "",
          "## Failure types (an image can have several)", "",
          table(["type", "images"], [[t, c] for t, c in tcount.most_common()]), "",
          "## Where the hard images concentrate (share of each bucket that is hard; buckets with >= 5 images)", ""]
    for key, title in (("size", "smallest object in image"), ("light", "image brightness"),
                       ("classes", "classes present"), ("family", "filename family / source")):
        md += [f"**{title}**", "", table([key, "images", "hard", "hard rate"], over(key)), ""]
    sus = [r for r in hard if r["suspects"]]
    md += ["## Label suspects", "",
           f"{len(sus)} train images where the model is >= {SUSPECT_CONF} confident about something the labels "
           "disagree with. These should be **re-checked by a human before** any new data is collected, because "
           "a wrong label is a hard example that more data cannot fix.", "",
           table(["image", "reason"], [[r["image"][:60], "; ".join(r["suspects"])] for r in sus[:25]]), "",
           "## Top 15 hardest", "",
           table(["rank", "image", "worst GT conf", "types", "size", "light"],
                 [[k, r["image"][:55], f"{r['worst_gt_conf']:.2f}", ", ".join(r["types"]), r["size"], r["light"]]
                  for k, r in enumerate(hard[:15], 1)]), "",
           "Contact sheet of the 24 hardest: `results/hard_examples.jpg` (green = GT, red = predictions >= 0.1). "
           "Full ranking: `results/hard_examples.csv`.", ""]
    md += strategy(rows, hard, sus)
    (RESULTS / "hard_examples.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
