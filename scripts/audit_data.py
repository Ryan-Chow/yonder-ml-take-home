"""Audit the raw dataset before any modelling.

Writes results/audit.md, results/audit_plots.png, results/audit_samples.jpg (GT boxes on 16 random images, to
check label parsing by eye) and results/leakage_pairs.jpg (examples of val images whose source photo/scene also
appears in train). Read-only with respect to Sampled-YD-Object-Detection-2/.

    python scripts/audit_data.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import (CLASS_NAMES, RAW, ROOT, build_groups, compute_phashes, link_pairs, load_all,  # noqa: E402
                            source_stem)

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache"
SEED = 0
# palette validated with the dataviz validator (CVD dE 24.7): blue = bottle / train, orange = mallet / valid
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
CLASS_BGR = {0: (214, 120, 42), 1: (52, 104, 235)}

# brightness / contrast buckets on the LAB L channel (0-255), used here and in error analysis
DARK_L, BRIGHT_L, LOW_CONTRAST = 80, 170, 40


def image_stats(path: Path) -> dict:
    img = cv2.imread(str(path))
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    h, w = L.shape
    k = 6
    corners = [L[:k, :k], L[:k, -k:], L[-k:, :k], L[-k:, -k:]]
    # Roboflow's rotation augmentation fills the uncovered corners with black
    rotated = sum(float(c.mean()) < 8 for c in corners) >= 2
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    return {"L_mean": float(L.mean()), "L_std": float(L.std()), "rotated": rotated, "h": h, "w": w,
            "clip_dark": float((L < 10).mean()), "clip_bright": float((L > 245).mean()),
            "sat_mean": float(sat.mean())}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def draw_boxes(img, boxes, thickness=2):
    h, w = img.shape[:2]
    for b in boxes:
        x1, y1 = int((b.xc - b.w / 2) * w), int((b.yc - b.h / 2) * h)
        x2, y2 = int((b.xc + b.w / 2) * w), int((b.yc + b.h / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), CLASS_BGR[b.cls], thickness)
        cv2.putText(img, CLASS_NAMES[b.cls], (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    CLASS_BGR[b.cls], 2)
    return img


def sample_sheet(samples, out: Path, n=16, size=256):
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(samples), n, replace=False)
    tiles = []
    for i in pick:
        s = samples[i]
        img = cv2.resize(draw_boxes(cv2.imread(str(s.image)), s.boxes), (size, size))
        cv2.putText(img, f"{s.split}:{s.name[:22]}", (3, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(n // 4)])
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def leakage_sheet(samples, pairs_by_rule, out: Path, per_rule=4, size=160):
    rows = []
    for rule, pairs in pairs_by_rule.items():
        if not pairs:
            continue
        rng = np.random.default_rng(SEED)
        sel = [pairs[k] for k in rng.choice(len(pairs), min(per_rule, len(pairs)), replace=False)]
        row = []
        for vi, ti in sel:
            a = cv2.resize(cv2.imread(str(samples[vi].image)), (size, size))
            b = cv2.resize(cv2.imread(str(samples[ti].image)), (size, size))
            cv2.putText(a, "val", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(b, "train", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            row.append(np.hstack([a, b, np.full((size, 8, 3), 255, np.uint8)]))
        while len(row) < per_rule:
            row.append(np.full((size, 2 * size + 8, 3), 255, np.uint8))
        label = np.full((size, 110, 3), 255, np.uint8)
        cv2.putText(label, rule, (4, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        rows.append(np.hstack([label] + row))
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def main():
    RESULTS.mkdir(exist_ok=True)
    samples = load_all(RAW)  # raises on any malformed label line
    stats = {f"{s.split}/{s.name}": image_stats(s.image) for s in samples}
    splits = ("train", "valid")
    by_split = {sp: [s for s in samples if s.split == sp] for sp in splits}
    md = ["# Data audit", "", f"Source: `{RAW.name}/` (read-only). Generated by `scripts/audit_data.py`.", ""]

    # ---- integrity -------------------------------------------------------------------------------------
    sizes = Counter((st["w"], st["h"]) for st in stats.values())
    empty = [s.name for s in samples if not s.boxes]
    md += ["## Integrity", "",
           f"- {len(samples)} images, every one with a label file; all {sum(len(s.boxes) for s in samples)} "
           "label lines parse as `class xc yc w h` with class in {0,1} and coords in [0,1].",
           f"- Image sizes: {dict(sizes)} (Roboflow 'Resize to 512x512 (Stretch)', so original aspect ratios "
           "are lost).",
           f"- Images with no boxes: {len(empty)}.", ""]

    # ---- counts ----------------------------------------------------------------------------------------
    rows = []
    for sp in splits:
        ss = by_split[sp]
        box_c = Counter(b.cls for s in ss for b in s.boxes)
        img_c = Counter(c for s in ss for c in s.classes)
        both = sum(1 for s in ss if len(s.classes) == 2)
        rows.append([sp, len(ss), box_c[0], box_c[1], img_c[0], img_c[1], both])
    tot = [sum(r[i] for r in rows) for i in range(1, 7)]
    rows.append(["**total**", *tot])
    md += ["## Class counts", "",
           md_table(["split", "images", "bottle boxes", "mallet boxes", "imgs w/ bottle", "imgs w/ mallet",
                     "imgs w/ both"], rows), ""]

    # ---- objects per image / box geometry -------------------------------------------------------------
    per_img = Counter(len(s.boxes) for s in samples)
    md += ["## Objects per image", "", md_table(["boxes", "images"], sorted(per_img.items())), ""]
    geo = defaultdict(lambda: defaultdict(list))
    for s in samples:
        for b in s.boxes:
            g = geo[b.cls]
            g["area"].append(b.w * b.h)
            g["aspect"].append(b.w / b.h)
            g["px"].append(np.sqrt(b.w * b.h) * 512)
    rows = []
    for c in (0, 1):
        a, px, asp = np.array(geo[c]["area"]), np.array(geo[c]["px"]), np.array(geo[c]["aspect"])
        rows.append([CLASS_NAMES[c], len(a), f"{np.median(a) * 100:.2f}%", f"{np.percentile(a, 10) * 100:.2f}%",
                     f"{(px < 32).mean() * 100:.0f}%", f"{((px >= 32) & (px < 96)).mean() * 100:.0f}%",
                     f"{(px >= 96).mean() * 100:.0f}%", f"{np.median(asp):.2f}",
                     f"{np.percentile(asp, 10):.2f}–{np.percentile(asp, 90):.2f}"])
    md += ["## Box geometry (at 512x512)", "",
           md_table(["class", "boxes", "median area", "p10 area", "small <32px", "medium 32–96px", "large >=96px",
                     "median w/h", "w/h p10–p90"], rows),
           "", "Size buckets use sqrt(box area) in pixels at 512x512, roughly following COCO's small/medium/large.",
           ""]

    # ---- lighting ---------------------------------------------------------------------------------------
    rows = []
    for sp in splits:
        L = np.array([stats[f"{sp}/{s.name}"]["L_mean"] for s in by_split[sp]])
        C = np.array([stats[f"{sp}/{s.name}"]["L_std"] for s in by_split[sp]])
        rot = np.mean([stats[f"{sp}/{s.name}"]["rotated"] for s in by_split[sp]])
        rows.append([sp, f"{L.mean():.0f} ± {L.std():.0f}", f"{np.percentile(L, 5):.0f}–{np.percentile(L, 95):.0f}",
                     f"{(L < DARK_L).mean() * 100:.0f}%", f"{(L > BRIGHT_L).mean() * 100:.0f}%",
                     f"{C.mean():.0f}", f"{(C < LOW_CONTRAST).mean() * 100:.0f}%", f"{rot * 100:.0f}%"])
    md += ["## Lighting and contrast (LAB L channel, 0–255)", "",
           md_table(["split", "mean L", "L p5–p95", f"dark (L<{DARK_L})", f"bright (L>{BRIGHT_L})",
                     "mean contrast (std L)", f"low contrast (std<{LOW_CONTRAST})", "rotated (black corners)"],
                    rows), ""]
    allL = np.array([v["L_mean"] for v in stats.values()])
    allC = np.array([v["L_std"] for v in stats.values()])
    md += [f"- Global brightness spans L {allL.min():.0f}–{allL.max():.0f}; contrast (std of L) spans "
           f"{allC.min():.0f}–{allC.max():.0f}. Correlation(brightness, contrast) = "
           f"{np.corrcoef(allL, allC)[0, 1]:.2f}.", ""]

    # ---- leakage ---------------------------------------------------------------------------------------
    hashes = compute_phashes(samples, CACHE / "phash_center.json")
    is_val = np.array([s.split == "valid" for s in samples])
    rule_pairs, val_hit = {}, {}
    exact = defaultdict(set)
    for i, s in enumerate(samples):
        exact[source_stem(s.name)].add(i)
    rule_pairs["exact stem"] = [(i, j) for grp in exact.values() for i in grp for j in grp
                                if is_val[i] and not is_val[j]]
    for rule in ("family", "sequence", "phash"):
        pairs = []
        for i, j, _ in link_pairs(samples, hashes, rules=(rule,)):
            if is_val[i] != is_val[j]:
                pairs.append((i, j) if is_val[i] else (j, i))
        rule_pairs[rule] = pairs
    groups = build_groups(samples, hashes)
    grp_splits = defaultdict(set)
    for g, s in zip(groups, samples):
        grp_splits[g].add(s.split)
    for rule, pairs in rule_pairs.items():
        val_hit[rule] = len({v for v, _ in pairs})
    leaked_val = [i for i, g in enumerate(groups) if is_val[i] and len(grp_splits[g]) > 1]
    val_hit["any (connected groups)"] = len(leaked_val)
    n_val = int(is_val.sum())
    gsize = Counter(groups)
    rows = [[r, n, f"{n / n_val * 100:.0f}%"] for r, n in val_hit.items()]
    md += ["## Train/val leakage", "",
           "The task says images are augmented variants of shared source photos (`<orig>_jpg.rf.<hash>.jpg`). "
           "In this sample that is **not** how leakage shows up: `sampling/sample-roboflow.py` keeps one `.rf.` "
           "copy per source photo, so the exact-stem check finds nothing. Same-origin images still cross the "
           "split through other paths, so each val image is checked against train with four rules:",
           "",
           "- **exact stem**: same `<orig>` before `.rf.`.",
           "- **family**: frames of one video (`<video>_mp4-<n>`) and `-copy` duplicates of one photo.",
           "- **sequence**: consecutive capture numbers in one series (`IMG_0619`/`IMG_0620`, "
           "`t_bot_mal262`/`263`, gap <= 2) or phone timestamps within 30 s (`20231021_173204`). Manual precision "
           "check (30 random cross-split sequence pairs, seed 42, contact sheet reviewed by eye): 28 same scene "
           "and object, 2 same bottle and session with a different background, 0 unrelated.",
           "- **phash**: perceptual hash of the central 40% of the image, Hamming distance <= 10. The centre "
           "crop is deliberate: the black corners from Roboflow's rotation made whole-image hashes match "
           "unrelated rotated photos at distance 8. 13 near-uniform crops (white synthetic collages) are "
           "excluded, because they all hash identically.",
           "",
           md_table(["rule", "val images with a same-origin train image", f"% of {n_val} val"], rows), "",
           f"Connected components over all rules: **{len(gsize)} groups** for {len(samples)} images (largest "
           f"{max(gsize.values())}, singletons {sum(1 for v in gsize.values() if v == 1)}). "
           f"**{len(leaked_val)} of {n_val} val images ({len(leaked_val) / n_val * 100:.0f}%) share a group "
           "with at least one train image.** Metrics on the original split therefore partly measure "
           "recall of near-duplicates. `scripts/make_split.py` re-splits by group.",
           "", "Examples (left = val, right = its train sibling): `results/leakage_pairs.jpg`.", ""]
    leakage_sheet(samples, {k: v for k, v in rule_pairs.items() if k != "exact stem"},
                  RESULTS / "leakage_pairs.jpg")

    # ---- plots -----------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor="#fcfcfb")
    ax = axes[0, 0]
    for c, col in ((0, BLUE), (1, ORANGE)):
        ax.hist(np.sqrt(geo[c]["area"]) * 512, bins=30, alpha=0.65, color=col, label=CLASS_NAMES[c],
                edgecolor="#fcfcfb", linewidth=0.5)
    style(ax, "Box size", "sqrt(box area) in px @512", "boxes")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    for c, col in ((0, BLUE), (1, ORANGE)):
        ax.hist(np.log2(geo[c]["aspect"]), bins=30, alpha=0.65, color=col, label=CLASS_NAMES[c],
                edgecolor="#fcfcfb", linewidth=0.5)
    style(ax, "Box aspect ratio", "log2(width / height)", "boxes")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 2]
    for c, col in ((0, BLUE), (1, ORANGE)):
        bx = [b for s in samples for b in s.boxes if b.cls == c]
        ax.scatter([b.xc for b in bx], [b.yc for b in bx], s=6, alpha=0.5, color=col, label=CLASS_NAMES[c])
    ax.invert_yaxis()
    style(ax, "Box centres", "x (normalised)", "y (normalised)")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 0]
    for sp, col in (("train", BLUE), ("valid", ORANGE)):
        ax.hist([stats[f"{sp}/{s.name}"]["L_mean"] for s in by_split[sp]], bins=30, density=True, alpha=0.65,
                color=col, label=sp, edgecolor="#fcfcfb", linewidth=0.5)
    for x in (DARK_L, BRIGHT_L):
        ax.axvline(x, color=MUTED, lw=0.8, ls="--")
    style(ax, "Image brightness", "mean L (LAB)", "density")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 1]
    for sp, col in (("train", BLUE), ("valid", ORANGE)):
        ax.hist([stats[f"{sp}/{s.name}"]["L_std"] for s in by_split[sp]], bins=30, density=True, alpha=0.65,
                color=col, label=sp, edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(LOW_CONTRAST, color=MUTED, lw=0.8, ls="--")
    style(ax, "Image contrast", "std of L (LAB)", "density")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 2]
    k = sorted(per_img)
    ax.bar([str(x) for x in k], [per_img[x] for x in k], color=BLUE, width=0.6)
    style(ax, "Objects per image", "boxes in image", "images")
    fig.tight_layout()
    fig.savefig(RESULTS / "audit_plots.png", dpi=110)
    plt.close(fig)
    sample_sheet(samples, RESULTS / "audit_samples.jpg")

    md += ["## Figures", "", "- `results/audit_plots.png`: box size, aspect, centres, brightness, contrast, "
           "objects per image.",
           "- `results/audit_samples.jpg`: 16 random images with GT boxes (blue = bottle, orange = mallet), "
           "to check label parsing by eye.", ""]
    (RESULTS / "audit.md").write_text("\n".join(md))
    summary = {"n_images": len(samples), "val_leaked": len(leaked_val), "n_val": n_val, "n_groups": len(gsize),
               "val_hits_by_rule": val_hit}
    (RESULTS / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
