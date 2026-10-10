"""Group-aware, class-stratified ~80/20 train/val split.

All 999 images are pooled, grouped by same origin (src/data_utils.build_groups: video / -copy family, consecutive
capture numbers or timestamps, centre-crop phash), and whole groups are assigned to val so that ~20% of the images
containing each class end up in val. Many random group orders are tried (fixed seed) and the most balanced one is
kept.

Outputs (git-ignored except the manifest):
    data/split/{train,valid}/{images,labels}/   copies of the raw files
    data/split/data.yaml                        no `test:` line
    data/orig/data.yaml                         the ORIGINAL Roboflow split, kept for the leakage comparison
    results/split_manifest.csv                  name, original split, new split, group id, leaked flag

    python scripts/make_split.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import CLASS_NAMES, RAW, ROOT, build_groups, compute_phashes, load_all  # noqa: E402

SEED = 42
VAL_FRAC = 0.20
TRIALS = 5000
OUT = ROOT / "data" / "split"
ORIG = ROOT / "data" / "orig"


def write_yaml(path: Path, root: Path, train: str, val: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"path": str(root.resolve()), "train": train, "val": val, "nc": len(CLASS_NAMES),
           "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def choose_val_groups(groups, samples, rng):
    """Random group orders; fill val until it reaches VAL_FRAC of images; keep the order whose per-class image
    fractions and box fractions are closest to VAL_FRAC."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    gids = sorted(members)
    n = len(samples)
    img_has = np.array([[c in s.classes for c in range(len(CLASS_NAMES))] for s in samples], dtype=float)
    box_cnt = np.array([[sum(b.cls == c for b in s.boxes) for c in range(len(CLASS_NAMES))] for s in samples],
                       dtype=float)
    tot_img, tot_box = img_has.sum(0), box_cnt.sum(0)
    best, best_cost = None, np.inf
    for _ in range(TRIALS):
        order = rng.permutation(len(gids))
        val, count = [], 0
        for k in order:
            g = members[gids[k]]
            if count + len(g) > VAL_FRAC * n * 1.02:  # don't overshoot; skip big groups that no longer fit
                continue
            val.extend(g)
            count += len(g)
            if count >= VAL_FRAC * n * 0.98:
                break
        idx = np.array(val)
        f_img = img_has[idx].sum(0) / tot_img
        f_box = box_cnt[idx].sum(0) / tot_box
        cost = np.abs(f_img - VAL_FRAC).sum() + np.abs(f_box - VAL_FRAC).sum() + abs(count / n - VAL_FRAC)
        if cost < best_cost:
            best, best_cost = set(val), cost
    return best, best_cost


def main():
    samples = load_all(RAW)
    hashes = compute_phashes(samples, ROOT / "data" / "cache" / "phash_center.json")
    groups = build_groups(samples, hashes)
    rng = np.random.default_rng(SEED)
    val_idx, cost = choose_val_groups(groups, samples, rng)

    # sanity: no group may straddle the new split
    grp_split = defaultdict(set)
    for i, g in enumerate(groups):
        grp_split[g].add("valid" if i in val_idx else "train")
    straddle = [g for g, s in grp_split.items() if len(s) > 1]
    assert not straddle, f"{len(straddle)} groups straddle the split"

    if OUT.exists():
        shutil.rmtree(OUT)
    for i, s in enumerate(samples):
        split = "valid" if i in val_idx else "train"
        for src, kind in ((s.image, "images"), (s.label, "labels")):
            dst = OUT / split / kind / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    write_yaml(OUT / "data.yaml", OUT, "train/images", "valid/images")
    write_yaml(ORIG / "data.yaml", RAW, "train/images", "valid/images")

    # original-split leakage flag per image (val image whose group also has an original-train member)
    orig_grp = defaultdict(set)
    for g, s in zip(groups, samples):
        orig_grp[g].add(s.split)
    manifest = ROOT / "results" / "split_manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "orig_split", "new_split", "group", "classes", "orig_val_has_train_sibling"])
        for i, (g, s) in enumerate(zip(groups, samples)):
            leaked = s.split == "valid" and len(orig_grp[g]) > 1
            w.writerow([s.name, s.split, "valid" if i in val_idx else "train", g,
                        "+".join(CLASS_NAMES[c] for c in sorted(s.classes)), int(leaked)])

    # report
    for split in ("train", "valid"):
        idx = [i for i in range(len(samples)) if (i in val_idx) == (split == "valid")]
        img_c = Counter(c for i in idx for c in samples[i].classes)
        box_c = Counter(b.cls for i in idx for b in samples[i].boxes)
        print(f"{split:5s} images={len(idx):4d} groups={len({groups[i] for i in idx}):3d} "
              + " ".join(f"{CLASS_NAMES[c]}: imgs={img_c[c]} boxes={box_c[c]}" for c in range(len(CLASS_NAMES))))
    moved = sum(1 for i, s in enumerate(samples) if (i in val_idx) != (s.split == "valid"))
    print(f"balance cost={cost:.4f}; straddling groups=0; {moved} images changed split vs original")
    print(f"wrote {OUT}/, {ORIG}/data.yaml, {manifest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
