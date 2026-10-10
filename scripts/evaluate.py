"""Evaluate the ablation and the leakage comparison. Final numbers are computed on CPU.

    python scripts/evaluate.py [--skip-mps]

Writes results/metrics.md, results/metrics.json, results/threshold.json, results/pr_curves.png,
results/confusion_final.png. Ultralytics' validator gives mAP and the P/R/F1 curves; an independent matcher
(src/matching.py) re-counts TP/FP/FN at the chosen threshold as a cross-check and builds the confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import CLASS_NAMES, ROOT, load_split  # noqa: E402
from src.infer import IMGSZ, NMS_IOU, detect, is_finished, load_model  # noqa: E402
from src.matching import confusion  # noqa: E402

RESULTS = ROOT / "results"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
ABLATION = ("baseline", "final")
RUNS = {
    "baseline": dict(weights="runs/baseline/weights/best.pt", data="data/split/data.yaml", raw="data/split",
                     clahe=False, desc="(a) baseline: grouped split, raw images, default augmentation"),
    "final": dict(weights="weights/best.pt", data="data/split_clahe/data.yaml", raw="data/split", clahe=True,
                  desc="(b) final: grouped split, CLAHE, custom augmentation"),
    "leak": dict(weights="runs/leak/weights/best.pt", data="data/orig_clahe/data.yaml", raw=None, clahe=True,
                 desc="(c) final recipe trained on the ORIGINAL (leaky) Roboflow split"),
}


def ultra_val(weights: Path, data: Path, device: str, name: str) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import smooth

    # rect=False: Ultralytics' default rect val pads 512x512 images to 544x544 (pad=0.5 stride rounding in
    # data/build.py), an input size predict.py never sees. rect=False evaluates exactly the deployed 512 input and
    # is what Ultralytics itself uses for ONNX models, so PyTorch and ONNX numbers are comparable.
    m = YOLO(str(weights)).val(data=str(data), split="val", imgsz=IMGSZ, batch=16, rect=False, conf=0.001,
                               iou=NMS_IOU, device=device, plots=True, project=str(ROOT / "runs" / "eval"),
                               name=name, exist_ok=True, verbose=False)
    b = m.box
    i = int(smooth(b.f1_curve.mean(0), 0.1).argmax())  # same index Ultralytics uses for its P/R/F1
    per = {}
    for k, c in enumerate(b.ap_class_index):
        per[CLASS_NAMES[int(c)]] = {"precision": float(b.p[k]), "recall": float(b.r[k]), "f1": float(b.f1[k]),
                                    "mAP50": float(b.ap50[k]), "mAP50-95": float(b.ap[k])}
    return {"per_class": per, "mean": {"precision": float(b.mp), "recall": float(b.mr),
                                       "f1": float(np.mean(b.f1)), "mAP50": float(b.map50),
                                       "mAP50-95": float(b.map)},
            "conf_at_max_mean_f1": float(b.px[i]), "px": b.px.tolist(), "f1_curve": b.f1_curve.tolist(),
            "prec_values": b.prec_values.tolist(), "n_images": int(m.seen) if hasattr(m, "seen") else None,
            "save_dir": str(m.save_dir)}


def subset_yaml(base_yaml: Path, names: list[str], tag: str) -> Path:
    cfg = yaml.safe_load(base_yaml.read_text())
    root = Path(cfg["path"])
    lst = root / f"val_{tag}.txt"
    lst.write_text("\n".join(str(root / "valid" / "images" / n) for n in names) + "\n")
    out = root / f"val_{tag}.yaml"
    out.write_text(yaml.safe_dump(dict(cfg, val=lst.name), sort_keys=False))
    return out


def own_counts(weights: Path, raw_root: Path, clahe: bool, conf: float) -> dict:
    """Independent TP/FP/FN at `conf` by running the shared inference path on the RAW val JPEGs."""
    samples = load_split(raw_root, "valid")
    model = load_model(weights)
    preds = detect(model, [cv2.imread(str(s.image)) for s in samples], conf=conf, clahe=clahe, device="cpu")
    gts = [np.array([[b.cls, b.xc, b.yc, b.w, b.h] for b in s.boxes]).reshape(-1, 5) for s in samples]
    cm = confusion(preds, gts, nc=len(CLASS_NAMES))
    per = {}
    for c, n in enumerate(CLASS_NAMES):
        tp, fp, fn = cm[c, c], cm[c].sum() - cm[c, c], cm[:, c].sum() - cm[c, c]
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        per[n] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r,
                  "f1": 2 * p * r / max(p + r, 1e-9), "n_gt": int(cm[:, c].sum()),
                  "recall_95ci": 1.96 * np.sqrt(r * (1 - r) / max(cm[:, c].sum(), 1))}
    return {"per_class": per, "confusion": cm.tolist()}


def train_summary(run: str) -> dict:
    csv = ROOT / "runs" / run / "results.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    return {"epochs_run": int(df["epoch"].max()), "best_epoch": int(df.loc[fit.idxmax(), "epoch"]),
            "train_minutes": float(df["time"].max() / 60)}


def fmt(x):
    return f"{x:.3f}"


def plot_pr(res: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#fcfcfb")
    for k, name in enumerate(CLASS_NAMES):
        ax = axes[k]
        for run, col, ls in (("baseline", GRAY, "--"), ("final", BLUE, "-")):
            if run not in res:
                continue
            r = res[run]
            x = np.array(r["px"])
            ax.plot(x, np.array(r["prec_values"])[k], color=col, ls=ls, lw=2,
                    label=f"{run} (AP50 {r['per_class'][name]['mAP50']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{name}: precision-recall @ IoU 0.5 (grouped val)", fontsize=10, loc="left", color=INK)
        ax.set_xlabel("recall", fontsize=8, color=MUTED)
        ax.set_ylabel("precision", fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, conf: float, out: Path):
    labels = CLASS_NAMES + ["background"]
    fig, ax = plt.subplots(figsize=(4.6, 4), facecolor="#fcfcfb")
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == cm.shape[0] - 1 and j == cm.shape[1] - 1:
                continue
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.5 else INK)
    ax.set_xticks(range(3), labels, fontsize=8)
    ax.set_yticks(range(3), CLASS_NAMES + ["missed"], fontsize=8)
    ax.set_xlabel("true", fontsize=8, color=MUTED)
    ax.set_ylabel("predicted", fontsize=8, color=MUTED)
    ax.set_title(f"final, grouped val, conf>={conf:.2f}, IoU 0.5", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mps", action="store_true")
    ap.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS),
                    help="subset of runs to evaluate (e.g. skip 'leak' while it is still training)")
    a = ap.parse_args()
    res = {}
    for run, cfg in RUNS.items():
        if run not in a.runs:
            continue
        w = ROOT / cfg["weights"]
        if not is_finished(w):
            print(f"skip {run}: {w} missing or its training run has not finished")
            continue
        res[run] = ultra_val(w, ROOT / cfg["data"], "cpu", f"{run}_cpu")
        res[run]["train"] = train_summary(run)
        res[run]["desc"] = cfg["desc"]

    # leakage: the leak model scored on original-val images that do / don't have a same-origin train image
    if "leak" in res:
        man = pd.read_csv(RESULTS / "split_manifest.csv")
        val = man[man.orig_split == "valid"]
        stem = lambda n: Path(n).stem + ".png"  # noqa: E731
        for tag, flag in (("leaked", 1), ("clean", 0)):
            names = [stem(n) for n in val[val.orig_val_has_train_sibling == flag].name]
            y = subset_yaml(ROOT / RUNS["leak"]["data"], names, tag)
            res[f"leak_{tag}"] = ultra_val(ROOT / RUNS["leak"]["weights"], y, "cpu", f"leak_{tag}_cpu")
            res[f"leak_{tag}"]["n_images"] = len(names)

    # the shipped threshold is rounded to 3 decimals; every count below uses exactly that value, so predict.py
    # (which reads threshold.json) reproduces them detection for detection
    thr = round(res["final"]["conf_at_max_mean_f1"], 3)
    (RESULTS / "threshold.json").write_text(json.dumps({
        "conf": thr, "criterion": "confidence maximising the mean (over classes) F1 on the grouped val "
        "split, final model, CPU; same smoothed-F1 argmax Ultralytics uses for its P/R/F1",
        "per_class_f1_at_conf": {k: round(v["f1"], 4) for k, v in res["final"]["per_class"].items()}}, indent=2))

    own = {r: own_counts(ROOT / RUNS[r]["weights"], ROOT / RUNS[r]["raw"], RUNS[r]["clahe"],
                         round(res[r]["conf_at_max_mean_f1"], 3)) for r in ABLATION if r in res}

    mps = None
    if not a.skip_mps:
        mps = ultra_val(ROOT / RUNS["final"]["weights"], ROOT / RUNS["final"]["data"], "mps", "final_mps")

    # ---------------------------------------------------------------- report
    md = ["# Metrics", "", "Generated by `scripts/evaluate.py`. All numbers below are from **CPU** validation, IoU 0.5 "
          "for P/R/F1/mAP50, NMS IoU 0.7, imgsz 512.", "",
          "**Why these metrics.** A rover that must find the mallet cares about two different failures: "
          "missing it (recall) and chasing ghosts (precision). F1 at the deployed threshold summarises that "
          "trade-off at the operating point we actually ship; mAP50 summarises ranking quality across all "
          "thresholds; mAP50-95 adds localisation tightness (matters for grasping / approach). All are reported "
          "per class because a mean can hide one class failing.", "",
          "P, R and F1 are each taken at that model's own confidence that maximises mean F1 on val.", "",
          "## Ablation: per-class results on the grouped (leak-free) val split", "",
          "| run | class | precision | recall | F1 | mAP50 | mAP50-95 | conf | best epoch / run |",
          "|---|---|---|---|---|---|---|---|---|"]
    for run in [r for r in ABLATION if r in res]:
        r = res[run]
        t = r.get("train", {})
        for n in CLASS_NAMES + ["mean"]:
            v = r["per_class"][n] if n != "mean" else r["mean"]
            md.append(f"| {run} | {'**' + n + '**' if n == 'mean' else n} | {fmt(v['precision'])} | "
                      f"{fmt(v['recall'])} | {fmt(v['f1'])} | {fmt(v['mAP50'])} | {fmt(v['mAP50-95'])} | "
                      f"{r['conf_at_max_mean_f1']:.3f} | {t.get('best_epoch', '?')}/{t.get('epochs_run', '?')} |")
    if "baseline" in res:
        d = {k: res["final"]["mean"][k] - res["baseline"]["mean"][k] for k in ("f1", "mAP50", "mAP50-95")}
        md += ["", f"Final minus baseline (mean over classes): F1 {d['f1']:+.3f}, mAP50 {d['mAP50']:+.3f}, "
               f"mAP50-95 {d['mAP50-95']:+.3f}."]
    md += ["",
           "![PR curves](pr_curves.png)", "",
           "## Deployed threshold", "",
           f"`results/threshold.json`: **conf = {thr:.3f}** (max mean F1 on grouped val, final model). "
           "Caveat: it is chosen on the same val set the numbers are reported on (there is no separate test "
           "set in this 999-image sample), so P/R/F1 are slightly optimistic; mAP is threshold-free.", "",
           "## Independent cross-check at the deployed threshold", "",
           "Re-counted with `src/matching.py` (greedy, IoU >= 0.5) by running the *deployment* path "
           "(`src/infer.py`: raw JPEG -> `preprocess()` in memory -> YOLO) on the raw val images, not "
           "Ultralytics' validator on the pre-made PNGs. Agreement with the table above shows predict.py "
           "reproduces the reported metrics.", "",
           "| run | class | GT | TP | FP | FN | precision | recall (±95% CI) | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for run in own:
        for n in CLASS_NAMES:
            v = own[run]["per_class"][n]
            md.append(f"| {run} | {n} | {v['n_gt']} | {v['tp']} | {v['fp']} | {v['fn']} | {fmt(v['precision'])} | "
                      f"{fmt(v['recall'])} ± {v['recall_95ci']:.3f} | {fmt(v['f1'])} |")
    cm = np.array(own["final"]["confusion"])
    plot_confusion(cm, thr, RESULTS / "confusion_final.png")
    md += ["", "Confusion matrix (final, rows = predicted, cols = true; 'missed' = GT with no prediction, "
           "'background' = prediction on no object):", "",
           "| predicted \\ true | bottle | mallet | background |", "|---|---|---|---|"]
    for i, n in enumerate(CLASS_NAMES + ["missed"]):
        md.append(f"| {n} | " + " | ".join(str(cm[i, j]) if not (i == 2 and j == 2) else "–" for j in range(3)) + " |")
    md += ["", "![confusion](confusion_final.png)", ""]

    if "leak" in res:
        md += ["## Leakage inflation", "",
               "Run (c) uses the final recipe on Roboflow's original split. Its val set is broken down by whether "
               "each image has a same-origin image in train (results/audit.md). Same model on both subsets; what "
               "differs is whether a near-duplicate of the image was in its training set.", "",
               "| model / val subset | images | mean P | mean R | mean F1 | mAP50 | mAP50-95 | bottle mAP50 | mallet mAP50 |",
               "|---|---|---|---|---|---|---|---|---|"]
        rows = [("(c) leak model, all original val", res["leak"], 199),
                ("(c) leak model, val images WITH train sibling", res["leak_leaked"], res["leak_leaked"]["n_images"]),
                ("(c) leak model, val images WITHOUT sibling", res["leak_clean"], res["leak_clean"]["n_images"]),
                ("(b) final model, grouped val", res["final"], 199)]
        for label, r, n in rows:
            m = r["mean"]
            md.append(f"| {label} | {n} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                      f"{fmt(m['mAP50'])} | {fmt(m['mAP50-95'])} | {fmt(r['per_class']['bottle']['mAP50'])} | "
                      f"{fmt(r['per_class']['mallet']['mAP50'])} |")
        infl = res["leak"]["mean"]["mAP50"] - res["final"]["mean"]["mAP50"]
        gap = res["leak_leaked"]["mean"]["mAP50"] - res["leak_clean"]["mean"]["mAP50"]
        md += ["", f"- Reported-number inflation from using Roboflow's split: mAP50 {infl:+.3f} "
               "(original val vs grouped val, same recipe).",
               f"- Within the same model: leaked minus clean subset mAP50 = {gap:+.3f}. Caveat: the subsets also "
               "differ in content (the leaked ones are mostly multi-shot outdoor sessions), so this gap is not "
               "*only* leakage.", ""]

    if mps:
        md += ["## MPS vs CPU validation (final model, grouped val)", "",
               "| device | mean P | mean R | mAP50 | mAP50-95 | bottle mAP50 | mallet mAP50 | conf@maxF1 |",
               "|---|---|---|---|---|---|---|---|"]
        for dev, r in (("cpu", res["final"]), ("mps", mps)):
            m = r["mean"]
            md.append(f"| {dev} | {m['precision']:.4f} | {m['recall']:.4f} | {m['mAP50']:.4f} | "
                      f"{m['mAP50-95']:.4f} | {r['per_class']['bottle']['mAP50']:.4f} | "
                      f"{r['per_class']['mallet']['mAP50']:.4f} | {r['conf_at_max_mean_f1']:.3f} |")
        md += ["", f"Max absolute difference in mean mAP50-95: "
               f"{abs(mps['mean']['mAP50-95'] - res['final']['mean']['mAP50-95']):.4f}. CPU numbers are the "
               "ones reported everywhere else.", ""]

    plot_pr(res, RESULTS / "pr_curves.png")
    (RESULTS / "metrics.md").write_text("\n".join(md) + "\n")
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("px", "f1_curve", "prec_values")}
            for k, v in res.items()}
    if mps:
        slim["final_mps"] = {kk: vv for kk, vv in mps.items() if kk not in ("px", "f1_curve", "prec_values")}
    slim["own_matching"] = own
    (RESULTS / "metrics.json").write_text(json.dumps(slim, indent=2, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
