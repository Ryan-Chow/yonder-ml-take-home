"""Run the detector on frames of a video (README Part 1, item 6).

    python scripts/video_test.py data/video/test.mov --every 10
    python scripts/video_test.py --every 3              # no real video: builds a stand-in clip from val images

Extracts every N-th frame, runs the same pipeline as predict.py (stretch to 512x512 + CLAHE + YOLO), writes
annotated frames to results/video/ (at most --max-save, evenly spaced, so git stays small), a per-frame confidence
plot, and results/video.md.

Without a real video, the stand-in clip is made from grouped-VAL images (never trained on) with simulated camera
motion (slow zoom/pan), motion blur and exposure drift, at 1280x720 so the non-square path is exercised. Because
those frames come from labelled images, frame-level recall can be measured; for a real video only detection rate,
confidence and temporal stability are reported (no ground truth).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import draw  # noqa: E402
from src.data_utils import CLASS_NAMES, ROOT, load_split  # noqa: E402
from src.infer import detect, load_model  # noqa: E402

RESULTS = ROOT / "results"
REAL_VIDEO = ROOT / "data" / "video" / "test.mov"
DEMO_VIDEO = ROOT / "data" / "video" / "demo_from_val.mp4"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def build_demo_clip(out: Path, n_images=30, frames_per_image=9, size=(1280, 720), seed=0) -> dict:
    """Stand-in video from grouped-val images. Returns {frame_index: set(GT classes)} for frame-level recall."""
    rng = np.random.default_rng(seed)
    samples = load_split(ROOT / "data" / "split", "valid")
    pick = rng.choice(len(samples), n_images, replace=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    gt, f = {}, 0
    for i in pick:
        s = samples[i]
        base = cv2.resize(cv2.imread(str(s.image)), size, interpolation=cv2.INTER_LINEAR)  # 16:9 distortion
        gain = rng.uniform(0.6, 1.4)  # per-shot exposure
        for t in range(frames_per_image):
            z = 1.0 + 0.12 * t / frames_per_image  # slow zoom toward the centre
            M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), 0, z)
            fr = cv2.warpAffine(base, M, size, borderMode=cv2.BORDER_REFLECT)
            k = int(rng.integers(1, 6)) * 2 + 1
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2] = 1.0 / k  # horizontal motion blur
            fr = cv2.filter2D(fr, -1, kernel)
            fr = np.clip(fr.astype(np.float32) * (gain + 0.02 * t), 0, 255).astype(np.uint8)
            vw.write(fr)
            # keep only GT objects whose centre is still in frame after the zoom
            keep = {b.cls for b in s.boxes
                    if abs((b.xc - 0.5) * z) < 0.5 and abs((b.yc - 0.5) * z) < 0.5}
            gt[f] = keep
            f += 1
    vw.release()
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", type=Path)
    ap.add_argument("--every", type=int, default=10, help="use every N-th frame")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    ap.add_argument("--conf", type=float)
    ap.add_argument("--max-save", type=int, default=30)
    a = ap.parse_args()
    conf = a.conf if a.conf is not None else json.loads((RESULTS / "threshold.json").read_text())["conf"]

    gt = None
    video = a.video or (REAL_VIDEO if REAL_VIDEO.exists() else None)
    if video is None:
        video = DEMO_VIDEO
        gt = build_demo_clip(DEMO_VIDEO)
        print(f"no real video found at {REAL_VIDEO}; built stand-in clip {DEMO_VIDEO}")
    is_demo = gt is not None

    cap = cv2.VideoCapture(str(video))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # respect phone rotation metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, idx, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            frames.append(fr)
            idx.append(i)
        i += 1
    cap.release()
    if not frames:
        sys.exit(f"could not read frames from {video}")
    h, w = frames[0].shape[:2]

    dets = detect(load_model(a.weights), frames, conf=conf, clahe=True, device="cpu")
    has = {c: np.array([any(int(d[0]) == c for d in D) for D in dets]) for c in range(len(CLASS_NAMES))}
    maxconf = {c: np.array([max([d[5] for d in D if int(d[0]) == c], default=0.0) for D in dets])
               for c in range(len(CLASS_NAMES))}

    out_dir = RESULTS / "video"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    keep = np.unique(np.linspace(0, len(frames) - 1, min(a.max_save, len(frames))).astype(int))
    for k in keep:
        vis = draw(frames[k].copy(), dets[k])
        scale = 640 / max(vis.shape[:2])
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
        if is_demo:
            label = "GT: " + (", ".join(CLASS_NAMES[c] for c in sorted(gt[idx[k]])) or "none")
            cv2.putText(vis, label, (6, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"frame_{idx[k]:05d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#fcfcfb")
    for c, col in ((0, BLUE), (1, ORANGE)):
        ax.plot(idx, maxconf[c], color=col, lw=2, label=f"max {CLASS_NAMES[c]} conf")
    ax.axhline(conf, color=MUTED, lw=0.8, ls="--")
    ax.text(idx[-1], conf + 0.02, f"threshold {conf:.2f}", ha="right", fontsize=7, color=MUTED)
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame", fontsize=8, color=MUTED)
    ax.set_ylabel("confidence", fontsize=8, color=MUTED)
    ax.set_title(f"Per-frame max confidence: {video.name}", fontsize=10, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "video_conf.png", dpi=110)
    plt.close(fig)

    def flicker(x):  # share of consecutive sampled frames where presence toggles
        return float(np.mean(x[1:] != x[:-1])) if len(x) > 1 else 0.0

    md = ["# Video test", ""]
    if is_demo:
        md += ["> **Stand-in only: the real phone video is still pending.** No `data/video/test.mov` was available "
               "when this ran. This clip was synthesised from 30 grouped-*val* images (never trained on), resized to "
               "1280x720, with simulated zoom, horizontal motion blur and exposure drift (gain 0.6-1.4). It tests "
               "the frame pipeline and robustness to blur/exposure, **not** real out-of-distribution performance. "
               "To run the real test: put the video at `data/video/test.mov` and run "
               "`python scripts/video_test.py data/video/test.mov --every 10`.", ""]
    md += [f"- Source: `{video.relative_to(ROOT) if video.is_relative_to(ROOT) else video}`, {w}x{h}, "
           f"{fps:.1f} fps, {total} frames; every {a.every}-th frame -> {len(frames)} frames analysed.",
           f"- Pipeline: identical to `predict.py` (stretch to 512x512 + CLAHE), conf >= {conf:.3f}, CPU.", "",
           "| class | frames with >=1 detection | median max-conf when detected | presence flicker |",
           "|---|---|---|---|"]
    for c, n in enumerate(CLASS_NAMES):
        det_conf = maxconf[c][has[c]]
        md.append(f"| {n} | {has[c].sum()}/{len(frames)} ({has[c].mean() * 100:.0f}%) | "
                  f"{np.median(det_conf) if len(det_conf) else 0:.2f} | {flicker(has[c]) * 100:.0f}% |")
    md += ["", "Presence flicker = share of consecutive analysed frames where the class appears/disappears.", ""]
    if is_demo:
        rows = []
        for c, n in enumerate(CLASS_NAMES):
            present = np.array([c in gt[j] for j in idx])
            tp = (present & has[c]).sum()
            fp = (~present & has[c]).sum()
            rows.append(f"| {n} | {present.sum()} | {tp} ({tp / max(present.sum(), 1) * 100:.0f}%) | {fp} |")
        md += ["Frame-level check against the source labels (is the class detected in frames where it is present?):",
               "", "| class | frames where present | detected | frames with a detection but class absent |",
               "|---|---|---|---|", *rows, ""]
    md += ["![confidence over time](video_conf.png)", "",
           f"Annotated frames (evenly spaced, {len(keep)} saved): `results/video/`."]
    (RESULTS / "video.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
