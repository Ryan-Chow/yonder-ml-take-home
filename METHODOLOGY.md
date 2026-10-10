# Methodology

Mallet / bottle detector for the Yonder rover: YOLOv8n trained on the 999-image Roboflow sample, with a
leak-free grouped split, CLAHE preprocessing, an ablation with uncertainty estimates, and error analysis.

> **Who wrote what.** Section 1 (how to run it) was written by Claude Code. Sections 2, 3, and 4 are mine, co-written
> with Claude Code, which helped me rewrite and expand my thinking. The technical details under section 2.
> `AI_LOG.md` summarises how I used AI

## 1. How to run it

**Tested on:** macOS 26.6.2 (Apple M5, 24 GB), Python 3.12.15, PyTorch 2.14.1 (MPS), Ultralytics 8.4.175.
Any Python >= 3.12 should work.

### Install

```bash
git clone https://github.com/Ryan-Chow/yonder-ml-take-home.git
cd yonder-ml-take-home
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # every library pinned to an exact version
```

### Dataset (needs your Roboflow key)

The download script reads the key from the **`ROBOFLOW_API_KEY`** environment variable. If that is not set, it
falls back to a git-ignored `.env` file; see `.env.example`. The key is never printed, and any error text has it
redacted.

```bash
export ROBOFLOW_API_KEY=your_key_here
python scripts/download_data.py        # -> Sampled-YD-Object-Detection-2/ (skipped if already present)
```

### Inference on a folder of images (the deliverable)

```bash
python predict.py path/to/images path/to/output
python predict.py path/to/images path/to/output --save-vis      # also writes <stem>.jpg with boxes drawn
```

- **Output files.** For every image, `predict.py` writes `path/to/output/<same_stem>.txt` with one line per
  detection: `class_id x_center y_center width height confidence`.
  - Coordinates are normalised to 0–1, like the dataset labels.
  - Class ids: `0 = bottle`, `1 = mallet`.
  - An image with no detections gets an empty file.
- **Threshold.** The default confidence threshold is **0.441**, read from `results/threshold.json`. It is the
  confidence that maximises mean F1 on the grouped val split; override it with `--conf`.
- **Weights.** The default weights are `weights/best.pt` (6.5 MB, committed).
- **Preprocessing.** Images go through the same preprocessing as training (`src/preprocess.py`): stretch to 512×512,
  then CLAHE. Any input size works.
- **Device.** `--device cpu` is the default.

### Everything end to end

```bash
./run_all.sh                 # venv -> install -> download -> audit -> split -> CLAHE -> smoke test -> train the
                             # 2x2 ablation + leak run -> bootstrap -> ship.py -> evaluate -> error analysis /
                             # hard examples / video -> predict demo -> tests
SKIP_TRAIN=1 ./run_all.sh    # same, but evaluates the committed weights instead of training (~2 min)
```

- Device selection is automatic: CUDA, then MPS, then CPU.
- Full training is 5 runs of about 30-50 min each on an M5 laptop (MPS), roughly 3.5 h in total; on a GPU, a few
  minutes per run.
- `colab.ipynb` runs the same pipeline on a Colab GPU and asks for the key with `getpass`.
- I checked `SKIP_TRAIN=1 ./run_all.sh` in a fresh `git clone`. It reproduced every committed result file byte
  for byte; see `docs/AGENT_RUN_LOG.md` entry 14.
- The efficiency study is also separate, takes about 10 min, and must run with nothing else using the machine:
  `python scripts/benchmark.py`.

### Step by step (what `run_all.sh` does)

| step | command | output |
|---|---|---|
| audit + leakage check | `python scripts/audit_data.py` | `results/audit.md`, `audit_plots.png`, `audit_samples.jpg`, `leakage_pairs.jpg` |
| grouped split | `python scripts/make_split.py` | `data/split/`, `data/orig/`, `results/split_manifest.csv` |
| preprocessing | `python -m src.preprocess --src data/split --dst data/split_clahe` (and `data/orig` → `data/orig_clahe`) | CLAHE PNGs |
| training | `python scripts/train.py --run {baseline,clahe_only,aug_only,final,leak}` (`--smoke` first) | `runs/<run>/` |
| ablation + uncertainty | `python scripts/bootstrap_ablation.py` | `results/ablation.md` (paired cluster bootstrap) |
| choose the shipped model | `python scripts/ship.py` | `weights/best.pt`, `results/shipped.json` |
| evaluation (CPU) | `python scripts/evaluate.py` | `results/metrics.md`, `metrics.json`, `threshold.json`, PR curves, confusion matrix |
| error analysis | `python scripts/error_analysis.py` | `results/errors.md`, `results/errors/` (every mistake) |
| hard examples | `python scripts/hard_examples.py` | `results/hard_examples.md`, `.csv`, `.jpg` |
| video | `python scripts/video_test.py media/test_video.mp4 --every 20` | `results/video.md`, `results/video/` |
| efficiency | `python scripts/benchmark.py` | `results/efficiency.md` |
| tests | `python -m pytest -q` | 28 tests, covering:<br>- output format<br>- training PNGs equal inference pixels<br>- matcher<br>- AP equal to Ultralytics<br>- `predict.py` reproduces the reported metrics |

## 2. Thought process (my summary)

- **Looking at the data first:** The images varied a lot in lighting (dark, bright, low contrast),
  and a lot were already rotated. That's why we added CLAHE and brightness/rotation augmentation.

- **Leakage:** The default split had near-copies of the same scene in both train and val
  (back-to-back video frames, photos from the same shoot). 54% of val images had a near-copy in train.
  That makes the val score look better than the model really is, because it's being tested on
  things it basically already saw.

- **How I split instead:** We grouped similar photos together, and each group goes entirely into
  train or entirely into val. Scores went down (0.883 vs 0.914 mAP50 on the leaky split), but the
  lower number is the honest one.

- **Preprocessing:** We used CLAHE to deal with the lighting variation, plus augmentations for rover
  conditions (brightness, small rotations, no vertical flip since the camera is never upside down).
  In the end, neither clearly helped on this val set.

- **Which model shipped:** All three top models were basically tied. The rule (set before seeing
  results) was: mAP50 first, which knocked out the baseline model, then mAP50-95 to break the tie
  among the rest. That picked clahe_only, even though it doesn't win on every number.

- **Metrics:** Detection needs per-class precision, recall and mAP, not accuracy. One overall number
  would hide that bottles are much worse than mallets.

- **Where it fails:** Bottles were missed 25% of the time vs 7% for mallets. Misses were much more common in
  very dark or very bright images (30–35% vs 12% in normal light), plus one indoor hand-held video that was
  nothing like the training data.

- **Real video:** My hair dryer was never detected as a mallet, top-down angles failed on the water
  bottle, and the fan and hair dryer were sometimes called "bottle." Either the hair dryer just
  isn't close enough to a mallet, or the model only learned "orange mallet" (almost every training
  mallet is orange), which isn't great for general use.

- **If I kept going:** Film a real mallet outdoors at rover height from more angles, test whether
  it only recognizes orange mallets, and test speed and accuracy on the OrangePi NPU.

*Full technical details below were drafted by Claude Code and reviewed by me.*

### Technical details

**Shipped model** (`weights/best.pt`, YOLOv8n, CLAHE + default augmentation), leak-free grouped val, 199 images,
CPU, threshold 0.426:

| class | precision | recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| bottle | 0.747 | 0.746 | 0.747 | 0.773 | 0.442 |
| mallet | 0.955 | 0.926 | 0.940 | 0.971 | 0.629 |

**Ablation** (one training run per arm; 95% CIs from a cluster bootstrap over source groups,
`results/ablation.md`):

| arm | mAP50 | mAP50-95 | mean F1 |
|---|---|---|---|
| baseline (raw, default aug) | 0.875 (0.80–0.94) | 0.541 (0.48–0.61) | 0.851 |
| **clahe_only (shipped)** | 0.873 (0.80–0.94) | 0.536 (0.48–0.60) | 0.844 |
| aug_only (raw, custom aug) | 0.902 (0.84–0.95) | 0.521 (0.48–0.57) | 0.868 |
| final (CLAHE + custom aug) | 0.882 (0.80–0.95) | 0.490 (0.45–0.56) | 0.868 |

Key evidence, with where to find it:

- **Data** (`results/audit.md`): 999 images, all stretched to 512×512 by Roboflow. 1222 boxes: 685 bottle, 537
  mallet. About 18% of boxes are under 32 px, and 38% of images carry rotation corners. Labels were checked visually
  in `audit_samples.jpg`.
- **Leakage and split** (`results/audit.md`, `results/split_manifest.csv`):
  - Exact filename overlap is 0, because the sampling script keeps one copy per photo. Leakage comes through video
    frames, numbered photo series, and `-copy` duplicates.
  - Near-duplicates are found by filename families, capture-sequence adjacency, and a centre-crop perceptual hash
    (whole-image hashes were fooled by rotation corners). The sequence rule was checked by eye: 28 of 30 random
    pairs were the same scene.
  - 999 images form 532 groups, split 80/20 by group, with 20.0% / 20.1% of each class in val.
- **Leakage measured** (`results/metrics.md`): the same recipe trained on the original split scores 0.914 / 0.558
  (mAP50 / mAP50-95) vs. 0.883 / 0.490 on the grouped split. Inside that model, val images with a train sibling
  score 0.930 vs. 0.894 without.
- **Preprocessing** (`src/preprocess.py`, `results/clahe_stats.md`): CLAHE on LAB lightness only (colour change
  0.11/255). Dark images drop from 15% to 7%. The same function runs at training (saved as lossless PNG) and
  inference, and a test checks the pixels match. Non-square inputs are stretched to 512 like the Roboflow export.
- **Training:** 60 epochs, patience 15, AdamW, seed 0, on MPS (38 s per epoch vs. 66 s on CPU). Best epochs were
  55–60, and no run stopped early.
- **Evaluation:**
  - Validation runs at the deployed 512 input (`rect=False`); Ultralytics' default pads to 544.
  - MPS and CPU validation are identical, and all reported numbers are CPU.
  - The threshold is the max-mean-F1 confidence on val. It is tuned on the same set it is reported on, so P/R/F1
    are slightly optimistic.
  - An independent matcher and a from-scratch AP (tested equal to Ultralytics) re-derive the numbers, and
    `tests/test_consistency.py` proves `predict.py` reproduces them.
- **Errors** (`results/errors.md`, every mistake in `results/errors/`):
  - Bottle: 35 of 138 missed, 34 false positives. Mallet: 8 missed, 4 false positives.
  - Miss rate is 30–35% in dark or bright images vs. 12% in normal light.
  - One indoor hand-held video misses 12 of 13 frames.
  - False positives land on a blue barrel, product-page swatches, and collages.
  - 2 of 3 wrong-class errors are dataset mislabels (`t_bot_mal578`, `t_bot_mal581`).
- **Video** (`results/video.md`, `media/test_video.mp4`): 15 s indoor clip, 46 frames analysed. "Mallet" was
  predicted in 0 frames (the stand-in is a blue hair dryer, and 89% of training mallet boxes are orange). The water
  bottle was found in 28 frames, mostly in the first ~7 s. The earlier synthetic stand-in built from val images got
  83% / 94% frame recall, so real footage is much harder.
- **Efficiency** (`results/efficiency.md`):
  - 3.14 M params, 5.6 GFLOPs, 6.5 MB.
  - 12.8 ms in ONNX Runtime on CPU and 4.0 ms on MPS at 512.
  - Naive INT8 gives mAP 0. INT8 with the box decode kept in float loses only 0.003 mAP50 at 3.6 MB.
  - Input 320 is 2.6× faster but costs 0.041 mAP50.
- **Hard examples** (`results/hard_examples.md`):
  - Training images with an object under 32 px are hard 21% of the time vs. 2%, and multi-object images 24% vs. 3%.
  - Some of that is missing labels.
  - Data-collection priorities are listed there.

Long-form write-up of all of the above, with every check and mistake: `docs/AGENT_RUN_LOG.md`.

## 3. Known limitations

- **The video test is narrow.** It is one 15 s indoor clip, the mallet stand-in is a blue hair dryer (no mallet was
  available), and everything is within ~2 m. Outdoor, distant, and true-mallet performance are still unmeasured.
  This is the most important gap.
- **The uncertainty estimates are incomplete.** Each ablation arm is a single training run (seed 0), so the
  confidence intervals cover val-set noise but not training-seed noise.
  - MPS training is not bit-reproducible: PyTorch has no deterministic MPS kernel for two operations used here.
  - The val set is small (199 images in 102 source groups), so differences of a few mAP points are within noise.
    My next step would be 3 seeds per arm, or k-fold over groups.
- **The threshold is tuned on the reported val set.** There is no held-out test set, so P/R/F1 at 0.426 are
  slightly optimistic.
- **Same-origin grouping is heuristic.** Photos of the same place with unrelated filenames can still sit on both
  sides of the split, so even the grouped numbers may be a little optimistic.
- **The val labels are imperfect.** Two mallets are labelled as bottles, and some bottles have no label at all. I
  left the labels untouched so all arms are compared on the same ground truth; correcting the two mislabels would
  move 2 "wrong class" errors into true positives.
- **CLAHE stays in the shipped model despite no measured benefit.** It is there because of the pre-registered
  tie-break, not because it helps. It costs 0.6 ms per image. For deployment simplicity I would consider dropping
  it, which means shipping baseline or aug_only; the evidence doesn't separate those options.
- **Nothing ran on the real target.** The NPU, RKNN conversion, and power draw were not tested. INT8 was only
  measured with ONNX Runtime on a Mac CPU.
- **Not exercised here:**
  - The Roboflow download (no API key was available). Only its skip path and key redaction are tested.
  - The full-training path of `run_all.sh` as one script (each step was run separately).
  - `colab.ipynb`.
- **What I'd do next, in order:**
  1. Film the real mallet (or an orange stand-in) outdoors, at rover height, at several distances.
  2. Fix the label errors.
  3. Run multiple seeds.
  4. Collect small/distant and hard-negative data per the hard-example plan.
  5. Evaluate with the web screenshots and collages removed.
  6. Try imgsz 640 for small objects against its NPU cost.

## 4. My takeaways (Ryan)
- The biggest lesson was leakage: 54% of val had a near-copy in train, and fixing the split
  dropped mAP50 from 0.914 to 0.883. The default split would have made the model look better than it is.
- None of the preprocessing choices clearly helped. clahe_only shipped through a tie-break
  rule written in advance; I'd probably drop CLAHE for deployment simplicity.
- My video test showed the real gap: it never recognized my hair dryer as a mallet, struggled
  with top-down angles, and called my fan/hair dryer "bottle." It seems to have learned
  "orange mallet," not "mallet shape."
- Next I'd film a real mallet outdoors at rover height and collect more varied angles and
  hard negatives.
