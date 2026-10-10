# Methodology

Mallet / bottle detector for the Yonder rover: YOLOv8n trained on the 999-image Roboflow sample, with a
leak-free grouped split, CLAHE preprocessing, an ablation with uncertainty estimates, and error analysis.

> **Who wrote this.** The whole submission (code, experiments, this document) was produced by Claude Code (an AI
> coding agent) from one autonomous prompt by Ryan Chow. `docs/AGENT_RUN_LOG.md` gives the full prompt and records every
> mistake that was caught along the way.

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

## 2. Thought process

### Headline

The shipped model is `weights/best.pt`: YOLOv8n with CLAHE preprocessing and default augmentation. Numbers are on
the **leak-free grouped val split** (199 images), on CPU, at threshold 0.426.

| class | precision | recall | F1 | mAP50 | mAP50-95 | TP / FP / FN at 0.426 |
|---|---|---|---|---|---|---|
| bottle | 0.747 | 0.746 | 0.747 | 0.773 | 0.442 | 103 / 34 / 35 |
| mallet | 0.955 | 0.926 | 0.940 | 0.971 | 0.629 | 100 / 4 / 8 |
| mean | 0.851 | 0.836 | 0.844 | 0.872 | 0.536 | |

- **The mallet, the object the rover has to retrieve, is found reliably:** AP50 0.97, 8 of 108 missed.
- **Bottles are the weak class:** a quarter are missed, and most false positives are bottles.
- **The original Roboflow split would have overstated every number.** Section 2.1 explains why, and the error
  analysis (2.6) explains where the model fails.

### 2.1 Data: audit first, then a split that doesn't leak

- **I looked at the data before modelling** (`results/audit.md`):
  - 999 images, all 512×512, because Roboflow "stretched" them, so original aspect ratios are gone.
  - 1222 boxes: 685 bottle, 537 mallet. 90% of images contain a single object.
  - About 18% of boxes are smaller than 32 px.
  - Lighting varies widely: mean brightness spans almost the full range, 15% of train images are dark, 7% bright,
    and 13% low-contrast.
  - About 38% of images already carry Roboflow's rotation augmentation (black corners).
  - I drew the labels back onto 16 random images to check that label parsing was right before trusting anything
    downstream.
- **The leakage risk was real, but not where the task hinted.**
  - The task says images are augmented variants of shared source photos, with names like `<orig>.rf.<hash>`.
    `sampling/sample-roboflow.py` keeps only **one** `.rf.` copy per photo, so exact-name overlap between train and
    val is zero.
  - Same-origin images still cross the split through three routes, each caught by its own rule:
    - consecutive frames of the same video;
    - consecutive photo numbers or timestamps from the same shoot (`IMG_0619` / `IMG_0620`, or shots seconds apart);
    - `-copy` duplicates.
  - **107 of 199 original val images (54%) have a same-origin image in train.**
- **How the near-duplicate detector was built:** it combines filename families, capture-sequence adjacency, and a
  perceptual hash.
  - The obvious whole-image hash was fooled by the black rotation corners: unrelated rotated photos looked alike.
    So I hash the central 40% of the image instead.
  - I checked the sequence rule by eye on 30 random cross-split pairs: 28 were the same scene, and 2 were the same
    object and session with a changed background.
- **The split is group-aware and stratified.** All 999 images are pooled into 532 same-origin groups, and whole
  groups are assigned to val.
  - Val gets 20.0% of the images containing a bottle and 20.1% of those containing a mallet. No group straddles the
    split.
  - The split is seeded and deterministic, and its manifest is committed.
  - The original split is kept for comparison.
- **How much the leak inflates the numbers:** I trained the same recipe on Roboflow's original split.
  - It scores **mAP50 0.914 / mAP50-95 0.558** on the original val, vs. **0.883 / 0.490** on the grouped val.
  - Inside that model, val images *with* a train sibling score 0.930 mAP50 (recall 0.910), vs. 0.894 (recall 0.835)
    for those *without*.
  - Caveat: those two subsets also differ in content, so not all of that gap is leakage.

### 2.2 Preprocessing and augmentation

- **CLAHE on the L channel in LAB** (`src/preprocess.py`), chosen because of the lighting spread above.
  - CLAHE equalises *local* contrast, which is what a small orange or blue object loses in shade or glare.
  - Working in LAB leaves colour untouched. I measured this: the mean change in the colour channels is 0.11/255.
    Colour is the strongest cue for the mallet.
  - On the dataset, CLAHE cuts the share of dark images from 15% to 7% and low-contrast images from 13% to 5%.
  - It cannot rescue near-black images.
- **The same function runs at training and at inference.** Training images are written as lossless PNG, so the
  training pixels are exactly what `predict.py` computes in memory. A test checks this pixel for pixel.
- **Inputs are stretched to 512×512 before CLAHE.** This mirrors the Roboflow export: Ultralytics would otherwise
  letterbox a 16:9 video frame into a geometry the model never saw. Normalised output coordinates are unaffected.
- **Custom augmentation for outdoor use** (`configs/train.yaml`, every value commented):
  - brightness/exposure jitter raised to 0.5;
  - small rotations (±10°) for camera roll on uneven ground;
  - mild perspective for camera pitch;
  - horizontal flip only. There is no vertical flip, because a rover camera always sees sky up and ground down.
- **What the ablation said about these choices:** see 2.4. In short, neither CLAHE nor the custom augmentation
  produced a clear win.

### 2.3 Model and training

- **YOLOv8n, pretrained on COCO.** It is the size class that is realistic on the OrangePi NPU (3.1 M params,
  5.6 GFLOPs at 512), and with 999 images a bigger model is more likely to overfit than to help. 512 is the native
  resolution of the data.
- **Training settings:** 60 epochs, early stopping with patience 15, seed 0, AdamW (picked automatically by
  Ultralytics). Training ran on MPS, at 38 s per epoch vs. 66 s on CPU in a timing test.
  - No run stopped early. Best epochs fell between 55 and 60; the leak run was still improving at 60, so a longer
    budget might help slightly.
  - All runs dip sharply in epochs 2–5, while the learning rate warms up, and then recover. One-epoch test runs
    looked catastrophic for the same reason; this was not an MPS bug.

### 2.4 Ablation: does CLAHE help? Does the augmentation help?

The task's ablation was (a) baseline vs. (b) CLAHE + custom aug. When (b) came out *worse* at localisation, I added
the two missing cells to make a 2×2. I also computed **paired cluster-bootstrap** confidence intervals
(`results/ablation.md`): whole source groups are resampled, because errors are correlated within a source. Each arm
is one training run; all numbers are on grouped val.

| arm | CLAHE | custom aug | mAP50 (95% CI) | mAP50-95 (95% CI) | mean F1 |
|---|---|---|---|---|---|
| baseline | no | no | 0.875 (0.80–0.94) | 0.541 (0.48–0.61) | 0.851 |
| **clahe_only (shipped)** | yes | no | 0.873 (0.80–0.94) | 0.536 (0.48–0.60) | 0.844 |
| aug_only | no | yes | 0.902 (0.84–0.95) | 0.521 (0.48–0.57) | 0.868 |
| final (task's recipe) | yes | yes | 0.882 (0.80–0.95) | 0.490 (0.45–0.56) | 0.868 |

- **CLAHE has no measurable effect.** Its main effect is −0.011 mAP50 and −0.017 mAP50-95, both CIs spanning zero.
  My lighting-based reason for adding it did not show up as accuracy on this val set.
- **The custom augmentation leans toward finding more objects and drawing looser boxes.**
  - aug_only − baseline: **+0.028 mAP50 [+0.000, +0.049]**, which is borderline significant.
  - The same comparison gives −0.020 mAP50-95 [−0.050, +0.012], which is not significant.
  - Plausible cause (not tested): rotating a box makes its axis-aligned target larger.
- **The two changes combine badly.** The task's recipe (final) is worse than aug_only on both metrics.
- **How the shipped model was chosen.** *Before* the last two arms finished, I wrote down a rule: mAP50 matters
  most (the rover first has to *find* the object), and box tightness breaks ties. The precise rule: among arms
  whose mAP50 is not significantly below the best arm's, ship the one with the best mAP50-95.
  - Applied mechanically (`scripts/ship.py`), it picks **clahe_only**.
  - **To be clear:** clahe_only, aug_only, and final are statistically indistinguishable here, and clahe_only has
    the lowest point-estimate F1. The rule is a principled tie-break, not evidence that clahe_only is better.
  - If recall at the threshold mattered more to the team than box tightness, aug_only is the one I'd pick. Choosing
    it is a one-line `python scripts/ship.py --arm aug_only`, and `predict.py` adapts automatically, because each
    checkpoint records whether it was trained on CLAHE images.

### 2.5 Evaluation choices

- **Metrics, all per class:**
  - precision, recall, and F1 at the deployed threshold: what the rover actually experiences (missed mallets vs.
    chasing ghosts);
  - mAP50: detection quality independent of the threshold;
  - mAP50-95: box tightness, which matters when approaching or grasping.

  Accuracy is not meaningful for detection, and a single mean would hide that bottles are much harder than mallets.
- **Threshold = 0.426:** the confidence that maximises mean F1 on val (`results/threshold.json`). Because it is
  tuned on the same val set it is reported on, P/R/F1 are slightly optimistic. There is no separate test split in
  this sample.
- **Validation runs at the real input size.** Ultralytics' default validation pads 512 images to 544, an input size
  the deployed model never sees. I validate at `rect=False`, which is exactly the 512 input that `predict.py` uses.
  I found this when ONNX and PyTorch disagreed by 0.015 mAP50 on the same weights.
- **MPS vs. CPU validation:** identical to four decimals for the shipped model. All reported numbers are CPU.
- **Checks on the checks:**
  - An independent IoU matcher and a from-scratch AP (tested equal to Ultralytics' on random curves) re-derive the
    numbers through the deployment code path.
  - `tests/test_consistency.py` proves that `predict.py`'s output files reproduce the reported confusion matrix
    exactly.

### 2.6 Error analysis (`results/errors.md`, every mistake saved in `results/errors/`)

What the data shows, for the shipped model at 0.426 on grouped val:

- **Bottle is the problem class:** 35 of 138 bottles missed and 34 bottle false positives, vs. 8 missed and 4 false
  positives for mallets.
- **Exposure extremes:** miss rate is 35% in dark images and 30% in bright ones, vs. 12% at normal exposure.
- **One whole val video fails:** a green bottle held in someone's hand indoors (`green_nalgene_vid`) is missed in
  12 of 13 frames. The grouped split put the entire video in val, so nothing like it was in train.
- **Recurring false positives:**
  - a blue barrel on a rooftop, predicted as bottle;
  - colour-swatch circles and a "NEW!" badge on product-page screenshots;
  - thumbnails in white synthetic collages.
- **Wrong-class errors:** 2 of the 3 are not model errors. `t_bot_mal578` and `t_bot_mal581` are orange mallets
  labelled "bottle", while the neighbouring frames from the same shoot are labelled mallet; the model is right. The
  third is an extreme close-up of a mallet handle, predicted as bottle.
- **What I think is causing these** (hypotheses, not tested):
  - **Bottle is a much broader class than mallet.** In the audit samples the mallets are one kind of orange rubber
    mallet, whereas "bottle" covers many shapes, colours, and transparencies. The model leans on colour and
    silhouette, which explains the blue barrel and the swatches.
  - **Unseen context.** Hand-held and indoor shots barely exist in training once that video is held out.
  - **Off-domain images.** Product screenshots and collages are web-scraped content the rover will never see, yet
    they produce a disproportionate share of the errors.
  - **Light and dark extremes.** CLAHE improves local contrast, but it cannot recover clipped or near-black pixels.

### 2.7 Video test

**The real video** is `media/test_video.mp4`: Ryan's 15 s phone clip.
- Indoors on a concrete floor, portrait 1080×1920 at 60 fps.
- Objects: a white water bottle, and a **blue hair dryer** as the mallet stand-in (no mallet was available).
- Distractors: crocs, a backpack, a standing fan, chair legs.
- The last seconds show neither object.
- Re-encoded to H.264 with metadata (incl. GPS) and audio stripped.

To reproduce: `python scripts/video_test.py media/test_video.mp4 --every 20` (3 frames/s, 46 frames;
`results/video.md`, `results/video/`).

- **The mallet stand-in was never detected as a mallet (0 of 46 frames).** It was called "bottle" in about 6 of the
  30 saved frames instead. In training every mallet is the same orange rubber mallet, so the model seems to have
  learned "orange mallet", not "hammer-shaped object". A blue hair dryer is a hard stand-in, but the result fits the
  error analysis: the model leans on colour.
- **The real bottle is found well for the first ~7 s** (mid-distance and close).
- **It is then mostly missed**, at steep top-down angles, with motion blur, and near the bright ceiling-light
  reflection. Overall, "bottle" fires in 28 of 46 frames, and presence flickers between consecutive frames 24% of
  the time.
- **False "bottle" boxes:** the white standing fan (first frames) and the hair dryer. The empty-floor frames at the
  end are clean.
- **Sensitive to small pixel changes.** On the original 44 MB iPhone file the bottle count was 31/46 rather than
  28/46, and one extra false positive on a croc appeared at the end.
- **Verdict:** outside the clean training distribution the detector is usable for the bottle at moderate distance,
  and not robust to viewpoint, blur, or glare. For the mallet class it only knows the orange mallet. This test is
  indoors and short-range, so it says nothing yet about distant objects outdoors.
- **Earlier stand-in:** before the real video existed, a clip built from 30 grouped-val images with simulated blur
  and exposure gave 83% bottle and 94% mallet frame recall. The gap to the real video shows how much easier
  in-distribution images are.

### 2.8 Stretch goals

- **Efficiency** (`results/efficiency.md`): 3.14 M params, 5.6 GFLOPs, 6.5 MB.
  - Forward pass latency, batch 1, at 512: 28.7 ms on CPU (PyTorch), 12.8 ms in ONNX Runtime on CPU, and 4.0 ms on
    MPS.
  - The ONNX export is lossless.
  - **Naive INT8 quantisation destroys the model (mAP 0.000).** The detection head packs box coordinates (0–512) and
    class scores (0–1) into one tensor. Keeping that decode in float costs only 0.003 mAP50, at 3.6 MB.
  - Input 320 is 2.6× faster but loses 0.041 mAP50, mostly on bottles.
  - On the OrangePi I would quantise to INT8 with the box decode on the CPU, and choose the input size from the
    distance at which the mallet must be detected. Small objects are already this model's weak spot.
- **Hard examples** (`results/hard_examples.md`): 36 of 800 *training* images still fail after training.
  - Images whose smallest object is under 32 px are hard 21% of the time, vs. 2% otherwise.
  - Images containing both classes are hard 24% of the time, vs. 3% otherwise, partly because of **missing labels**
    (the model confidently finds unlabelled bottles).
  - The prioritised plan:
    1. Fix labels first.
    2. Collect distant small objects at rover height.
    3. Collect multi-object scenes.
    4. Collect exposure extremes and occlusion.
    5. Collect hard negatives (blue barrels, orange debris).
    6. Drop off-domain web images.
    7. Pick new frames by uncertainty and flicker on real rover footage.

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
