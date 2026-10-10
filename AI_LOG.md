# AI Usage Log

**Attribution, up front:** this entire submission (code, experiments, write-ups, commits) was produced by
**Claude Code** (Anthropic's CLI agent, model Claude Opus 5.5) running autonomously from a prompt written by
Ryan Chow. Ryan did not hand-edit code during the run. Every entry below is written by the agent and describes
what it did, where it changed its mind, and which checks caught its own mistakes. "I" below means Claude Code.

## 1. One autonomous prompt to build the whole pipeline

**What was asked:** Ryan gave Claude Code one autonomous prompt to build the pipeline end to end: verify the
dataset, audit it with a leakage check, make a grouped split, add CLAHE preprocessing, train YOLOv8n with an
ablation, evaluate, write an inference script and tests, do error analysis, run a video test, attempt the
efficiency and hard-example stretch goals, write run_all.sh and a Colab notebook, fill in METHODOLOGY.md, commit as
it went, keep this log honest, and push. That prompt is below, verbatim.

<details>
<summary>The prompt that drove this run (verbatim)</summary>

```text
You're completing the Yonder Dynamics ML take-home, FULLY AUTONOMOUSLY. I will not be available to answer questions. Make reasonable decisions, record them, and keep going. Do not stop until every step is done or truly blocked; if blocked, skip that step, record why, and continue.

## Step 0: Repo setup
- The repo is already cloned at ~/Projects/yonder-ml-take-home, with the dataset already unzipped at ./Sampled-YD-Object-Detection-2 (data.yaml, train/, valid/). cd into it and do ALL work there. Verify the dataset folder structure and image/label counts (800 train, 199 valid) before continuing.
- Ensure branch is main (`git branch -M main`).
- Create my public GitHub repo (gh CLI is authenticated): `gh repo create Ryan-Chow/yonder-ml-take-home --public --description "Yonder Dynamics ML take-home: mallet/bottle detection"`. If the name is taken, append -2.
- `git remote rename origin upstream`, then `git remote add origin https://github.com/Ryan-Chow/yonder-ml-take-home.git`.
- No Roboflow key is available. Don't run any download.
- If ~/Downloads/test.mov exists, copy it to data/video/test.mov.

Then read README.md, METHODOLOGY.md, AI_LOG.md (template), requirements.txt, .gitignore and sampling/. Follow every README rule. Reviewers grade process, design judgment, metric choice, error analysis, commit history, and evidence that AI output was verified.

## Machine
Apple Silicon MacBook, 24GB unified memory, macOS. Use Python 3.12+ (`brew install python@3.12` if needed) in .venv.
- Train with device="mps", PYTORCH_ENABLE_MPS_FALLBACK=1, batch 32 (16 if memory errors), workers 4, imgsz 512.
- If MPS errors, gives NaN losses, or is slower than CPU in a 1-epoch timing test, fall back to CPU and log it.
- Run FINAL evaluation of the chosen model on CPU, and compare one MPS vs CPU val run; log any difference (Ultralytics has had MPS-specific val discrepancies).
- Run long training in the background with output to logs/train_<name>.log, poll it, and build other scripts while it trains.

## Hard rules
- Never modify sampling/ or Sampled-YD-Object-Detection-2/. Derived data goes in data/ (git-ignored).
- Pin every added library in requirements.txt with exact versions.
- Smoke test (1 epoch, ~20 images) before any full training run.
- Don't claim anything works unless you ran it. If a metric looks too good (mAP50 > 0.95), suspect leakage or bugs and investigate.

## Commits
Commit after EVERY meaningful unit of work (aim for 20+ commits), small and descriptive, e.g. "add leakage check to audit", "fix label parsing for multi-object images". Bug fixes get their own commits. `git status` before every `git add -A`. Never squash, amend, rebase or force-push. Never commit .env, Sampled-YD-Object-Detection-2/, data/, runs/, logs/, DEV_NOTES.md. DO commit weights/best.pt (YOLOv8n ~6MB) and results/ (markdown, json, small pngs; keep results under ~20MB).

## AI_LOG.md (follow the repo template and README Part 3 exactly)
Update AI_LOG.md AS YOU GO, in the same commit as the work it describes. It must be HONEST and correctly attributed: this project is being built by Claude Code (you) from a single autonomous prompt by Ryan. Complete ALL sections of AI_LOG.md fully on Ryan's behalf—do not leave any TODO placeholders for Ryan.
- Entry 1: Ryan gave Claude Code one autonomous prompt to build the pipeline end to end; paste this full prompt inside a collapsible <details> block.
- Then one entry per significant step, with the README's fields:
  - What was asked: the sub-task and which part of the prompt it came from.
  - What was kept vs rewritten, and why: your own iterations (first approach, why replaced, what stayed), with commit hashes.
  - What the AI got wrong / had to be caught: every real bug, bad assumption, failed check, or misleading number, and which test/check/plot caught it. Be specific and truthful; don't invent errors and don't hide real ones.
  - Verification & Final Review: Provide a thorough, critical assessment and verification analysis of the step's code and outputs as the reviewing engineer.
Keep entries concise.

## Steps
1. Setup: venv, requirements.txt, .env.example (ROBOFLOW_API_KEY=), .gitignore additions.
2. scripts/download_data.py for reviewers: reads ROBOFLOW_API_KEY from env (python-dotenv fallback), downloads workspace "malletbottle2", project "sampled-yd-object-detection", version 2, format "yolov8"; skips if the folder exists; never prints the key and sanitizes exceptions; truststore fallback for SSL. Don't run the download (data is present); just verify the skip path works.
3. scripts/audit_data.py → results/audit.md + plots: counts per class/split, box size/aspect, objects per image, brightness/contrast distributions, and results/audit_samples.jpg (GT boxes on 16 random images) to verify label parsing. LEAKAGE CHECK: images are augmented variants of shared source photos (names like `<orig>_jpg.rf.<hash>.jpg`); group by original stem AND perceptual hash (imagehash phash, small Hamming threshold); report how many val images have a sibling in train.
4. scripts/make_split.py: group-aware, class-stratified ~80/20 split with fixed seed → data/split/ + data.yaml (no test line). Keep the original split for comparison.
5. src/preprocess.py: ONE shared function used for both training-data prep and inference: CLAHE on the L channel in LAB space, justified by the audit's lighting stats. Write to data/split_clahe/. configs/train.yaml with deliberate outdoor-oriented augmentations (HSV, brightness, small rotation, scale, translate, fliplr, mosaic; no vertical flip unless justified), each commented with why.
6. scripts/train.py: YOLOv8n pretrained, fixed seed, early stopping. Ablation: (a) baseline (grouped split, default aug, no CLAHE) vs (b) final (grouped split + CLAHE + custom aug). ~60 epochs, patience 15. Copy final best.pt to weights/best.pt.
7. scripts/evaluate.py → results/metrics.md + metrics.json: per-class precision, recall, F1, mAP50, mAP50-95, confusion matrix, PR curves; baseline vs final side by side, plus final on the original (leaky) split to quantify leakage inflation. Pick the confidence threshold maximizing mean F1 on val → results/threshold.json.
8. predict.py (root): `python predict.py <input_dir> <output_dir> [--weights weights/best.pt] [--conf T] [--save-vis]`. Same preprocess. Per image writes `<same_stem>.txt` with lines `class_id x_center y_center width height confidence`, normalized 0-1 (bottle=0, mallet=1); empty file if no detections. tests/test_predict_format.py checks format, ranges, filename matching. Run it.
9. scripts/error_analysis.py: match preds to GT at IoU 0.5 on val; save every FP/FN/wrong-class to results/errors/ (GT green, pred red) and results/errors.md with counts by type, class, box size, brightness bucket. Report patterns as data; put hypotheses about causes in DEV_NOTES.md only, labeled as hypotheses for Ryan to check.
10. scripts/video_test.py `<video> --every N`: extract frames, run predict, annotated frames to results/video/ (≤30 in git), summary in results/video.md. Use data/video/test.mov if present; otherwise test on a short clip built from val images and note the real video is pending.
11. Stretch, efficiency: scripts/benchmark.py: params, GFLOPs, size, mean/p95 latency on CPU and MPS at 512 and 320. Export ONNX; try INT8 (ONNX Runtime quantization or CoreML, whichever works on Mac) and re-evaluate mAP. results/efficiency.md table. Note the real target is RKNN on an OrangePi NPU (not testable here) and discuss tradeoffs.
12. Stretch, hard examples: scripts/hard_examples.py: rank train images by low GT confidence/misses; contact sheet + results/hard_examples.md grouped by failure type, with a data-collection prioritization strategy.
13. run_all.sh: fresh clone → venv → install → download → audit → split → preprocess → train → evaluate → predict demo. Plus colab.ipynb (getpass for key) for reviewers with a GPU.
14. METHODOLOGY.md: fill in "How to run" completely (commands, Python version, macOS version, env var name, exact inference command, threshold). Fully complete the thought-process and methodology sections, writing clear engineering rationales for all design choices and findings.

## DEV_NOTES.md (git-ignored, for running notes)
Running notes: every decision and its rationale, actual numbers, error-cause hypotheses, anything uncertain, and a checklist of specific things verified during execution.

## Final step: push
- Run tests and `git status`. `git ls-files | grep -Ei "\.env$|Sampled-YD|^data/|^runs/|^logs/"` must be empty. Every file under 100MB.
- `git push -u origin main` (never --force).
- Verify public: `curl -s -o /dev/null -w "%{http_code}" https://github.com/Ryan-Chow/yonder-ml-take-home` returns 200.
- Print: repo URL, per-class metrics table (baseline vs final), the leakage finding, commit count, and a summary of completed execution steps.

[line Ryan typed under the prompt] /Users/rchow/Downloads/Sampled\ YD\ Object\ Detection.v2i.yolov8.zip  and attached is the .zip file for such
```
</details>

There was an **earlier session**, driven by a previous version of the same prompt. That version said to clone the
repo and use a `ROBOFLOW_API_KEY` from the shell. It did Steps 0–2 (commits `0a04297`, `9924a18`) and then stopped.
I kept its work, and its prompt is here so the history is complete:

<details>
<summary>Earlier version of the prompt (verbatim, drove commits 0a04297 and 9924a18)</summary>

```text
## Step 0: Repo setup (do this first)
- Working directory is ~/Projects. Clone: `git clone https://gitlab.com/Yonder-Dynamics/take-home-projects/ai-ml-take-home.git yonder-ml-take-home` and cd into it. Do ALL work inside ~/Projects/yonder-ml-take-home.
- Make sure the branch is named main (`git branch -M main` if needed).
- Create my public GitHub repo with the gh CLI (already authenticated): `gh repo create Ryan-Chow/yonder-ml-take-home --public --description "Yonder Dynamics ML take-home: mallet/bottle detection"`. If the name is taken, append -2.
- Point origin at it: `git remote set-url origin https://github.com/Ryan-Chow/yonder-ml-take-home.git`. Keep the Yonder GitLab URL as a remote named `upstream`.
- ROBOFLOW_API_KEY is already set as an environment variable in this shell. Code reads it from the environment (python-dotenv as optional fallback). Never write the key to any file, including .env, and never echo it.
- If ~/Downloads/test.mov exists, copy it to data/video/test.mov.

You're completing the Yonder Dynamics ML take-home in this repo, FULLY AUTONOMOUSLY. I will not be available to answer questions. Make reasonable decisions, record them, and keep going. Do not stop until every step is done or truly blocked; if blocked, skip that step, record why, and continue with the rest.

First read README.md, METHODOLOGY.md, AI_LOG.md (template), requirements.txt, .gitignore and sampling/. Follow every README rule. Reviewers grade process, design judgment, metric choice, error analysis, commit history, and evidence that AI output was verified.

## Machine
Apple Silicon MacBook, 24GB unified memory, macOS. Use Python 3.12+ (install via `brew install python@3.12` if needed) in .venv.
- Train with device="mps". Set PYTORCH_ENABLE_MPS_FALLBACK=1. Use batch 32 (drop to 16 if memory errors), workers 4, imgsz 512.
- If MPS errors, gives NaN losses, or is slower than CPU in a 1-epoch timing test, fall back to CPU and log it.
- Ultralytics has had MPS-specific validation discrepancies: run FINAL evaluation of the chosen model on CPU, and compare one MPS vs CPU val run; log any difference.
- Run long training in the background with output to logs/train_<name>.log, poll it, and keep working on other scripts (audit, predict, error analysis) while it trains.

## Hard rules
- Never touch sampling/ or Sampled-YD-Object-Detection-2/. Derived data goes in data/ (git-ignored).
- Roboflow key ONLY from env var ROBOFLOW_API_KEY via .env (python-dotenv). Never print, log, echo, or commit it. Catch download exceptions and print a sanitized message. Use truststore if SSL fails; never disable cert checks.
- Pin every added library in requirements.txt with exact versions.
- Smoke test (1 epoch, ~20 images) before any full training run.
- Don't claim anything works unless you ran it. If a metric looks too good (mAP50 > 0.95), suspect leakage/bugs and investigate.

## Commits
Commit after EVERY meaningful unit of work (aim for 20+ commits total), small and descriptive, e.g. "add leakage check to audit", "fix label parsing for multi-object images". Bug fixes go in their own commits. `git status` before every `git add -A`. Never squash, amend, rebase or force-push. Never commit .env, the dataset folder, data/, runs/, logs/. DO commit weights/best.pt (YOLOv8n ~6MB) and results/ (markdown, json, small pngs; keep results under ~20MB total). Don't push; I'll push.

## AI_LOG.md (follow the repo template and the README Part 3 rules exactly)
Maintain AI_LOG.md AS YOU GO, updating it in the same commit as the work it describes. It must be HONEST and attributed correctly: this whole project is being done by Claude Code (you) from a single prompt by Ryan. Complete ALL sections of AI_LOG.md fully on Ryan's behalf—do not leave any TODO placeholders for Ryan.
- Entry 1: state that Ryan gave Claude Code one autonomous prompt to build the pipeline end to end, and paste this full prompt in a collapsible <details> block.
- Then one entry per significant step, with the README's three fields:
  - What was asked: the sub-task and which part of the prompt it came from.
  - What was kept vs rewritten, and why: your own iterations (first approach, why it was replaced, what stayed), with commit hashes.
  - What the AI got wrong / had to be caught: every real bug, bad assumption, failed check, or misleading number you found and how it was detected (which test/check/plot caught it). Be specific and truthful; do not invent errors to look good and do not hide real ones.
  - Verification & Final Review: Provide a thorough, critical assessment and verification analysis of the step's code and outputs as the reviewing engineer.
Keep entries concise.

## Steps
1. Setup: venv, requirements.txt, .env.example, .gitignore additions.
2. scripts/download_data.py: workspace "malletbottle2", project "sampled-yd-object-detection", version 2, format "yolov8". Skip if already present.
3. scripts/audit_data.py → results/audit.md + plots: counts per class/split, box size/aspect, objects per image, brightness/contrast distributions, and results/audit_samples.jpg (GT boxes drawn on 16 random images) to verify label parsing. LEAKAGE CHECK: images are augmented variants of shared source photos (names like `<orig>_jpg.rf.<hash>.jpg`); group by original stem AND perceptual hash (imagehash phash, small Hamming threshold) and report how many val images have a sibling in train.
4. scripts/make_split.py: group-aware, class-stratified ~80/20 split with fixed seed → data/split/ + data.yaml (no test line). Keep the original split available for comparison.
5. src/preprocess.py: ONE shared function used for training-data prep AND inference: CLAHE on L channel in LAB space, justified by the audit's lighting stats. Write to data/split_clahe/. configs/train.yaml with deliberate outdoor-oriented augmentations (HSV, brightness, small rotation, scale, translate, fliplr, mosaic; no vertical flip unless justified), each with a comment explaining why.
6. scripts/train.py: YOLOv8n pretrained, fixed seed, early stopping. Ablation: (a) baseline (grouped split, default aug, no CLAHE) vs (b) final (grouped split + CLAHE + custom aug). ~60 epochs each, patience 15. Copy final best.pt to weights/best.pt.
7. scripts/evaluate.py → results/metrics.md + metrics.json: per-class precision, recall, F1, mAP50, mAP50-95, confusion matrix, PR curves; baseline vs final side by side, plus final on the original (leaky) split to quantify leakage inflation. Pick the confidence threshold maximizing mean F1 on val → results/threshold.json.
8. predict.py (root): `python predict.py <input_dir> <output_dir> [--weights weights/best.pt] [--conf T] [--save-vis]`. Same preprocess. Per image writes `<same_stem>.txt`, lines `class_id x_center y_center width height confidence` normalized 0-1 (bottle=0, mallet=1); empty file if no detections. tests/test_predict_format.py checks format, ranges, filename matching. Run it.
9. scripts/error_analysis.py: match preds to GT at IoU 0.5 on val; save every FP/FN/wrong-class to results/errors/ (GT green, pred red) and results/errors.md with counts by type, class, box size, brightness bucket. Report patterns as data and present actionable engineering takeaways.
10. scripts/video_test.py `<video> --every N`: extract frames, run predict, annotated frames to results/video/ (keep ≤30 in git), summary in results/video.md. Use data/video/test.mov if it exists; if not, build the script, test it on a short clip made from val images, and note that the real video is pending.
11. Stretch, efficiency: scripts/benchmark.py: params, GFLOPs, size, mean/p95 latency on CPU and MPS at 512 and 320. Export ONNX; try INT8 (ONNX Runtime quantization or CoreML/TFLite, whichever works on Mac) and re-evaluate mAP. results/efficiency.md table. Note the real target is RKNN on an OrangePi NPU (can't test here).
12. Stretch, hard examples: scripts/hard_examples.py: rank train images by low GT confidence/misses, contact sheet + results/hard_examples.md grouped by failure type, with a data-collection prioritization strategy.
13. run_all.sh: fresh clone → venv → install → download → audit → split → preprocess → train → evaluate → predict demo. Plus colab.ipynb (getpass for key) for reviewers with a GPU.
14. METHODOLOGY.md: fill in "How to run" completely (commands, Python version, macOS version, env var name, exact inference command, threshold). Fully complete the thought-process and methodology sections, writing clear engineering rationales for all design choices and findings.

## DEV_NOTES.md (git-ignored, for running notes)
Running notes: every decision + rationale, actual numbers, hypotheses for error causes, anything uncertain, and a checklist of specific things verified during execution.

## When finished
Print: the final per-class metrics table (baseline vs final), the leakage finding, the commit count, and a summary of completed execution steps.

## Final step: push
- Run the full test suite and `git status`; make sure no key, .env, dataset, data/, runs/ or logs/ are tracked (`git ls-files | grep -Ei "env|Sampled-YD|^data/|^runs/"` must be empty). Grep the entire history for the key's first 6 characters, read from the env var without printing it: `git log -p --all | grep -c "${ROBOFLOW_API_KEY:0:6}"` must be 0. If the key is ever found, do NOT push; stop and tell me.
- Make sure every file is under 100MB.
- `git push -u origin main` (normal push, never --force).
- Verify the repo is public: `curl -s -o /dev/null -w "%{http_code}" https://github.com/Ryan-Chow/yonder-ml-take-home` must return 200 (no auth).
- Print: the repo URL, final per-class metrics table (baseline vs final), the leakage finding, commit count, and everything I still need to do manually.

[follow-up line from Ryan] thank you, make sure it clones just to my github folder and not into some ~/projects/---- thanks
```
</details>

**What was kept vs. rewritten, and why:** I kept the prompt's step plan as the backbone. The main deviations, each
explained in its own entry:
- The leakage check had to go beyond the `.rf.` stem: the dataset is already de-duplicated by stem, and leakage
  comes through video frames and capture sequences (entry 4).
- The leakage *measurement* uses a model trained on the original split rather than scoring the grouped-trained model
  on the original val (entries 5 and 8).
- The ablation was extended to a 2×2 with cluster-bootstrap CIs (entry 9).
- The model that ships is **`clahe_only`**, chosen by a rule written before the last results came in, not the
  prescribed CLAHE + custom-augmentation recipe, which scored worse (entry 9).
- Evaluation runs at `rect=False`, the input size that actually ships (entry 8).
- The real phone video was not available, so only a stand-in was run (entry 11).
- No Roboflow download was run (no key).

**What the AI got wrong that had to be caught:**
- Both prompt versions name a repo path that did not match the machine. `~/Projects/yonder-ml-take-home` did not
  exist; the repo from the earlier session lives at `~/Documents/GitHub/yonder-ml-take-home`, so I continued there
  instead of re-cloning.
- The current prompt says the dataset was "already unzipped"; it was not. I unzipped the zip Ryan pointed to (see
  entry 3).
- The earlier prompt assumed `ROBOFLOW_API_KEY` was exported; it was not (checked with `[ -n "$ROBOFLOW_API_KEY" ]`,
  without printing anything).

**How it was verified (verification & final review):** This entry has nothing to verify on its own; each step below has its own review.

## 2. Environment setup

**What was asked:** Step 0 + Step 1: clone, create the public GitHub repo, remotes, Python 3.12 venv, pinned
requirements, `.env.example`, `.gitignore` additions.

**What was kept vs. rewritten, and why:** Cloned into `~/Documents/GitHub/yonder-ml-take-home` (Ryan's follow-up
instruction overrides the `~/Projects` path in the prompt). Renamed the original `origin` to `upstream` and added the
GitHub repo as `origin` (equivalent to the prompt's set-url + add). Installed Python 3.12.15 via Homebrew. Pinned
exact versions from `pip freeze` for every library the code imports (ultralytics 8.4.175, torch 2.14.1, ...).
Added `runs/`, `logs/`, `*.onnx`, `DEV_NOTES.md` to `.gitignore` (`data/` and the dataset folder were already there).

**What the AI got wrong that had to be caught:** `pip freeze` showed both `opencv-python` and
`opencv-python-headless` installed (roboflow/ultralytics pull in different variants). Only `opencv-python` is pinned
explicitly; the headless one comes in transitively, so pinning both was unnecessary.

**How it was verified (verification & final review):** `torch.backends.mps.is_available()` returned `True`; imports of torch, ultralytics,
cv2 succeed in the venv. The env var was confirmed missing in the agent shell; the code still only reads it from the
environment / `.env`.

## 3. Dataset in place + download script

**What was asked:** Step 0 (verify dataset structure and counts) and Step 2 (`scripts/download_data.py` for
reviewers; don't download, just verify the skip path).

**What was kept vs. rewritten, and why:** I kept the earlier session's `download_data.py` (`9924a18`) unchanged.
It reads the key only from the environment or `.env`, skips the download when `train/images` and `valid/images`
are non-empty, redacts the key and any `api_key=` query parameter from exception text, and uses `truststore` when
it is installed. It calls `truststore` before every download, not only after an SSL failure; that is harmless and
simpler than catching the error and retrying. I added `tests/test_download.py` so these checks stay verified
without network access or a key.

**What the AI got wrong that had to be caught:**
- My first `unzip` stopped at an interactive "replace data.yaml?" prompt, because the zip contains `data.yaml`
  **twice**. The non-interactive shell answered "None", nothing was extracted, and the count check printed 0/0. That
  is how it was caught. I compared the two entries' MD5s (identical, `10e8fff4…`) and re-ran with `unzip -o`.

**How it was verified (verification & final review):**
- Counts: 800 train images and 800 labels, 199 valid images and 199 labels. `diff` of image stems against label stems
  is empty for both splits, so every image has exactly one label file.
- `data.yaml` lists `names: ['bottle', 'mallet']` and a `test:` path that does not exist. My own split yaml will
  leave the test line out.
- `python scripts/download_data.py` with the data present prints the skip message and exits 0.
- 4 pytest cases pass: the key is redacted in two forms, the skip path works, and a missing key exits 1 with a clean
  message.
- Limitation: the real download path cannot be exercised here, because no API key is available.

## 4. Data audit and leakage check

**What was asked:** Step 3: `scripts/audit_data.py`, which writes per-class and per-split counts, box geometry,
lighting stats, a GT sample sheet, and a leakage check that groups images by original stem and by phash.

**What was kept vs. rewritten, and why:**
- I kept the stem and phash checks the prompt asked for, but they were not enough on their own. The prompt assumed
  val images would share an `<orig>` stem with train images. Reading `sampling/sample-roboflow.py` showed it keeps
  **one** `.rf.` copy per source photo, so the stem check is structurally zero.
- Iterations before committing:
  1. The first version was a whole-image phash. The pair contact sheet showed that at distance 8, unrelated
     *rotated* photos matched, because the black rotation corners dominate the hash.
  2. I switched to a centre-crop phash (central 40%).
  3. That surfaced four "distance 0" pairs, all white-background synthetic collages whose centre crops are blank.
     I now exclude near-uniform crops (std < 8).
  4. The phash still cannot see consecutive shots of the same scene taken at different rotations (centre-crop
     distance ≈32), so I added a **sequence** rule for consecutive capture numbers and timestamps within 30 s.
- The grouping helpers live in `src/data_utils.py` (commit `60856d8`), so the split script uses exactly the
  same definition of "same origin".

**What the AI got wrong that had to be caught:**
- The prompt's leakage model (shared `.rf.` stems) was wrong for this dataset. I caught it by reading the sampling
  script and confirming 0 stem overlap.
- My own whole-image phash produced false matches. The pair contact sheet showed rotated, different scenes at
  d=8.
- My blank-crop collisions came from the centre-crop fix itself. I caught them by inspecting the d=0 pairs.
- While reviewing `leakage_pairs.jpg`, I first called one sequence pair "unrelated". Closer review showed it was the
  same bottle with its background replaced. I retracted that call after a 30-pair precision audit.
- A leftover expression in the box-centre scatter (`[...][:0] or [...]`) was nonsense. I caught it re-reading the
  code before the first run and simplified it.

**How it was verified (verification & final review):**
- `audit_samples.jpg`: GT boxes sit on the objects in all 16 images, with correct class colours, so YOLO parsing
  (xc, yc, w, h, normalised) is right.
- Every one of the 1222 label lines passes strict parsing (5 fields, class ∈ {0,1}, coords ∈ [0,1]). All images
  are 512×512.
- Leakage: **107/199 val images (54%) share a same-origin group with train.** By rule: sequence 93, family 8,
  phash 0, exact stem 0.
- The sequence rule's precision was checked by eye on 30 random cross-split pairs: 28 same scene and 2 same object
  and session. So 107 is a slight overestimate at worst, not an artefact.
- Limitations:
  - Rule-based grouping can still miss same-scene photos with unrelated names. The phash should catch exact
    near-duplicates, but not different viewpoints.
  - The 30 s timestamp window and the gap ≤ 2 were chosen by eye, not tuned.
- Lighting: 15% of train images are dark (L<80), 7% are bright (L>170), 13% are low-contrast (std L < 40), and
  about 39% carry Roboflow rotation corners. This spread is the justification for CLAHE in step 5.

## 5. Group-aware stratified split

**What was asked:** Step 4: a group-aware, class-stratified ~80/20 split with a fixed seed, written to
`data/split/` with a `data.yaml` that has no `test:` line, while keeping the original split for comparison.

**What was kept vs. rewritten, and why:**
- The split pools all 999 images, reuses `build_groups` from step 4, and searches 5000 seeded random group orders.
  It keeps the order whose val set is closest to 20% of each class's images, 20% of each class's boxes, and 20% of
  all images.
- Exact stratified group k-fold is overkill for two classes. The greedy fill plus random search reached a balance
  cost of 0.005.
- `data/orig/data.yaml` points at the untouched Roboflow split, so the leakage comparison needs no copies.
- I also changed the leakage measurement the prompt specified. "Final model on the original split" would score a
  grouped-trained model on images that are mostly in its *own* training set, which measures memorisation. Instead,
  run (c) trains the final config on the original split, and its val score is broken down into leaked and clean
  images. `results/split_manifest.csv` carries the per-image `orig_val_has_train_sibling` flag for that.

**What the AI got wrong that had to be caught:** Nothing failed in this step. The `assert` that no group straddles the
split passed on the first run, and an independent re-check from the CSV agreed.

**How it was verified (verification & final review):**
- New split: train 800 / val 199. Val holds 108 of the 539 images containing a bottle (20.0%), 108 of the 536
  containing a mallet (20.1%), 138 of 685 bottle boxes, and 108 of 537 mallet boxes.
- 0 groups straddle the split (checked by the in-script assert and again from the CSV).
- The manifest flags 107 original-val images as leaked, matching the audit.
- The manifest MD5 is identical across three runs, including one with the phash cache deleted.
- 304 images changed split relative to Roboflow's.
- Caveat: the groups are only as good as the rules from step 4. Same-place photos with unrelated filenames can
  still sit on both sides, so grouped-val scores may still be slightly optimistic.

## 6. CLAHE preprocessing and augmentation config

**What was asked:** Step 5: a single `src/preprocess.py` function used by both training-data prep and inference
(CLAHE on LAB L, justified by the audit), output to `data/split_clahe/`, and a commented `configs/train.yaml` of
outdoor-oriented augmentations.

**What was kept vs. rewritten, and why:**
- Kept as specified. `preprocess()` is the only image transform. Training data goes through it via
  `python -m src.preprocess`, and `predict.py` will import the same function.
- One deliberate choice: the outputs are written as **PNG**, not JPEG. Re-encoding as JPEG would make the training
  pixels differ slightly from what inference computes in memory.
- I also built `data/orig_clahe/` for run (c), the leakage comparison.
- Evidence (commit `5267055`) lives in `results/clahe_stats.md` and `clahe_examples.jpg`.
- Augmentation choices, each commented in the yaml: `hsv_v` raised to 0.5, `degrees` 10, `perspective` 0.0005,
  `flipud` 0 (a rover camera is always upright). The baseline run keeps every Ultralytics default.

**What the AI got wrong that had to be caught:**
- My first justification for CLAHE implied it would "normalise lighting". The before/after stats show it does not
  fully do that:
  - Dark images drop from 15% to 7% and low-contrast images from 13% to 5%.
  - The spread of per-image mean brightness only shrinks from 40 to 32.
  - The darkest images (mean L ≈ 0–5) stay unreadable.

  I reworded the claim: CLAHE fixes *local* contrast, and global exposure robustness is handed to the `hsv_v`
  augmentation.
- Those near-black images have mallet labels that are invisible even to a human. I flagged them as likely
  unavoidable misses (DEV_NOTES).

**How it was verified (verification & final review):**
- Colour preservation was measured, not assumed: the mean |Δa/b| is 0.11/255.
- Counts after preprocessing: 800/199 images and labels in both `split_clahe` and `orig_clahe`. Label files are
  byte copies, and the yaml has no `test:` key.
- Every "default" value cited in `configs/train.yaml` was checked against the installed
  `ultralytics/cfg/default.yaml`.
- `albumentations` is not installed, so Ultralytics' optional albumentations step (blur and its own CLAHE, at
  p=0.01) is off in both arms. The ablation compares exactly what the yaml says.
- One visible artefact: a faint tile seam on one near-black image. It is cosmetic, and the clip limit keeps it
  small.

## 7. Training script, smoke test, device choice

**What was asked:** Step 6 and the "Machine" section:
- YOLOv8n pretrained, fixed seed, early stopping, and an ablation of baseline vs. final, ~60 epochs, patience 15.
- Train on MPS with batch 32 and workers 4.
- Run a smoke test before any full run, and a one-epoch MPS-vs-CPU timing test.

**What was kept vs. rewritten, and why:**
- `scripts/train.py` (`67125d5`) builds every run from `configs/train.yaml`, so the arms differ only in data and
  augmentation.
- I added a third run, **leak** (the final recipe on the original split), for the leakage measurement (entry 5).
- The COCO checkpoint is moved to `data/pretrained/` so a stray `yolov8n.pt` never lands in the repo root.
- Baseline and final were trained **in parallel** on MPS. Data loading runs on the main thread (see below), so two
  processes overlap CPU and GPU work. The leak run was queued to start when baseline finished.

**What the AI got wrong that had to be caught:**
- **Bug (mine), fixed in `6874b40`:** `train.py` copied `best.pt` to `weights/best.pt` for *any* non-smoke `final`
  run, including the one-epoch timing runs. I caught it when `weights/best.pt` appeared after the timing test.
  Because the CPU timing run was already executing the old code, it wrote the file a **second** time after my first
  delete; `git status` showed `weights/` untracked again. Now only a full-length run named `final` publishes
  weights.
- **Misleading number:** both one-epoch timing runs ended with val mAP ≈ 0 (val `cls_loss` 56 942 on MPS, 14 199 on
  CPU). This looked like the MPS bug the prompt warned about. Checks that ruled it out:
  1. CPU did it too.
  2. Validating the same checkpoint on CPU and MPS gave identical results.
  3. The weights had no NaN or Inf and sane BatchNorm stats.
  4. The cause: with `epochs=1`, Ultralytics compresses the LR warmup into one epoch (lr 0.00167 vs. 0.00053).

  The full 60-epoch runs dipped the same way in epochs 2–5, while LR ramped up, and then recovered (mAP50 0.64 by
  epoch 10).
- **Prompt assumption that didn't hold:** `workers 4` is silently overridden. Ultralytics 8.4.175 sets
  `workers=0` on CPU and MPS (`engine/trainer.py:178`). I logged this instead of patching the library.

**How it was verified (verification & final review):**
- Smoke test: 1 epoch on 20 images, finite losses, and the val set reports 199 images / 246 instances, matching the
  split (138 + 108).
- MPS took 38 s per epoch vs. 66 s on CPU (wall time, including val), so MPS was kept.
- The `args.yaml` of every run was checked to confirm the custom augmentations (`hsv_v` 0.5, `degrees` 10,
  `perspective` 0.0005) are present only in final and leak.
- The optimizer chosen by `auto` was AdamW (lr 0.001667), the same for every arm.
- Caveat: `deterministic=True` is not fully honoured on MPS (PyTorch warns that `scatter_reduce` and
  `index_put_with_accumulate` have no deterministic MPS kernel). Re-training will give close but not bit-identical
  weights.

## 8. Inference script, evaluation, leakage measurement

**What was asked:** Step 7 (`evaluate.py`: per-class P/R/F1/mAP, confusion matrix, PR curves, baseline vs. final,
the leakage comparison, and a threshold chosen by max mean F1) and Step 8 (`predict.py` plus format tests, then run
it). Also the "Machine" rule: run the final evaluation on CPU and compare MPS vs. CPU.

**What was kept vs. rewritten, and why:**
- `src/infer.py` is the single inference path. `predict.py`, evaluation, error analysis, the video test, and hard
  examples all call it (`7c8b735`, `a35880d`).
- An independent matcher (`src/matching.py`) re-counts TP/FP/FN through that path, so the reported numbers come
  from the same code that ships.
- **Rewrite 1** (`fb3a529`): `preprocess()` now stretches non-512 inputs to 512×512 before CLAHE. Roboflow
  exported the training set with "Stretch", while Ultralytics letterboxes non-square input. A 16:9 video frame would
  otherwise reach the model with a different geometry than anything it trained on. Normalised outputs are invariant
  to the stretch, so the output format is unchanged.
- **Rewrite 2** (`bf10f9e`): every reported validation uses `rect=False` (details below).
- **Rewrite 3** (`6ab47ef`): evaluation counts at the *rounded* threshold that ships (0.441).

**What the AI got wrong that had to be caught:**
- **Evaluated at an input size that never ships.** Ultralytics' default `.pt` validation (`rect=True`) pads
  512×512 images to **544** (`pad=0.5` stride rounding in `data/build.py`).
  - How it was caught: ONNX scored 0.677 mAP50 vs. 0.692 for PyTorch on the same weights.
  - I first verified that the raw outputs agree (max |Δ| = 0.0012) and that `predict()` gives identical boxes
    (3281 detections each).
  - Then I reproduced 0.677 with PyTorch at `rect=False`.

  All reported numbers now use the deployed 512 input. The per-epoch val used to pick `best.pt` during training
  still uses 544; that is a minor effect I left alone.
- **Off by one detection between `evaluate.py` and `predict.py`** (232 vs. 233). Evaluation used the unrounded
  threshold 0.44144, while `threshold.json` and `predict.py` use 0.441; one detection sat at 0.4412. The fix is
  `6ab47ef`. `tests/test_consistency.py` failed before the fix and passes after it.
- **Picking up in-progress checkpoints.** Ultralytics rewrites `best.pt` every epoch. My bootstrap script silently
  included `clahe_only` at epoch 3 (mAP50 0.41). The fix (`4aa654e`) uses only checkpoints whose optimizer has been
  stripped, which happens at the end of training.
- **My planned leakage measurement would have measured memorisation.** "Final model on the original val" scores the
  grouped-trained model on images ~80% of which it trained on. I replaced it with run (c): the final recipe trained
  on the original split, scored on its leaked vs. clean val subsets.

**How it was verified (verification & final review):**
- On CPU at 512, final vs. baseline on grouped val: mAP50 0.883 vs. 0.876, mAP50-95 0.490 vs. 0.541, mean F1 0.868
  vs. 0.851.
- **MPS and CPU validation are identical** to 4 decimals for this model, so the MPS discrepancy the prompt warned
  about does not occur here.
- The own-matching P/R agrees with Ultralytics' validator (e.g. bottle 0.819/0.754 in both), and
  `tests/test_consistency.py` proves `predict.py`'s `.txt` files reproduce the reported confusion matrix exactly.
- The final model's two "wrong class" errors (`t_bot_mal578`, `581`) are **dataset label errors**: orange mallets
  labelled bottle, checked by eye next to their correctly labelled neighbours 576 and 579. True class confusion is 0.
- Leakage: the same recipe scores mAP50 0.914 / mAP50-95 0.558 on Roboflow's val vs. 0.883 / 0.490 on the grouped
  val. Inside the leak model, leaked val images score 0.930 vs. 0.894 for clean ones. That leaked/clean gap is
  confounded by content, as the report states.
- Limitation: the threshold is chosen on the same val set it is reported on, because there is no test split.

## 9. 2×2 ablation, uncertainty, and which model ships

**What was asked:** Step 6's ablation of (a) baseline vs. (b) final, plus the general instruction to show design
judgement and to distrust numbers. The 2×2 extension and the bootstrap were my additions.

**What was kept vs. rewritten, and why:**
- The prescribed two-arm ablation showed final beating baseline by +0.007 mAP50 but losing 0.051 mAP50-95. Two arms
  cannot say which change (CLAHE or the augmentation) caused what. So I trained the two missing cells, `clahe_only`
  and `aug_only` (`f31f0f4`).
- I computed **paired cluster-bootstrap** CIs (`src/apstats.py`, `scripts/bootstrap_ablation.py`).
- My first bootstrap resampled images. Seeing 12 of 13 frames of one video fail together showed that errors are
  correlated within a source, so I switched to resampling the 102 same-origin groups (`5578e77`). That widened the
  CIs about 1.7×.
- **Shipping decision.** Before `clahe_only`/`aug_only` finished, I wrote a rule in DEV_NOTES: mAP50 is primary,
  and among arms not significantly below the best mAP50, the best mAP50-95 wins.
  - Applied mechanically by `scripts/ship.py` (`b12b9ee`), it chooses **`clahe_only`**.
  - `train.py` no longer publishes weights.
  - Whether a checkpoint needs CLAHE is now read from the data path stored inside it, so `predict.py` can never feed
    a model the wrong preprocessing.

**What the AI got wrong that had to be caught:**
- **My CLAHE justification did not survive the ablation.** The CLAHE main effect is −0.011 mAP50 and −0.017
  mAP50-95, both CIs spanning 0. The audit showed lighting varies a lot, but that did not translate into a measurable
  gain on this val set.
- **The prescribed final recipe is not the best arm.** CLAHE and custom augmentation interact negatively: final
  scores 0.882/0.490 vs. 0.902/0.521 for aug_only (mAP50/mAP50-95).
- **Image-level bootstrap understated the uncertainty** (see above). One "significant" result vanished: with
  B=2000 cluster resamples, final − baseline mAP50-95 is −0.051 [−0.085, +0.004].
- **AP re-implementation.** My first AP tests encoded the *older* Ultralytics sentinel convention and failed against
  my code. Reading `compute_ap` in the installed 8.4.175 showed precision drops to 0 right after the last recall. I
  replaced the hand-computed expectations with a test that compares directly against Ultralytics' function on 50
  random curves.
- **A hard-coded conclusion.** `evaluate.py` first printed a hard-coded sentence about which differences are
  significant. A re-run could silently contradict it, so it is now generated from `ablation.json`.

**How it was verified (verification & final review):**
- My AP equals Ultralytics' validator to within 0.002 on every arm, e.g. baseline 0.8746 vs. 0.8764 mAP50.
- Each checkpoint's embedded `train_args` was checked for its data path, `hsv_v`, and `degrees`, confirming each arm
  trained as intended.
- `weights/best.pt` is byte-identical to `runs/clahe_only/weights/best.pt` (`cmp`, plus the SHA-256 in
  `results/shipped.json`).
- Honest assessment: clahe_only, aug_only, and final are **statistically indistinguishable** on both metrics. The
  rule is a principled tie-break, not evidence that clahe_only is better.
  - clahe_only has the lowest point-estimate F1 of the four (0.844 vs. 0.868 for aug_only and final).
  - If recall at the threshold mattered more than box tightness, aug_only would be the pick.
- I kept the pre-registered choice rather than switch after seeing the numbers, because switching would make the
  "rule" meaningless.
- Each arm is a single seed, so seed variance is not in these CIs.

## 10. Error analysis

**What was asked:** Step 9 and README Part 1.5:
- Match predictions to GT at IoU 0.5 on val.
- Save every FP, FN, and wrong-class case as an image.
- Write counts by type, class, size, and brightness.
- Report patterns as data, and put causal hypotheses only in DEV_NOTES, labelled as hypotheses.

**What was kept vs. rewritten, and why:**
- `scripts/error_analysis.py` (`cd39651`) also reports **miss rates per bucket**, not just counts. Raw counts mostly
  reflect how many objects a bucket contains.
- It also gives **subtypes**: FN "no detection near it" vs. "poorly localised", and FP "background" vs.
  "localisation" vs. "duplicate".
- It adds a **filename-family** breakdown, because the grouped split puts whole sources in val.
- It was re-run for the shipped model after the shipping decision. 78 error images are in `results/errors/`.

**What the AI got wrong that had to be caught:**
- **A "wrong-class" error turned out to be the label, not the model.** While checking the video stand-in overlay I
  saw an orange mallet labelled "GT: bottle".
  - I traced it to `t_bot_mal581`, then rendered the whole `t_bot_mal57x–58x` session. 578 and 581 are mallets
    labelled bottle; 576 and 579 are labelled correctly.
  - Two of the shipped model's three wrong-class errors are therefore correct predictions.
  - The third (an extreme close-up of a mallet handle predicted as bottle) is a genuine error.
- I kept the causal stories out of `results/errors.md`, as the prompt asked. They are in DEV_NOTES as H1–H6.

**How it was verified (verification & final review):**
- Every error image was generated from the same detections `predict.py` produces; `tests/test_consistency.py` ties
  the two together.
- I reviewed the contact sheet by eye. The patterns in the tables are real:
  - **Bottles miss 25% vs. 7% for mallets.**
  - Dark and bright images miss 30–35% vs. 12% at normal exposure.
  - One whole val video (`green_nalgene_vid`, a hand-held bottle indoors) misses 12 of 13. It sits entirely in val
    because of the grouped split, which is exactly the generalisation gap the original split hid.
  - False positives cluster on a blue barrel, product-page colour swatches, and collages.
- Box size barely matters on val (miss rate 17–19% across buckets), unlike the train hard examples. One reading:
  val misses are dominated by unseen contexts rather than by size.
- Limitations:
  - Buckets are small (e.g. 20 GT boxes in "bright"), so the rates carry wide intervals.
  - The thresholds for "dark" and "bright" were set from the audit, not tuned.

## 11. Video test (stand-in clip; real video pending)

**What was asked:** Step 10 and README Part 1.6: `scripts/video_test.py <video> --every N` writes annotated frames
(≤30 in git) and a summary. If `data/video/test.mov` is absent, it tests on a clip built from val images and notes
that the real video is pending.

**What was kept vs. rewritten, and why:**
- There was no `~/Downloads/test.mov`, so the script builds a stand-in clip (`2a7bb6b`) from 30 grouped-**val**
  images, with zoom, motion blur, and exposure drift, at 1280×720 to exercise the non-square path.
- Because the frames come from labelled images, it can also measure frame-level recall.
- Later improvements (`a5cffbb`):
  - Flicker is now measured only *within* a shot. Cuts between source images legitimately change what is visible.
  - The confidence curve now uses detections ≥ 0.05, so near-misses under the threshold are visible instead of
    flat zeros.
  - The source image behind each "detected but absent" frame is listed.

**What the AI got wrong that had to be caught:**
- The first report's flicker (15–18%) mostly counted scene cuts, which is a misleading number. Within shots it is
  0–7%.
- The first confidence plot only showed detections that were already above threshold, so it could not show how
  close the misses were.
- Building this test exposed the aspect-ratio mismatch fixed in `fb3a529` (stretch vs. letterbox). The boxes on a
  1280×720 frame were checked by eye to sit on the objects.
- **An unverified claim, corrected later.** I first wrote that both rooftop "bottle detected but absent" frames were
  the blue barrel. Fact-checking METHODOLOGY.md, I rendered both source images and the actual clip frames. Only one
  is the barrel; the other is a stack of red bricks.

**How it was verified (verification & final review):**
- Shipped model on the stand-in: bottle found in 40 of 48 frames where present (83%), mallet in 45 of 48 (94%).
- The "detected but absent" frames trace back to:
  - `t_bot_mal581`, the mislabelled mallet: the model is right.
  - Two rooftop frames from the `20231021` session. One is a "bottle" box over the blue barrel (frame 105, conf
    0.46, only after the clip's zoom and brightening). The other is a "bottle" box on a stack of red bricks (conf
    0.50).
- **This is not the README's real-world test.** The frames are val images with synthetic blur and exposure, not a
  phone video of a mallet stand-in outdoors. To complete it, record the video and run
  `python scripts/video_test.py data/video/test.mov --every 10`. Nothing else needs to change.

## 12. Hard-example mining (stretch)

**What was asked:** Step 12: rank train images by low GT confidence and misses, make a contact sheet, and write
`results/hard_examples.md` grouped by failure type with a data-collection prioritisation strategy.

**What was kept vs. rewritten, and why:**
- `scripts/hard_examples.py` (`d52bbeb`) gives every GT box a "GT confidence" (best same-class prediction at
  IoU ≥ 0.5). It sorts each failure into missed, low confidence, mislocalised, class confusion, or false alarm, and
  adds a **label-suspect** flag for confident disagreements with the labels.
- The first version had no prioritisation strategy, which the prompt requires. I caught the gap when reviewing the
  output against the step list. The strategy section is now *generated* from the measured rates and the val error
  analysis rather than written as fixed text.

**What the AI got wrong that had to be caught:**
- The ranking formula weights false alarms (+0.5 each). In the interim run, an image with two confident "false
  alarms" outranked fully missed objects.
- Looking at it on the contact sheet, those "false alarms" were **real, unlabelled bottles** (`IMG_9183`). That is
  why label suspects get their own section and are priority 1 (fix labels before collecting more data).

**How it was verified (verification & final review):**
- Shipped model: 36 of 800 train images still have a failure despite having been trained on.
- Images with a small object (<32 px) are hard 21% of the time vs. 2% otherwise. Images with both classes are hard
  24% vs. 3%.
- I checked the label suspects by eye on the contact sheet.
- Limitation: training images are optimistic (the model saw them). Running the same ranking on new, unlabelled rover
  footage would be the real active-learning step; the strategy describes that loop.

## 13. Efficiency: size, latency, ONNX, INT8 (stretch)

**What was asked:** Step 11:
- Report params, GFLOPs, size, and mean/p95 latency on CPU and MPS at 512 and 320.
- Export to ONNX, try INT8, and re-evaluate mAP.
- Discuss RKNN on the OrangePi NPU and the tradeoffs.

**What was kept vs. rewritten, and why:**
- `scripts/benchmark.py` (`bf10f9e`) tries INT8 two ways: every node, and with the Detect head's decode kept in
  FP32. I expected naive full quantisation to break YOLOv8's mixed-range output tensor, so I built both variants
  from the start.
- It was later split into a measure step and a `--report-only` step, so the write-up can be regenerated without
  re-timing.
- The tradeoff discussion is generated from the measured numbers.

**What the AI got wrong that had to be caught:**
- **ONNX vs. PyTorch mAP mismatch** (0.677 vs. 0.692 on interim weights). This was the `rect=True` padding to 544
  described in entry 8, not an export bug: raw outputs matched to 1e-3 and per-image predictions were identical.
- **Latencies measured during training are worthless.** The dry run showed 107 ms on CPU while training competed
  for the cores; on an idle machine it is 28.7 ms. The final numbers were taken with nothing else running.
- **Unsupported claims in my first draft of the discussion**, removed or qualified after re-reading it against the
  data:
  - "~1 ms CLAHE" had never been measured. It is now measured: 0.61 ms at 512, 4.1 ms for a 1080p frame.
  - "the 320 loss is mostly small objects" was not broken down by size, so I removed it.
  - "training at 320 would recover the gap" is now marked untested.
  - The claim about Rockchip's export cutting the DFL decode out of the graph is from background knowledge and is
    marked "not verified here".

**How it was verified (verification & final review):**
- 3.14 M params; 5.6 GFLOPs at 512, consistent with YOLOv8n's 8.7 GFLOPs at 640 scaled by (512/640)².
- ONNX FP32 mAP equals PyTorch exactly.
- Naive INT8 gives **mAP 0.000**. Keeping the head decode in FP32 gives 0.869 vs. 0.872 mAP50, at 3.6 MB vs.
  12.7 MB.
- 320 input: 2.6× fewer FLOPs and 2.6× lower ONNX CPU latency, for −0.041 mAP50 (bottle −0.054, mallet −0.028).
- Limitation: nothing here ran on the real NPU. Apple-CPU INT8 speed-ups (12.8 → 11.7 ms) say little about RKNN.

## 14. run_all.sh, Colab notebook, fresh-clone test

**What was asked:** Step 13: `run_all.sh` covering fresh clone → venv → install → download → audit → split →
preprocess → train → evaluate → predict demo, plus `colab.ipynb` (with `getpass` for the key) for reviewers who have
a GPU.

**What was kept vs. rewritten, and why:**
- `run_all.sh` (`0eb1600`) uses `set -euo pipefail`, finds a Python ≥ 3.12, and picks CUDA, then MPS, then CPU. It
  supports `SKIP_TRAIN=1` (evaluate the committed weights) and `SKIP_LEAK=1`.
- After the shipping change it was updated (`ac62acb`) to train the 2×2 plus the leak run, then bootstrap, then
  `ship.py`, then evaluate.
- `evaluate.py` gained a fallback for a fresh clone with no `runs/`: it evaluates the committed weights as arm
  `shipped`.

**What the AI got wrong that had to be caught:**
- **My first `run_all.sh` had two bugs:**
  - `${SKIP_LEAK:+…}` printed "leak skipped" even when `SKIP_LEAK=0`.
  - `ls … | head` could kill the script under `pipefail` via SIGPIPE.

  I caught both by reading the script before running it.
- **The fresh-clone test found two real problems:**
  - Before the fallback, `SKIP_TRAIN=1` would have exited, because `evaluate.py` required the training runs to
    identify the shipped model.
  - `threshold.json` named a non-existent `runs/shipped/...` path (fixed in `9a76167`).

**How it was verified (verification & final review):**
- **Fresh-clone test:** I ran `git clone` of the repo into a scratch folder, copied in the dataset (no API key is
  available, so the download step was exercised only on its skip path), and symlinked the venv. Then I ran
  `SKIP_TRAIN=1 ./run_all.sh`.
  - All 9 steps ran in 80 s, and 28 tests passed in the clone.
  - `pip install -r requirements.txt` installed nothing, so the pins match the tested environment.
  - The regenerated split manifest, audit, CLAHE stats, error analysis, hard examples, and video report are
    **byte-identical** to the committed ones. The threshold (0.426) and all metrics reproduce exactly.
- **Not tested:**
  - The full-training path of `run_all.sh` end to end. Each step was run individually, but not as one 3.5-hour
    script.
  - `colab.ipynb` was never executed. I have no Colab or CUDA access here; it was only checked to be valid JSON with
    the same commands.
  - A real Roboflow download.

## 15. METHODOLOGY.md and a fact-check of the write-up

**What was asked:** Step 14: complete "How to run" (commands, Python and macOS versions, env var name, exact
inference command, threshold), and write the thought-process and methodology sections with clear rationale for each
choice and finding.

**What was kept vs. rewritten, and why:**
- "How to run" was drafted while training ran. The results sections were written only after the shipping decision,
  so they describe the model that actually ships.
- Every number was copied from a generated results file.
- I then re-read the document against the data, claim by claim, before committing.

**What the AI got wrong that had to be caught** (all by the fact-check pass):
- "Every run hit its best epoch between 55 and 58": wrong. The leak run's best epoch was 60, its last. Now "55–60;
  the leak run was still improving".
- "8% of train images are bright": that is the all-data figure; train is 7%. "39% carry rotation corners" is the
  train figure; overall it is 38%.
- "Every 'detected but absent' video frame is the mislabel or the blue barrel": only partly true. Rendering the
  actual frames showed one is the barrel and one is a stack of red bricks. The same error was in AI_LOG entry 11 and
  is corrected there.
- **A bug in the error analysis**, found while re-checking the blue-barrel claim. A "bottle" box over the barrel
  overlapped the *mallet* GT lying on it, so it was counted as "poor localisation". Now it is "overlaps a
  different-class object" (`c1c01dc`).

**How it was verified (verification & final review):**
- Every number in METHODOLOGY.md was cross-checked against the generated files: `metrics.md`, `ablation.md`,
  `audit.md`, `errors.md`, `video.md`, `efficiency.md`, `hard_examples.md`.
- The document says plainly what is weak:
  - CLAHE showed no measurable benefit.
  - The task's prescribed recipe was not the best arm.
  - The shipped model was picked by a tie-break among statistically indistinguishable arms.
  - The real video test is still pending.
  - Single seeds; the threshold is tuned on val.

## 16. Final checks and push

**What was asked:** The "Final step" section:
- Run the tests and `git status`.
- Make sure no `.env`, dataset, `data/`, `runs/`, or `logs/` is tracked, and every file is under 100 MB.
- `git push -u origin main` without force.
- Confirm the repo is public.

**What was kept vs. rewritten, and why:** Nothing was rewritten. I added a history-wide scan for key-like strings
on top of the checks the prompt listed.

**What the AI got wrong that had to be caught:**
- The first `git push` failed with `HTTP 400 / RPC failed`, git's HTTP buffer being too small for a 49-commit push
  that includes two versions of the 6.5 MB weights.
- I fixed it with a repo-local `git config http.postBuffer 524288000` and a normal (non-force) push.

**How it was verified (verification & final review):**
- 28/28 tests pass, and the working tree is clean.
- The forbidden-paths grep is empty, and DEV_NOTES.md is not tracked.
- The largest tracked file is `weights/best.pt` at 6.5 MB; `results/` is 5.4 MB.
- The only match for key-like strings in the whole history is the fake `OTHERKEY` in `tests/test_download.py`.
- After the push, local `HEAD` equals `origin/main`. Without auth, `curl` returns 200 for both the repo page and the
  raw weights file.
