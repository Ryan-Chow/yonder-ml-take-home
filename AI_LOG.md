# AI Usage Log

**Attribution, up front:** this entire submission (code, experiments, write-ups, commits) was produced by
**Claude Code** (Anthropic's CLI agent, model Claude Opus 5.5) running autonomously from a single prompt written by
Ryan Chow. Ryan did not hand-edit code during the run. Every entry below is written by the agent, describing
what it did, what it changed its mind about, and what checks caught its own mistakes. "I" below = Claude Code.

## 1. One autonomous prompt to build the whole pipeline

**What was asked:** Ryan gave Claude Code one prompt asking it to set up the repo, build the full
pipeline (download → audit with leakage check → grouped split → CLAHE preprocessing → YOLOv8n training with an
ablation → evaluation → inference script + tests → error analysis → video test → efficiency and hard-example
stretch goals → run_all.sh / Colab → METHODOLOGY.md), commit as it went, keep this log honest, then push.
He added one instruction at the end: clone into his `~/Documents/GitHub` folder instead of `~/Projects`.

<details>
<summary>Full prompt (verbatim)</summary>

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

**What was kept vs. rewritten, and why:** Kept the prompt's plan as the backbone. Deviations are recorded in the
entry for the step where they happened.

**What the AI got wrong that had to be caught:** The prompt assumed `ROBOFLOW_API_KEY` was already exported in the
agent's shell; it was not (checked with `[ -n "$ROBOFLOW_API_KEY" ]` without printing the value). See entry 2.

**Verification & final review:** n/a for the prompt itself; each step below has its own review.

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

**Verification & final review:** `torch.backends.mps.is_available()` returned `True`; imports of torch, ultralytics,
cv2 succeed in the venv. The env var was confirmed missing in the agent shell; the code still only reads it from the
environment / `.env`.
