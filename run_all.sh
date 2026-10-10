#!/usr/bin/env bash
# Fresh clone -> trained model -> metrics -> predictions, in one command.
#
#   export ROBOFLOW_API_KEY=...          # or put it in a git-ignored .env (see .env.example)
#   ./run_all.sh                         # full pipeline: ~1.5-2 h on an Apple-Silicon laptop, ~15 min on a GPU
#   SKIP_TRAIN=1 ./run_all.sh            # no training: evaluate the committed weights/best.pt only
#   SKIP_LEAK=1 ./run_all.sh             # skip run (c), the leakage-comparison model
#
# Device is picked automatically (CUDA -> MPS -> CPU); override with DEVICE=cpu|mps|0.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "need Python >= 3.12 (set PYTHON=...)"; exit 1; }

step() { printf '\n==== %s ====\n' "$*"; }

step "1/9 virtualenv + requirements ($PY)"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1

DEVICE=${DEVICE:-$(python -c 'import torch; print("0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))')}
echo "device: $DEVICE"

step "2/9 download dataset (skipped if Sampled-YD-Object-Detection-2/ exists)"
python scripts/download_data.py

step "3/9 audit + leakage check -> results/audit.md"
python scripts/audit_data.py

step "4/9 group-aware split -> data/split, data/orig"
python scripts/make_split.py

step "5/9 CLAHE preprocessing -> data/split_clahe, data/orig_clahe"
python -m src.preprocess --src data/split --dst data/split_clahe
python -m src.preprocess --src data/orig --dst data/orig_clahe
python scripts/show_preprocess.py >/dev/null

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  step "6/9 smoke test, then training (baseline, final, leak unless SKIP_LEAK=1)"
  python scripts/train.py --run final --smoke --device "$DEVICE"
  python scripts/train.py --run baseline --device "$DEVICE"
  python scripts/train.py --run final --device "$DEVICE"      # copies runs/final/weights/best.pt -> weights/best.pt
  [ "${SKIP_LEAK:-0}" = "1" ] || python scripts/train.py --run leak --device "$DEVICE"
else
  step "6/9 SKIP_TRAIN=1: using committed weights/best.pt"
fi

step "7/9 evaluation (CPU) -> results/metrics.md, results/threshold.json"
if [ "$DEVICE" = "mps" ]; then python scripts/evaluate.py; else python scripts/evaluate.py --skip-mps; fi

step "8/9 error analysis, hard examples, video test, inference demo"
python scripts/error_analysis.py >/dev/null && echo "results/errors.md written"
python scripts/hard_examples.py >/dev/null && echo "results/hard_examples.md written"
python scripts/video_test.py --every 3 >/dev/null && echo "results/video.md written (uses data/video/test.mov if present)"
python predict.py data/split/valid/images outputs/predictions --save-vis
ls outputs/predictions/*.txt | head -3 || true
echo "first prediction file:"; head -5 "$(ls outputs/predictions/*.txt | head -1)" || true

step "9/9 tests"
python -m pytest -q

echo
echo "Done. Read results/metrics.md (numbers), results/errors.md, results/hard_examples.md, results/video.md"
echo "and METHODOLOGY.md. Predictions are in outputs/predictions/. (scripts/benchmark.py is separate: ~10 min.)"
