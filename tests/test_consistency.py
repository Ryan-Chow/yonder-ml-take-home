"""The shipped inference script must reproduce the reported numbers exactly.

Runs predict.py (default threshold from results/threshold.json) on the grouped val images, re-scores its .txt
files against the labels and compares TP/FP/FN per class with results/metrics.json. Skipped if data/weights/results
are missing.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import predict
from src.matching import confusion

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data" / "split" / "valid"
METRICS = ROOT / "results" / "metrics.json"

pytestmark = pytest.mark.skipif(not ((ROOT / "weights" / "best.pt").exists() and VAL.exists() and METRICS.exists()),
                                reason="needs weights, data/split and results/metrics.json")


def test_predict_txt_reproduces_reported_counts(tmp_path):
    assert predict.main([str(VAL / "images"), str(tmp_path)]) == 0
    preds, gts = [], []
    for lbl in sorted((VAL / "labels").glob("*.txt")):
        gts.append(np.loadtxt(lbl, ndmin=2).reshape(-1, 5))
        txt = (tmp_path / lbl.name).read_text()
        preds.append(np.loadtxt(tmp_path / lbl.name, ndmin=2).reshape(-1, 6) if txt.strip() else np.zeros((0, 6)))
    cm = confusion(preds, gts, nc=2)
    reported = json.loads(METRICS.read_text())["own_matching"]["final"]["confusion"]
    assert cm.tolist() == reported
