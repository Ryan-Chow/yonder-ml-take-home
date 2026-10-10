"""Average precision computed from scratch, so it can be bootstrapped over images.

COCO-style: per class and IoU threshold, predictions are matched greedily in descending confidence to the
highest-IoU unmatched GT of the same class; AP is the area under the precision envelope sampled at 101 recall
points, with the same sentinels and interpolation as Ultralytics' compute_ap (tests/test_apstats.py compares the
two directly). Matching is precomputed per image, so a bootstrap
resample is just a concatenation of per-image arrays.
"""
from __future__ import annotations

import numpy as np

from src.matching import iou_matrix

IOU_THRS = np.linspace(0.5, 0.95, 10)


def image_tp(preds: np.ndarray, gts: np.ndarray) -> dict:
    """preds (N,6) cls,xc,yc,w,h,conf; gts (M,5). Returns per-image arrays for AP: cls, conf, tp (N,10), n_gt."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 6)
    gts = np.asarray(gts, dtype=np.float64).reshape(-1, 5)
    tp = np.zeros((len(preds), len(IOU_THRS)), dtype=bool)
    if len(preds) and len(gts):
        ious = iou_matrix(preds[:, 1:5], gts[:, 1:5])
        same = preds[:, 0][:, None] == gts[:, 0][None, :]
        ious = np.where(same, ious, 0.0)
        order = np.argsort(-preds[:, 5], kind="stable")
        for t, thr in enumerate(IOU_THRS):
            used = np.zeros(len(gts), dtype=bool)
            for p in order:
                cand = np.where(~used & (ious[p] >= thr))[0]
                if len(cand):
                    g = cand[np.argmax(ious[p, cand])]
                    used[g] = True
                    tp[p, t] = True
    n_gt = np.bincount(gts[:, 0].astype(int), minlength=2) if len(gts) else np.zeros(2, int)
    return {"cls": preds[:, 0].astype(int), "conf": preds[:, 5], "tp": tp, "n_gt": n_gt}


def _ap(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    if n_gt == 0:
        return np.full(tp.shape[1], np.nan)
    if len(conf) == 0:
        return np.zeros(tp.shape[1])
    o = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[o], 0)
    fpc = np.cumsum(~tp[o], 0)
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc)
    out = np.zeros(tp.shape[1])
    x = np.linspace(0, 1, 101)
    for t in range(tp.shape[1]):
        # sentinels exactly as ultralytics.utils.metrics.compute_ap (8.4.x): precision falls to 0 right after
        # the last achieved recall
        mrec = np.concatenate([[0.0], rec[:, t], [rec[-1, t]], [1.0]])
        mpre = np.concatenate([[1.0], prec[:, t], [0.0], [0.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        out[t] = np.trapezoid(np.interp(x, mrec, mpre), x)
    return out


def ap_per_class(items: list[dict], nc: int = 2) -> np.ndarray:
    """(nc, 10) AP matrix for a list of image_tp() dicts (may contain repeats, as in a bootstrap sample)."""
    cls = np.concatenate([d["cls"] for d in items]) if items else np.zeros(0, int)
    conf = np.concatenate([d["conf"] for d in items]) if items else np.zeros(0)
    tp = np.concatenate([d["tp"] for d in items]) if items else np.zeros((0, len(IOU_THRS)), bool)
    n_gt = np.sum([d["n_gt"] for d in items], axis=0)
    return np.stack([_ap(tp[cls == c], conf[cls == c], int(n_gt[c])) for c in range(nc)])


def summarize(ap: np.ndarray) -> dict:
    """mAP50, mAP50-95 (means over classes) and per-class AP50 / AP50-95."""
    return {"mAP50": float(np.nanmean(ap[:, 0])), "mAP50-95": float(np.nanmean(ap.mean(1))),
            "ap50": ap[:, 0].tolist(), "ap50_95": ap.mean(1).tolist()}
