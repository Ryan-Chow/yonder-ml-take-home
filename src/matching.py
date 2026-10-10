"""Prediction <-> ground-truth matching at a fixed IoU, independent of Ultralytics' validator.

Two passes per image:
  1. same-class greedy matching in descending confidence (IoU >= thr)            -> TP
  2. leftover predictions vs leftover GT of a *different* class (IoU >= thr)     -> wrong-class
  3. anything still unmatched: prediction -> FP (background), GT -> FN (missed)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def xywh_to_xyxy(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1)


def iou_matrix(a_xywh: np.ndarray, b_xywh: np.ndarray) -> np.ndarray:
    a, b = xywh_to_xyxy(a_xywh), xywh_to_xyxy(b_xywh)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


@dataclass
class Match:
    kind: str           # "tp", "fp", "fn", "wrong_class"
    gt_idx: int | None
    pred_idx: int | None
    iou: float = 0.0


def match_image(preds: np.ndarray, gts: np.ndarray, iou_thr: float = 0.5) -> list[Match]:
    """preds: (N,6) cls,xc,yc,w,h,conf sorted by conf desc. gts: (M,5) cls,xc,yc,w,h."""
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 6)
    gts = np.asarray(gts, dtype=np.float64).reshape(-1, 5)
    ious = iou_matrix(preds[:, 1:5], gts[:, 1:5])
    order = np.argsort(-preds[:, 5], kind="stable")
    used_p, used_g, out = set(), set(), []
    for same_class in (True, False):
        for p in order:
            if p in used_p:
                continue
            best, best_iou = None, iou_thr
            for g in range(len(gts)):
                if g in used_g or (int(gts[g, 0]) == int(preds[p, 0])) != same_class:
                    continue
                if ious[p, g] >= best_iou:
                    best, best_iou = g, ious[p, g]
            if best is not None:
                used_p.add(p)
                used_g.add(best)
                out.append(Match("tp" if same_class else "wrong_class", best, int(p), float(best_iou)))
    out += [Match("fp", None, int(p)) for p in order if p not in used_p]
    out += [Match("fn", g, None) for g in range(len(gts)) if g not in used_g]
    return out


def confusion(all_preds: list[np.ndarray], all_gts: list[np.ndarray], nc: int, iou_thr: float = 0.5) -> np.ndarray:
    """(nc+1)x(nc+1) matrix, rows = predicted class (last = no prediction), cols = true class (last = background)."""
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for preds, gts in zip(all_preds, all_gts):
        for m in match_image(preds, gts, iou_thr):
            if m.kind in ("tp", "wrong_class"):
                cm[int(preds[m.pred_idx, 0]), int(gts[m.gt_idx, 0])] += 1
            elif m.kind == "fp":
                cm[int(preds[m.pred_idx, 0]), nc] += 1
            else:
                cm[nc, int(gts[m.gt_idx, 0])] += 1
    return cm
