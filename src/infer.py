"""Shared inference: image -> preprocess (same function as training data) -> YOLO -> normalised detections.

Used by predict.py, scripts/evaluate.py (own-matching cross-check), scripts/error_analysis.py, scripts/video_test.py
and scripts/hard_examples.py, so every consumer sees exactly the same pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

from src.preprocess import preprocess, stretch  # noqa: E402

IMGSZ = 512
NMS_IOU = 0.7  # Ultralytics default for both val and predict, so predict.py matches the reported metrics
MAX_DET = 300


def is_finished(weights: str | Path) -> bool:
    """True only for a checkpoint whose training run has ended. Ultralytics writes best.pt every epoch while
    training (optimizer state included) and strips the optimizer only at the end, so this tells a final checkpoint
    from an in-progress one."""
    import torch

    p = Path(weights)
    return p.exists() and torch.load(p, map_location="cpu", weights_only=False).get("optimizer") is None


def uses_clahe(weights: str | Path) -> bool:
    """Whether a checkpoint was trained on CLAHE images, read from the data path it recorded at training time
    (data/split_clahe, data/orig_clahe vs data/split). Keeps inference preprocessing tied to the weights, so a model
    can never be fed differently-preprocessed images than it was trained on."""
    import torch

    ck = torch.load(Path(weights), map_location="cpu", weights_only=False)
    return Path(ck["train_args"]["data"]).parent.name.endswith("_clahe")


def load_model(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def detect(model, images_bgr: list[np.ndarray], conf: float, clahe: bool = True, device: str = "cpu",
           batch: int = 16) -> list[np.ndarray]:
    """Return one (N, 6) float array per image: class_id, x_center, y_center, width, height (0-1), confidence.
    Sorted by descending confidence."""
    out = []
    for k in range(0, len(images_bgr), batch):
        chunk = [preprocess(im) if clahe else stretch(im) for im in images_bgr[k:k + batch]]
        results = model.predict(chunk, imgsz=IMGSZ, conf=conf, iou=NMS_IOU, max_det=MAX_DET, device=device,
                                verbose=False)
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), dtype=np.float32))
                continue
            arr = np.concatenate([b.cls.cpu().numpy()[:, None], b.xywhn.cpu().numpy(),
                                  b.conf.cpu().numpy()[:, None]], axis=1).astype(np.float32)
            arr[:, 1:5] = arr[:, 1:5].clip(0.0, 1.0)
            out.append(arr[np.argsort(-arr[:, 5])])
    return out
