import cv2
import numpy as np

from src.preprocess import CLIP_LIMIT, TILE_GRID, TRAIN_SIZE, preprocess


def _clahe_only(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID).apply(lab[..., 0].copy())
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def test_512_input_is_clahe_only_and_deterministic():
    img = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out, _clahe_only(img))
    assert np.array_equal(out, preprocess(img))


def test_non_square_is_stretched_to_training_size():
    img = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    assert preprocess(img).shape == (TRAIN_SIZE, TRAIN_SIZE, 3)


def test_grayscale_input_ok():
    assert preprocess(np.zeros((100, 50), np.uint8)).shape == (TRAIN_SIZE, TRAIN_SIZE, 3)


def test_input_not_modified_in_place():
    img = np.full((512, 512, 3), 40, np.uint8)
    before = img.copy()
    preprocess(img)
    assert np.array_equal(img, before)


def test_training_pngs_equal_inference_pixels():
    """data/split_clahe PNGs (training input) must equal preprocess(raw JPEG) computed at inference time."""
    from pathlib import Path

    import pytest

    root = Path(__file__).resolve().parents[1] / "data"
    raw = sorted((root / "split" / "valid" / "images").glob("*.jpg"))[:5]
    if not raw:
        pytest.skip("data/split not built")
    for p in raw:
        png = cv2.imread(str(root / "split_clahe" / "valid" / "images" / f"{p.stem}.png"))
        assert np.array_equal(png, preprocess(cv2.imread(str(p))))
