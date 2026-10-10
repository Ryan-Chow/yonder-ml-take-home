"""End-to-end check of predict.py's output contract on real val images (skipped if weights/data are missing)."""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import predict

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("PREDICT_TEST_WEIGHTS", ROOT / "weights" / "best.pt"))
VAL = ROOT / "data" / "split" / "valid" / "images"

pytestmark = pytest.mark.skipif(not (WEIGHTS.exists() and VAL.exists()), reason="needs weights/best.pt and data/split")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    src = tmp_path_factory.mktemp("in")
    for p in sorted(VAL.iterdir())[:12]:
        shutil.copy2(p, src / p.name)
    cv2.imwrite(str(src / "blank_gray.png"), np.full((512, 512, 3), 127, np.uint8))  # nothing to detect
    cv2.imwrite(str(src / "small_640x360.jpg"), cv2.resize(cv2.imread(str(sorted(VAL.iterdir())[0])), (640, 360)))
    (src / "not_an_image.txt").write_text("ignored")
    out = tmp_path_factory.mktemp("out")
    conf = 0.25
    assert predict.main([str(src), str(out), "--conf", str(conf), "--save-vis", "--weights", str(WEIGHTS)]) == 0
    return src, out, conf


def test_one_txt_per_image_with_same_stem(run):
    src, out, _ = run
    imgs = {p.stem for p in src.iterdir() if p.suffix.lower() in {".jpg", ".png"}}
    assert {p.stem for p in out.glob("*.txt")} == imgs
    assert "not_an_image" not in {p.stem for p in out.glob("*.txt")}


def test_line_format_and_ranges(run):
    _, out, conf = run
    n = 0
    for f in out.glob("*.txt"):
        for line in f.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 6, line
            assert parts[0] in {"0", "1"}, line
            xc, yc, w, h, c = map(float, parts[1:])
            assert all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)), line
            assert w > 0 and h > 0
            assert conf <= c <= 1.0, line
            n += 1
    assert n > 0, "model produced no detections on 12 real val images"


def test_blank_image_gives_empty_file(run):
    _, out, _ = run
    assert (out / "blank_gray.txt").read_text() == ""


def test_non_square_input_still_normalised(run):
    _, out, _ = run
    for line in (out / "small_640x360.txt").read_text().splitlines():
        assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:5])


def test_matches_labels_roughly(run):
    """Sanity: on these val images most GT boxes should have a same-class prediction with IoU >= 0.5."""
    from src.matching import match_image

    src, out, _ = run
    tp = gt = 0
    for f in out.glob("*.txt"):
        lbl = ROOT / "data" / "split" / "valid" / "labels" / f.name
        if not lbl.exists():
            continue
        g = np.loadtxt(lbl, ndmin=2)
        p = np.loadtxt(f, ndmin=2) if f.read_text().strip() else np.zeros((0, 6))
        ms = match_image(p, g)
        tp += sum(m.kind == "tp" for m in ms)
        gt += len(g)
    assert gt > 0 and tp / gt >= 0.5, f"only {tp}/{gt} GT boxes recovered"


def test_stem_collision_is_rejected(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.imwrite(str(src / "a.jpg"), img)
    cv2.imwrite(str(src / "a.png"), img)
    assert predict.main([str(src), str(tmp_path / "out")]) == 1
