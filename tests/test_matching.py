import numpy as np

from src.matching import confusion, iou_matrix, match_image


def test_iou_identical_and_disjoint():
    a = np.array([[0.5, 0.5, 0.2, 0.2]])
    assert np.isclose(iou_matrix(a, a)[0, 0], 1.0)
    assert iou_matrix(a, np.array([[0.1, 0.1, 0.05, 0.05]]))[0, 0] == 0.0


def test_iou_half_overlap():
    # two 0.2x0.2 boxes shifted by 0.1 in x: inter 0.1*0.2=0.02, union 0.06
    assert np.isclose(iou_matrix(np.array([[0.5, 0.5, 0.2, 0.2]]), np.array([[0.6, 0.5, 0.2, 0.2]]))[0, 0], 1 / 3)


def kinds(preds, gts):
    return sorted(m.kind for m in match_image(np.array(preds), np.array(gts)))


def test_tp_fp_fn():
    gts = [[0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]]
    preds = [[0, 0.5, 0.5, 0.2, 0.2, 0.9], [0, 0.9, 0.9, 0.1, 0.1, 0.8]]
    assert kinds(preds, gts) == ["fn", "fp", "tp"]


def test_wrong_class_and_duplicate():
    gts = [[1, 0.5, 0.5, 0.2, 0.2]]
    # correct class wins even with lower confidence; the higher-conf wrong-class box on the same object is then
    # left over -> FP (duplicate), not wrong_class, because the GT is already taken
    preds = [[0, 0.5, 0.5, 0.2, 0.2, 0.9], [1, 0.5, 0.5, 0.2, 0.2, 0.6]]
    assert kinds(preds, gts) == ["fp", "tp"]
    assert kinds([[0, 0.5, 0.5, 0.2, 0.2, 0.9]], gts) == ["wrong_class"]


def test_no_preds_no_gts():
    assert kinds(np.zeros((0, 6)), np.zeros((0, 5))) == []
    assert kinds(np.zeros((0, 6)), [[0, 0.5, 0.5, 0.1, 0.1]]) == ["fn"]


def test_confusion_layout():
    gts = [np.array([[0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]])]
    preds = [np.array([[1, 0.5, 0.5, 0.2, 0.2, 0.9], [0, 0.9, 0.9, 0.1, 0.1, 0.5]])]
    cm = confusion(preds, gts, nc=2)
    assert cm[1, 0] == 1  # predicted mallet on a bottle
    assert cm[2, 1] == 1  # missed mallet
    assert cm[0, 2] == 1  # bottle on background
    assert cm.sum() == 3
