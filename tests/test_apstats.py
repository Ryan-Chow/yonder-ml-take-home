"""The AP here must reproduce Ultralytics' compute_ap so bootstrap numbers are comparable to the reported ones.
Note the Ultralytics convention: a perfect detector scores 0.995, not 1.0 (the last 101-point bin is halved).
"""
import numpy as np

from src.apstats import _ap, ap_per_class, image_tp, summarize


def test_matches_ultralytics_compute_ap_on_random_curves():
    from ultralytics.utils.metrics import compute_ap

    rng = np.random.default_rng(0)
    for _ in range(50):
        n, n_gt = int(rng.integers(1, 40)), int(rng.integers(1, 30))
        tp = rng.random((n, 1)) < 0.6
        tp[:, 0][np.cumsum(tp[:, 0]) > n_gt] = False  # can't have more TPs than GT
        conf = rng.random(n)
        o = np.argsort(-conf)
        rec = np.cumsum(tp[o, 0]) / n_gt
        prec = np.cumsum(tp[o, 0]) / np.arange(1, n + 1)
        ref = compute_ap(rec, prec)[0]
        assert np.isclose(_ap(tp, conf, n_gt)[0], ref), (n, n_gt)


def test_perfect_predictions_give_ultralytics_max_0995():
    g = np.array([[0, 0.3, 0.3, 0.2, 0.2], [1, 0.7, 0.7, 0.2, 0.2]])
    p = np.array([[0, 0.3, 0.3, 0.2, 0.2, 0.9], [1, 0.7, 0.7, 0.2, 0.2, 0.8]])
    s = summarize(ap_per_class([image_tp(p, g)]))
    assert np.isclose(s["mAP50"], 0.995) and np.isclose(s["mAP50-95"], 0.995)


def test_missing_half_gives_about_half():
    g = np.array([[0, 0.3, 0.3, 0.2, 0.2], [0, 0.7, 0.7, 0.2, 0.2]])
    p = np.array([[0, 0.3, 0.3, 0.2, 0.2, 0.9]])
    ap = ap_per_class([image_tp(p, g)])
    assert np.isclose(ap[0, 0], 0.5, atol=0.01)
    assert np.isnan(ap[1, 0])  # no GT of class 1 -> excluded from the mean


def test_high_conf_fp_ranked_first_lowers_ap():
    g = np.array([[0, 0.3, 0.3, 0.2, 0.2]])
    good = np.array([[0, 0.3, 0.3, 0.2, 0.2, 0.5]])
    with_fp = np.array([[0, 0.8, 0.8, 0.1, 0.1, 0.9], [0, 0.3, 0.3, 0.2, 0.2, 0.5]])
    assert ap_per_class([image_tp(with_fp, g)])[0, 0] < ap_per_class([image_tp(good, g)])[0, 0]


def test_wrong_class_is_not_tp_and_loose_box_fails_strict_iou():
    g = np.array([[0, 0.5, 0.5, 0.2, 0.2]])
    assert not image_tp(np.array([[1, 0.5, 0.5, 0.2, 0.2, 0.9]]), g)["tp"].any()
    loose = image_tp(np.array([[0, 0.52, 0.5, 0.2, 0.2, 0.9]]), g)["tp"]  # IoU ~0.82
    assert loose[0, 0] and not loose[0, -1]


def test_bootstrap_repeats_are_concatenated():
    g = np.array([[0, 0.3, 0.3, 0.2, 0.2]])
    d = image_tp(np.array([[0, 0.3, 0.3, 0.2, 0.2, 0.9]]), g)
    assert np.isclose(ap_per_class([d, d, d])[0, 0], 0.995)
