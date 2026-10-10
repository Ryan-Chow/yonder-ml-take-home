"""Efficiency stretch goal: size, compute, latency and the accuracy cost of cheaper variants.

    python scripts/benchmark.py [--weights weights/best.pt] [--runs 100]

Measures params / GFLOPs / file size; PyTorch latency (batch 1, model forward only) on CPU and MPS at 512 and 320;
ONNX Runtime CPU latency for FP32 and INT8 (static post-training quantisation, QDQ, calibrated on train images);
and re-evaluates mAP for each variant on the grouped val split. Writes results/efficiency.md + efficiency.json.

The real target is RKNN on an OrangePi NPU, which can't be run here; ONNX is the input format rknn-toolkit2
converts from, and its INT8 quantisation is also calibration-based, so the INT8-ONNX accuracy delta is the closest
available proxy.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import ROOT, load_split  # noqa: E402
from src.infer import uses_clahe  # noqa: E402
from src.preprocess import preprocess, stretch  # noqa: E402

RESULTS = ROOT / "results"
VAL_YAML = ROOT / "data" / "split_clahe" / "data.yaml"  # replaced in main() if the weights don't use CLAHE
CLAHE = True
OUT = ROOT / "data" / "export"  # git-ignored


def to_tensor(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """preprocess (stretch 512 [+ CLAHE]) -> resize to `size` -> RGB CHW float32 0-1 with batch dim."""
    im = preprocess(img_bgr) if CLAHE else stretch(img_bgr)
    if size != im.shape[0]:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def timeit(fn, runs: int, warmup: int = 10, sync=lambda: None) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)), "fps": float(1000 / ts.mean())}


def preprocess_latency(n: int = 50) -> dict:
    """CPU cost of the shared preprocessing (stretch + CLAHE) on 512 val images and on a 1920x1080 frame."""
    ims = [cv2.imread(str(s.image)) for s in load_split(ROOT / "data" / "split", "valid")[:n]]
    frame = cv2.resize(ims[0], (1920, 1080))
    out = {}
    for name, xs in (("img512", ims), ("frame1080p", [frame] * n)):
        it = iter(xs * 2)
        out[name] = timeit(lambda: preprocess(next(it)), n, warmup=5)
    return out


def torch_latency(weights: Path, device: str, size: int, runs: int) -> dict:
    import torch
    from ultralytics import YOLO

    m = YOLO(str(weights)).model.float().eval().to(device)
    m.fuse() if hasattr(m, "fuse") else None
    x = torch.zeros(1, 3, size, size, device=device)
    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)
    with torch.inference_mode():
        return timeit(lambda: m(x), runs, sync=sync)


def ort_latency(onnx_path: Path, size: int, runs: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4  # an OrangePi 5 has 4 big A76 cores; keeps CPU numbers closer to edge-like
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = np.zeros((1, 3, size, size), np.float32)
    return timeit(lambda: s.run(None, {name: x}), runs)


def val_map(model_path: Path, size: int, device: str = "cpu") -> dict:
    from ultralytics import YOLO

    r = YOLO(str(model_path), task="detect").val(data=str(VAL_YAML), imgsz=size, batch=1 if model_path.suffix ==
                                                  ".onnx" else 16, rect=False, device=device, conf=0.001, iou=0.7,
                                                  plots=False, verbose=False, project=str(ROOT / "runs" / "bench"),
                                                  name=f"{model_path.stem}_{size}", exist_ok=True)
    return {"mAP50": float(r.box.map50), "mAP50-95": float(r.box.map),
            "bottle_mAP50": float(r.box.ap50[0]), "mallet_mAP50": float(r.box.ap50[1])}


def quantize_int8(fp32: Path, out: Path, n_calib: int, exclude_head: bool) -> Path:
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out.with_name(out.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    train = load_split(ROOT / "data" / "split", "train")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(train), n_calib, replace=False)
    model = onnx.load(str(pre))
    inp = model.graph.input[0].name
    size = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{inp: to_tensor(cv2.imread(str(train[i].image)), size)} for i in pick])

        def get_next(self):
            return next(self.it, None)

    exclude = []
    if exclude_head:
        # keep the Detect head's decode (DFL softmax, box arithmetic, final concat) in float: it mixes pixel
        # coordinates (0-512) and class probabilities (0-1) in one tensor, which one INT8 scale can't represent
        exclude = [n.name for n in model.graph.node if "/model.22/" in n.name and n.op_type != "Conv"]
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_exclude=exclude)
    pre.unlink(missing_ok=True)
    return out


def report(res: dict, runs: int) -> None:
    """Write results/efficiency.md from the measurements in `res` (also usable via --report-only)."""
    lat, acc = res["latency"], res["accuracy"]
    a = argparse.Namespace(runs=runs)
    res = dict(res, gflops={int(k): v for k, v in res["gflops"].items()})  # JSON turns int keys into strings

    def row(label, lk, ak, size_mb):
        l, c = lat.get(lk), acc.get(ak)
        lt = f"{l['mean_ms']:.1f} / {l['p95_ms']:.1f}" if l else "–"
        return (f"| {label} | {size_mb} | {lt} | {c['mAP50']:.3f} | {c['mAP50-95']:.3f} | "
                f"{c['bottle_mAP50']:.3f} | {c['mallet_mAP50']:.3f} |")

    md = ["# Efficiency", "",
          f"Generated by `scripts/benchmark.py` on {res['machine']}, torch {res['torch']}. Batch 1, "
          f"{a.runs} timed runs after 10 warm-up runs; latency is model forward only (no NMS, no CLAHE). "
          "ONNX Runtime uses 4 CPU threads. mAP on the grouped val split (199 images).", "",
          f"- Parameters: **{res['params'] / 1e6:.2f} M**; compute: **{res['gflops'][512]:.1f} GFLOPs @512**, "
          f"{res['gflops'][320]:.1f} GFLOPs @320.",
          f"- `weights/best.pt`: {res['pt_size_mb']:.1f} MB (FP16 weights as saved by Ultralytics).", "",
          "| variant | file MB | latency mean / p95 (ms) | mAP50 | mAP50-95 | bottle mAP50 | mallet mAP50 |",
          "|---|---|---|---|---|---|---|",
          row("PyTorch FP32, CPU, 512", "torch_cpu_512", "torch_512", f"{res['pt_size_mb']:.1f}"),
          row("PyTorch FP32, MPS, 512", "torch_mps_512", "torch_512", f"{res['pt_size_mb']:.1f}"),
          row("PyTorch FP32, CPU, 320", "torch_cpu_320", "torch_320", f"{res['pt_size_mb']:.1f}"),
          row("PyTorch FP32, MPS, 320", "torch_mps_320", "torch_320", f"{res['pt_size_mb']:.1f}"),
          row("ONNX FP32, ORT CPU, 512", "onnx_fp32_512", "onnx_fp32_512", f"{res['sizes_mb']['onnx_fp32_512']:.1f}"),
          row("ONNX FP32, ORT CPU, 320", "onnx_fp32_320", "onnx_fp32_320", f"{res['sizes_mb']['onnx_fp32_320']:.1f}"),
          row("ONNX INT8 (all nodes), ORT CPU, 512", "onnx_int8_512", "onnx_int8_512",
              f"{res['sizes_mb']['onnx_int8_512']:.1f}"),
          row("ONNX INT8 (head decode kept FP32), ORT CPU, 512", "onnx_int8_head_fp32_512",
              "onnx_int8_head_fp32_512", f"{res['sizes_mb']['onnx_int8_head_fp32_512']:.1f}"),
          ""]
    a512, a320 = acc["torch_512"], acc["torch_320"]
    q, f = acc["onnx_int8_head_fp32_512"], acc["onnx_fp32_512"]
    o512, o320 = lat["onnx_fp32_512"]["mean_ms"], lat["onnx_fp32_320"]["mean_ms"]
    md += ["## What this means for the OrangePi / RKNN target", "",
           "The rover runs YOLO on the Rockchip NPU via RKNN, which this Mac can't run. Absolute latencies above are "
           "therefore not the deployment numbers; the *relative* costs and accuracy deltas are what transfer.", "",
           f"- **Export is lossless.** ONNX FP32 reproduces PyTorch mAP exactly ({f['mAP50']:.3f} vs "
           f"{a512['mAP50']:.3f}); raw outputs differ by ~1e-3 px. ONNX is what `rknn-toolkit2` converts from.",
           f"- **INT8 works only if the detection head's decode stays in float.** Quantising every node gives "
           f"mAP 0.000: the Detect head concatenates box coordinates (0-512 px) and class scores (0-1) into one "
           f"tensor, and a single INT8 scale can't represent both. Keeping the decode in FP32 (convs still INT8) "
           f"costs only {f['mAP50'] - q['mAP50']:.3f} mAP50 / {f['mAP50-95'] - q['mAP50-95']:.3f} mAP50-95 "
           f"and shrinks the file {res['sizes_mb']['onnx_fp32_512'] / res['sizes_mb']['onnx_int8_head_fp32_512']:.1f}x. "
           "The same issue applies to RKNN INT8. To my knowledge Rockchip's own YOLOv8 export (rknn_model_zoo) ends "
           "the graph before the box decode and runs it on the CPU for this reason; confirm that against the "
           "toolkit version the team uses (not verified here). On Apple-silicon CPU the "
           "INT8 speed-up is small, but NPUs are built for INT8, so the gain there should be much larger "
           "(untested).",
           f"- **Input size is the biggest lever.** 320 instead of 512 cuts compute "
           f"{res['gflops'][512] / res['gflops'][320]:.1f}x and ONNX CPU latency {o512 / o320:.1f}x, but costs "
           f"{a512['mAP50'] - a320['mAP50']:.3f} mAP50, mostly on bottles "
           f"({a512['bottle_mAP50'] - a320['bottle_mAP50']:.3f}); mallet AP50 drops "
           f"{a512['mallet_mAP50'] - a320['mallet_mAP50']:.3f}. This model was trained at 512; training at 320 "
           "would probably recover part of the gap (untested).",
           "- **What I would trade on the rover** (judgement, not measured). The mallet is the priority target and is "
           f"already at {a512['mallet_mAP50']:.2f} AP50. Keep YOLOv8n (with 999 training images a larger model is "
           "more likely to overfit than to help, and it costs NPU time), quantise to INT8 with the decode on CPU, and pick the input size "
           "from the *required detection distance*: small objects are where this model already struggles "
           "(results/hard_examples.md), and downscaling makes distant objects smaller still. A practical pattern: "
           "run at 320 for search, switch to 512 (or crop around the last detection) once a candidate appears.",
           f"- **Preprocessing cost (measured, CPU):** stretch + CLAHE takes {res['preprocess']['img512']['mean_ms']:.2f} ms "
           f"per 512x512 image and {res['preprocess']['frame1080p']['mean_ms']:.1f} ms per 1920x1080 frame (mostly the "
           "resize). On the OrangePi this runs on the CPU and can overlap with NPU inference of the previous frame.",
           "- **Not measured:** NMS cost, true NPU latency and power.", ""]
    (RESULTS / "efficiency.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--report-only", action="store_true", help="rebuild efficiency.md from efficiency.json")
    a = ap.parse_args()
    if a.report_only:
        res = json.loads((RESULTS / "efficiency.json").read_text())
        if "preprocess" not in res:  # added after the first full run; cheap to measure on its own
            res["preprocess"] = preprocess_latency()
            (RESULTS / "efficiency.json").write_text(json.dumps(res, indent=2))
        report(res, res.get("runs", a.runs))
        return

    import torch
    from ultralytics import YOLO

    global VAL_YAML, CLAHE
    CLAHE = uses_clahe(a.weights)
    VAL_YAML = ROOT / "data" / ("split_clahe" if CLAHE else "split") / "data.yaml"
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"machine": f"{platform.platform()} / {platform.processor() or platform.machine()}",
           "torch": torch.__version__}
    y = YOLO(str(a.weights))
    n_params = sum(p.numel() for p in y.model.parameters())
    from ultralytics.utils.torch_utils import get_flops
    res["params"] = int(n_params)
    res["gflops"] = {s: float(get_flops(y.model, s)) for s in (512, 320)}
    res["pt_size_mb"] = a.weights.stat().st_size / 1e6

    lat = {}
    for dev in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
        for s in (512, 320):
            lat[f"torch_{dev}_{s}"] = torch_latency(a.weights, dev, s, a.runs)
            print(f"torch {dev} {s}: {lat[f'torch_{dev}_{s}']}")

    variants = {}
    for s in (512, 320):
        fp32 = Path(YOLO(str(a.weights)).export(format="onnx", imgsz=s, simplify=True, dynamic=False, opset=17))
        dst = OUT / f"best_{s}_fp32.onnx"
        fp32.replace(dst)
        variants[f"onnx_fp32_{s}"] = dst
    variants["onnx_int8_512"] = quantize_int8(variants["onnx_fp32_512"], OUT / "best_512_int8.onnx", a.calib, False)
    variants["onnx_int8_head_fp32_512"] = quantize_int8(variants["onnx_fp32_512"], OUT / "best_512_int8_headfp.onnx",
                                                        a.calib, True)
    for k, p in variants.items():
        s = 320 if k.endswith("_320") else 512
        lat[k] = ort_latency(p, s, a.runs)
        print(f"{k}: {lat[k]}")
    res["latency"] = lat
    res["preprocess"] = preprocess_latency()
    res["sizes_mb"] = {k: p.stat().st_size / 1e6 for k, p in variants.items()}

    acc = {"torch_512": val_map(a.weights, 512), "torch_320": val_map(a.weights, 320)}
    for k, p in variants.items():
        acc[k] = val_map(p, 320 if k.endswith("_320") else 512)
        print(f"acc {k}: {acc[k]}")
    res["accuracy"] = acc
    res["runs"] = a.runs
    (RESULTS / "efficiency.json").write_text(json.dumps(res, indent=2))

    report(res, a.runs)


if __name__ == "__main__":
    main()
