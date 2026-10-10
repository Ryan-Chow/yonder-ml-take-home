# CLAHE preprocessing: effect on the dataset

`src/preprocess.py`: CLAHE on LAB L, clipLimit=2.0, tileGridSize=(8, 8). All 999 images.

| | mean L | spread of mean L across images | mean contrast (std L) | dark | bright | low contrast |
|---|---|---|---|---|---|---|
| raw | 120 | 40 | 55 | 15% | 8% | 13% |
| CLAHE | 123 | 32 | 63 | 7% | 7% | 5% |

- Mean absolute change of the LAB a/b (colour) channels: 0.11 / 255, i.e. hue is essentially untouched; only lightness is redistributed.
- CLAHE is a *local* operation: it raises contrast inside each tile but deliberately does not normalise global exposure fully (the clip limit bounds the change). Overall exposure robustness is therefore left to the HSV-value augmentation in `configs/train.yaml`.

`results/clahe_examples.jpg`: rows = darkest / brightest / lowest-contrast train images; left raw, right CLAHE (GT boxes drawn).
