# Video test

> **Stand-in only: the real phone video is still pending.** No `data/video/test.mov` was available when this ran. This clip was synthesised from 30 grouped-*val* images (never trained on), resized to 1280x720, with simulated zoom, horizontal motion blur and exposure drift (gain 0.6-1.4). It tests the frame pipeline and robustness to blur/exposure, **not** real out-of-distribution performance. To run the real test: put the video at `data/video/test.mov` and run `python scripts/video_test.py data/video/test.mov --every 10`.

- Source: `data/video/demo_from_val.mp4`, 1280x720, 10.0 fps, 270 frames; one frame in every 3 -> 90 frames analysed.
- Pipeline: identical to `predict.py` (stretch to 512x512 + CLAHE), conf >= 0.426, CPU.

| class | frames with >=1 detection | median max-conf when detected | presence flicker |
|---|---|---|---|
| bottle | 42/90 (47%) | 0.82 | 7% |
| mallet | 48/90 (53%) | 0.72 | 0% |

Presence flicker = share of consecutive analysed frames where the class appears/disappears (within a shot; changes at the cuts between source images don't count).

Frame-level check against the source labels (is the class detected in frames where it is present?):

| class | frames where present | detected | frames with a detection but class absent |
|---|---|---|---|
| bottle | 48 | 40 (83%) | 2 |
| mallet | 48 | 45 (94%) | 3 |

Source images behind the 'class absent' frames: `20231021_174410_jpg.rf.d1f0e08aaae570d02`, `20231021_174551_jpg.rf.603a928a7cee8e31e`, `t_bot_mal581_jpeg_jpg.rf.45604396d6a77c4`.

![confidence over time](video_conf.png)

Annotated frames (evenly spaced, 30 saved): `results/video/`.
