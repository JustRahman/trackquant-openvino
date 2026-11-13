# TrackQuant-OV: final results

One table per question. Every table names its source file. All runs are single runs on one machine.
Speeds are comparable only inside the same run.

Setup for everything below:
- Kaggle, Intel Xeon @ 2.20 GHz, 4 vCPU, 31 GB RAM (AVX2). ultralytics 8.4.211, OpenVINO 2026.4.1, NNCF 3.4.0, torch 2.11.0 CPU.
- MOT17 train (official zip), FRCNN folder. Test sequences MOT17-02 (static camera), MOT17-04 (crowded, high view), MOT17-11 (moving camera).
  Scores are the three sequences combined.
- INT8 calibration: 300 frames of MOT17-10, which is never scored. NNCF post-training quantization through the ultralytics OpenVINO export.
- Input 384x640 for every variant (what `imgsz=640` rect inference gives PyTorch on a 1080p frame).
- Tracker: ByteTrack, stock `bytetrack.yaml`, person class only. `track_low_thresh` is set to the detector threshold (see section 5 for why).
- Metrics: HOTA, CLEAR and Identity with my own implementation of the TrackEval formulas and MOTChallenge pre-processing.
  Cross-checked against PyPI `trackeval` 1.3.0 on one run: HOTA identical to 3 decimals, IDSW identical (section 7).
- `fps_e2e` = pre-process + inference + post-process + ByteTrack. JPEG decoding is not included.

---

## 1. YOLOv8n: accuracy and speed per precision

Source: `results/kaggle_yolov8_v2/p2_report.md`. Each variant at its own best threshold, detect every frame.

| variant | best conf | HOTA | IDF1 | MOTA | IDSW/min | inference ms | fps_e2e | model MB |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.03 | 40.30 | 47.40 | 34.72 | 82.6 | 90.2 | 10.04 | 6.25 |
| OpenVINO FP32 | 0.03 | 40.30 | 47.40 | 34.72 | 82.6 | 64.7 | 13.73 | 12.3 |
| OpenVINO FP16 | 0.01 | 40.48 | 47.64 | 34.07 | 92.5 | 63.4 | 13.66 | 6.33 |
| OpenVINO INT8 | 0.01 | **39.71** | 46.86 | 35.30 | 102.4 | **42.5** | **19.23** | **3.54** |

- OpenVINO FP32 gives exactly the same tracks as PyTorch.
- INT8 vs OpenVINO FP32: HOTA -0.59, IDF1 -0.55, ID switches +19.8 per minute, 1.40x faster end to end, 1.52x faster inference, 3.5x smaller.
- INT8 vs PyTorch FP32: 1.92x faster end to end.

## 2. YOLO26n: accuracy and speed per precision

Source: `results/kaggle_yolo26_v2/p2_report.md`. OpenVINO models exported with `nms=True, conf=0.01` (section 6 explains why).

| variant | best conf | HOTA | IDF1 | MOTA | IDSW/min | inference ms | fps_e2e | model MB |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 36.68 | 40.70 | 30.39 | 75.5 | 65.2 | 13.91 | 5.29 |
| OpenVINO FP32 | 0.01 | 36.68 | 40.70 | 30.39 | 75.5 | 39.8 | 21.72 | 9.64 |
| OpenVINO FP16 | 0.01 | 36.67 | 40.71 | 30.42 | 76.9 | 38.8 | 22.18 | 5.14 |
| OpenVINO INT8 | 0.03 | **36.38** | 40.82 | 30.85 | 76.9 | **29.2** | **28.64** | **3.21** |

- OpenVINO FP32 gives exactly the same tracks as PyTorch (same HOTA, IDF1, MOTA and IDSW at every threshold).
- INT8 vs OpenVINO FP32: HOTA -0.29, IDF1 +0.12, ID switches +1.4 per minute, 1.32x faster end to end, 1.36x faster inference, 3.0x smaller.
- INT8 vs PyTorch FP32: 2.06x faster end to end.
- YOLO26n scores lower than YOLOv8n on these three sequences (36.7 vs 40.3 HOTA). I did not look into why.

## 3. Threshold drift after INT8

Sources: section 2 of both v2 reports. Grid: conf 0.01, 0.03, 0.05, 0.1, 0.2.
Question: if you keep the FP32 best threshold after quantizing, how much do you lose?

| model | FP32 best conf | INT8 best conf | INT8 HOTA at FP32 conf | INT8 HOTA at own conf | gain from re-tuning | IDF1 gain |
|---|---|---|---|---|---|---|
| YOLOv8n | 0.03 | 0.01 | 39.60 | 39.71 | +0.10 | +0.17 |
| YOLO26n | 0.01 | 0.03 | 35.12 | 36.38 | **+1.26** | **+1.92** |

For YOLO26n, re-tuning the threshold after INT8 matters more than the INT8 loss itself (-0.29 HOTA when re-tuned, -1.55 when not).

## 4. Detect every N frames

Sources: section 3 of both v2 reports. Frames without detection run only the Kalman predict step, so tracks keep their IDs.

YOLOv8n:

| variant | N | HOTA | IDSW/min | fps_e2e | speed vs PyTorch N=1 |
|---|---|---|---|---|---|
| PyTorch FP32 | 1 | 40.30 | 82.6 | 10.04 | 1.0x |
| OpenVINO FP32 | 3 | 40.23 | 67.1 | 42.73 | 4.3x |
| OpenVINO INT8 | 1 | 39.71 | 102.4 | 19.23 | 1.9x |
| OpenVINO INT8 | 3 | 38.55 | 74.1 | 56.08 | **5.6x** |
| OpenVINO INT8 | 5 | 37.29 | 83.3 | 91.24 | 9.1x |

YOLO26n:

| variant | N | HOTA | IDSW/min | fps_e2e | speed vs PyTorch N=1 |
|---|---|---|---|---|---|
| PyTorch FP32 | 1 | 36.68 | 75.5 | 13.91 | 1.0x |
| OpenVINO FP32 | 3 | 35.81 | 53.7 | 63.82 | 4.6x |
| OpenVINO INT8 | 1 | 36.38 | 76.9 | 28.64 | 2.1x |
| OpenVINO INT8 | 3 | 35.24 | 52.9 | 83.73 | **6.0x** |
| OpenVINO INT8 | 5 | 32.86 | 60.0 | 137.39 | 9.9x |

- Detecting every 3rd frame costs 0.1 to 1.2 HOTA and gives 3x the speed.
- ID switches per minute go down from N=1 to N=3, probably because the tracker sees fewer noisy boxes. At N=5 they go up again.
- `fps_e2e` does not include JPEG reading. With it (`fps_wall`), YOLOv8n INT8 N=3 is 34.1 fps and YOLO26n INT8 N=3 is 46.4 fps.

## 5. Low detector thresholds need a lower ByteTrack threshold

Source: `results/kaggle_followup/p2f_report.md`, section 1. YOLO26n, the first export (`nms=False`).

Stock `bytetrack.yaml` throws away every box under `track_low_thresh = 0.1`. So setting the detector threshold under 0.1 does nothing:
every metric is identical at conf 0.01, 0.03, 0.05 and 0.1.

| variant | stock ByteTrack, best HOTA | `track_low_thresh` = conf, HOTA at 0.01 | gain |
|---|---|---|---|
| PyTorch FP32 | 34.66 (same at every conf) | 36.68 | +2.01 |
| OpenVINO FP32 | 33.77 | 36.20 | +2.43 |
| OpenVINO INT8 | 32.76 | 33.91 | +1.15 |

All v2 runs set `track_low_thresh` to the detector threshold.

## 6. A YOLO26 OpenVINO export problem: the default export does not match PyTorch

Source: `results/kaggle_followup/p2f_report.md`, section 2. Same frames, same input tensor.

The first YOLO26n run (`results/kaggle_yolo26/`) showed OpenVINO FP32 tracking worse than PyTorch FP32. I checked where the difference starts.

- The pre-processed input tensors are identical (0 of 737,280 values differ).
- With the default export (`nms=False`, end-to-end head) the boxes already differ before the tracker.
  MOT17-02, score > 0.1, per frame: 9.8 boxes match (IoU > 0.9), 8.0 are PyTorch-only, 6.6 are OpenVINO-only. Mean score difference 0.07.
- The same happens at 640x640 letterbox and with a dynamic-shape export. It is not the input size.
- Effect on tracking, MOT17-02 at conf 0.1: ID switches 32 (PyTorch) vs 55 (OpenVINO), HOTA 29.94 vs 26.50.

| export | boxes match PyTorch? | note |
|---|---|---|
| `nms=False` (default, 384x640) | no | different boxes, more ID switches |
| `nms=False`, 640x640 | no | same problem |
| `nms=False`, dynamic | no | same problem |
| `nms=True` (default conf) | yes, max box difference 0.001 px | but the graph drops every box under 0.25, so low-threshold tracking loses boxes |
| `nms=True, conf=0.01` (used in v2) | yes | full tracking result identical to PyTorch (section 2) |

## 7. Evaluator cross-check

Sources: "Evaluator cross-check" at the end of both v2 reports. One run scored by my code and by PyPI `trackeval` 1.3.0.

| metric | YOLOv8n mine / trackeval | YOLO26n mine / trackeval |
|---|---|---|
| HOTA | 40.250 / 40.250 | 36.675 / 36.675 |
| AssA | 48.545 / 48.545 | 47.965 / 47.964 |
| IDF1 | 47.444 / 47.444 | 40.702 / 40.701 |
| MOTA | 34.102 / 34.104 | 30.386 / 30.383 |
| IDSW | 130 / 130 | 107 / 107 |
| Frag | 284 / 286 | 243 / 244 |

## 8. Earlier runs (kept for the record)

- `results/kaggle_yolov8/`: first YOLOv8n run with thresholds 0.1 to 0.5 and stock ByteTrack. Replaced by v2, which sweeps lower thresholds with `track_low_thresh` = conf.
- `results/kaggle_yolo26/`: first YOLO26n run with the `nms=False` export. Its OpenVINO numbers are affected by the export problem in section 6. Replaced by v2.
