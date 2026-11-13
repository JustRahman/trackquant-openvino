# TrackQuant-OV: yolov8n + ByteTrack on MOT17 (auto-generated)

- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4 vCPU, 31.3 GB RAM, ISA flags: avx2
- Versions: {'ultralytics': '8.4.211', 'torch': '2.11.0+cpu', 'openvino': '2026.4.1', 'nncf': '3.4.0', 'numpy': '2.1.3', 'scipy': '1.16.3', 'lap': '0.5.14', 'trackeval': '1.3.0', 'opencv-python': '4.14.0.94'}
- Data: motchallenge.net (official): https://motchallenge.net/data/MOT17.zip (0.524 GB in 22 s). Test sequences: MOT17-02, MOT17-04, MOT17-11 (FRCNN folder, images + gt only).
- INT8 calibration: 300 frames of MOT17-10 (not a test sequence), NNCF post-training quantization through ultralytics' OpenVINO export.
- Input: 640x384 (imgsz 640, rectangular) for all variants. Tracker: ByteTrack, ultralytics bytetrack.yaml unchanged: {'tracker_type': 'bytetrack', 'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.25, 'track_buffer': 30, 'match_thresh': 0.8, 'fuse_score': True}.
- The swept threshold is the detector confidence filter applied BEFORE the tracker. ByteTrack's own thresholds stay fixed, so boxes under its track_high_thresh only feed its second (low-score) association.
- Detect-every-N: frames without detection use 'kalman' (kalman = Kalman predict only, predicted boxes keep their IDs; hold = repeat the last boxes).
- Metrics: own implementation of the TrackEval formulas with MOTChallenge pre-processing (class 1, consider-flag 1, distractor matches removed). COMBINED = counts summed over the sequences.
- Timings are comparable only inside this kernel run. pre/inference/post = ultralytics profilers; track = ByteTrack update; JPEG decoding is excluded from fps_e2e.
- Status: complete; 32 grid points done; elapsed 5494 s.

## Exported models

| variant | export s | size MB | export args | INFERENCE_PRECISION_HINT | layers by runtime precision |
|---|---|---|---|---|---|
| PyTorch FP32 | - | 6.25 | PyTorch reference (no export) | - | - |
| OpenVINO FP32 | 3.2 | 12.3 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu'} | <Type: 'float32'> | {'f32': 186} |
| OpenVINO FP16 | 2.4 | 6.33 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'quantize': 16} | <Type: 'float32'> | {'f32': 122, 'f16': 64} |
| OpenVINO INT8 | 54.0 | 3.54 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'quantize': 8, 'fraction': 1.0} | <Type: 'float32'> | {'f32': 91, 'u8': 96} |

## 1. Tracking quality at each variant's best threshold (detect-every-1, 3 sequences combined)

| variant | best conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDSW_per_min | Frag | fps_e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.1 | 39.171 | 32.459 | 47.585 | 34.083 | 45.531 | 135 | 95.29 | 312 | 10.72 |
| OpenVINO FP32 | 0.1 | 39.171 | 32.459 | 47.585 | 34.083 | 45.531 | 135 | 95.29 | 312 | 14.39 |
| OpenVINO INT8 | 0.1 | 39.109 | 32.79 | 46.964 | 34.64 | 45.391 | 146 | 103.06 | 340 | 20.45 |
| OpenVINO FP16 | 0.1 | 39.248 | 32.45 | 47.782 | 34.086 | 45.562 | 137 | 96.71 | 315 | 14.06 |

Difference to OpenVINO FP32 (each at its own best threshold):

| variant | dHOTA | dDetA | dAssA | dMOTA | dIDF1 | dIDSW/min | speed-up (fps_e2e) |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | 0.74x |
| OpenVINO INT8 | -0.06 | +0.33 | -0.62 | +0.56 | -0.14 | +7.77 | 1.42x |
| OpenVINO FP16 | +0.08 | -0.01 | +0.20 | +0.00 | +0.03 | +1.42 | 0.98x |

## 2. Threshold drift

Full sweep (HOTA / IDF1 per threshold, detect-every-1). Reference = OpenVINO FP32.

| variant | conf 0.1 | conf 0.2 | conf 0.3 | conf 0.4 | conf 0.5 |
|---|---|---|---|---|---|
| PyTorch FP32 | 39.17 / 45.53 **best** | 38.93 / 45.00 | 38.20 / 43.94 | 36.19 / 40.24 | 34.53 / 37.27 |
| OpenVINO FP32 | 39.17 / 45.53 **best** | 38.93 / 45.00 | 38.20 / 43.94 | 36.19 / 40.24 | 34.53 / 37.27 |
| OpenVINO INT8 | 39.11 / 45.39 **best** | 38.78 / 45.26 | 38.15 / 44.18 | 36.63 / 41.47 | 34.84 / 38.10 |
| OpenVINO FP16 | 39.25 / 45.56 **best** | 38.93 / 44.98 | 38.19 / 43.92 | 36.20 / 40.24 | 34.52 / 37.26 |

What happens if OpenVINO FP32's best threshold is reused as-is, versus re-tuning the threshold for the variant:

| variant | ref best conf | own best conf | drifted? | HOTA @ref conf | HOTA @own conf | gain from re-tuning | IDF1 @ref conf | IDF1 @own conf | gain | IDSW/min @ref conf | IDSW/min @own conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.1 | 0.1 | no | 39.171 | 39.171 | +0.00 | 45.531 | 45.531 | +0.00 | 95.29 | 95.29 |
| OpenVINO INT8 | 0.1 | 0.1 | no | 39.109 | 39.109 | +0.00 | 45.391 | 45.391 | +0.00 | 103.06 | 103.06 |
| OpenVINO FP16 | 0.1 | 0.1 | no | 39.248 | 39.248 | +0.00 | 45.562 | 45.562 | +0.00 | 96.71 | 96.71 |

Note: the grid step is 0.1, so 'no drift' means the optimum moved by less than one grid step.

## 3. Detect every N frames (at each variant's best threshold)

| variant | conf | N | HOTA | DetA | AssA | IDF1 | IDSW | IDSW/min | Frag | fps_e2e | fps_wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.1 | 1 | 39.171 | 32.459 | 47.585 | 45.531 | 135 | 95.29 | 312 | 10.72 | 9.56 |
| PyTorch FP32 | 0.1 | 2 | 39.172 | 31.664 | 48.765 | 46.22 | 111 | 78.35 | 284 | 20.91 | 17.07 |
| PyTorch FP32 | 0.1 | 3 | 39.698 | 31.191 | 50.797 | 47.618 | 91 | 64.24 | 256 | 31.03 | 23.25 |
| PyTorch FP32 | 0.1 | 5 | 37.626 | 29.8 | 47.775 | 44.852 | 97 | 68.47 | 245 | 53.02 | 34.41 |
| OpenVINO FP32 | 0.1 | 1 | 39.171 | 32.459 | 47.585 | 45.531 | 135 | 95.29 | 312 | 14.39 | 12.47 |
| OpenVINO FP32 | 0.1 | 2 | 39.172 | 31.664 | 48.765 | 46.22 | 111 | 78.35 | 284 | 28.18 | 21.67 |
| OpenVINO FP32 | 0.1 | 3 | 39.698 | 31.191 | 50.797 | 47.618 | 91 | 64.24 | 256 | 41.23 | 28.91 |
| OpenVINO FP32 | 0.1 | 5 | 37.626 | 29.8 | 47.775 | 44.852 | 97 | 68.47 | 245 | 68.17 | 40.14 |
| OpenVINO INT8 | 0.1 | 1 | 39.109 | 32.79 | 46.964 | 45.391 | 146 | 103.06 | 340 | 20.45 | 16.75 |
| OpenVINO INT8 | 0.1 | 2 | 38.938 | 32.193 | 47.409 | 46.004 | 118 | 83.29 | 302 | 38.09 | 26.38 |
| OpenVINO INT8 | 0.1 | 3 | 38.989 | 31.364 | 48.747 | 46.449 | 104 | 73.41 | 277 | 59.64 | 36.92 |
| OpenVINO INT8 | 0.1 | 5 | 36.979 | 29.886 | 46.003 | 43.328 | 112 | 79.06 | 268 | 96.99 | 47.52 |
| OpenVINO FP16 | 0.1 | 1 | 39.248 | 32.45 | 47.782 | 45.562 | 137 | 96.71 | 315 | 14.06 | 12.21 |
| OpenVINO FP16 | 0.1 | 2 | 39.247 | 31.65 | 48.97 | 46.374 | 108 | 76.24 | 283 | 28.31 | 21.75 |
| OpenVINO FP16 | 0.1 | 3 | 39.696 | 31.193 | 50.792 | 47.609 | 91 | 64.24 | 257 | 42.05 | 29.38 |
| OpenVINO FP16 | 0.1 | 5 | 37.577 | 29.831 | 47.601 | 44.731 | 97 | 68.47 | 244 | 69.17 | 40.74 |

## 4. Speed split (detect-every-1, best threshold; ms per frame, median / p90)

| variant | pre-process | inference | post-process | ByteTrack | fps_e2e | fps_wall (incl. JPEG read) | peak RSS MB |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 2.48 / 2.67 | 85.63 / 89.67 | 1.32 / 1.52 | 2.93 / 4.95 | 10.72 | 9.56 | 773 |
| OpenVINO FP32 | 2.26 / 2.45 | 61.99 / 64.23 | 1.61 / 1.81 | 2.64 / 4.18 | 14.39 | 12.47 | 868 |
| OpenVINO INT8 | 2.31 / 2.53 | 41.35 / 43.02 | 1.62 / 1.86 | 2.68 / 4.27 | 20.45 | 16.75 | 824 |
| OpenVINO FP16 | 2.31 / 2.54 | 62.89 / 67.36 | 1.63 / 1.86 | 2.68 / 4.18 | 14.06 | 12.21 | 833 |

Post-process = box decoding + NMS (YOLOv8).
Post-process time depends on the threshold (lower threshold = more candidate boxes), so compare it at the listed thresholds only.
Peak RSS: all variants run one after another in the same process, so it is a running maximum, not a per-variant footprint.

## 5. Per-sequence table in the style of Intel ENAB #3798 (MOTA FP32 vs INT8), extended with HOTA / IDF1 / IDSW per minute

| sequence | precision | conf | MOTA | HOTA | IDF1 | IDSW | IDSW/min | FPS | model MB |
|---|---|---|---|---|---|---|---|---|---|
| MOT17-02 | PyTorch FP32 | 0.1 | 16.969 | 26.887 | 28.07 | 36 | 108.0 | 10.87 | 6.25 |
| MOT17-02 | OpenVINO FP32 | 0.1 | 16.969 | 26.887 | 28.07 | 36 | 108.0 | 14.51 | 12.3 |
| MOT17-02 | OpenVINO INT8 | 0.1 | 17.62 | 25.943 | 27.615 | 43 | 129.0 | 20.73 | 3.54 |
| MOT17-02 | OpenVINO FP16 | 0.1 | 16.958 | 26.89 | 28.07 | 36 | 108.0 | 14.53 | 6.33 |
| MOT17-04 | PyTorch FP32 | 0.1 | 37.044 | 41.379 | 50.138 | 66 | 113.14 | 10.65 | 6.25 |
| MOT17-04 | OpenVINO FP32 | 0.1 | 37.044 | 41.379 | 50.138 | 66 | 113.14 | 14.23 | 12.3 |
| MOT17-04 | OpenVINO INT8 | 0.1 | 37.702 | 41.511 | 50.074 | 67 | 114.86 | 19.95 | 3.54 |
| MOT17-04 | OpenVINO FP16 | 0.1 | 37.065 | 41.497 | 50.189 | 68 | 116.57 | 14.06 | 6.33 |
| MOT17-11 | PyTorch FP32 | 0.1 | 52.861 | 47.125 | 51.572 | 33 | 66.0 | 10.7 | 6.25 |
| MOT17-11 | OpenVINO FP32 | 0.1 | 52.861 | 47.125 | 51.572 | 33 | 66.0 | 14.5 | 12.3 |
| MOT17-11 | OpenVINO INT8 | 0.1 | 52.724 | 47.095 | 51.434 | 36 | 72.0 | 20.88 | 3.54 |
| MOT17-11 | OpenVINO FP16 | 0.1 | 52.798 | 47.112 | 51.559 | 33 | 66.0 | 13.78 | 6.33 |
| COMBINED | PyTorch FP32 | 0.1 | 34.083 | 39.171 | 45.531 | 135 | 95.29 | 10.72 | 6.25 |
| COMBINED | OpenVINO FP32 | 0.1 | 34.083 | 39.171 | 45.531 | 135 | 95.29 | 14.39 | 12.3 |
| COMBINED | OpenVINO INT8 | 0.1 | 34.64 | 39.109 | 45.391 | 146 | 103.06 | 20.45 | 3.54 |
| COMBINED | OpenVINO FP16 | 0.1 | 34.086 | 39.248 | 45.562 | 137 | 96.71 | 14.06 | 6.33 |

## Evaluator cross-check

One run (pt_fp32 conf=0.1 N=1) scored with this script and with PyPI trackeval 1.3.0:

| metric | this script | trackeval |
|---|---|---|
| HOTA | 39.171 | 39.171 |
| DetA | 32.459 | 32.459 |
| AssA | 47.585 | 47.585 |
| MOTA | 34.083 | 34.083 |
| IDF1 | 45.531 | 45.531 |
| IDSW | 135 | 135 |
| Frag | 312 | 313 |

Plots: p2_hota_vs_threshold.png, p2_detect_every_n.png, p2_speed_split.png. Raw rows: p2_results.csv / p2_results.json.
