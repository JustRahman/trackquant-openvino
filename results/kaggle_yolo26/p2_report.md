# TrackQuant-OV: yolo26n + ByteTrack on MOT17 (auto-generated)

- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4 vCPU, 31.3 GB RAM, ISA flags: avx2
- Versions: {'ultralytics': '8.4.211', 'torch': '2.11.0+cpu', 'openvino': '2026.4.1', 'nncf': '3.4.0', 'numpy': '2.1.3', 'scipy': '1.16.3', 'lap': '0.5.14', 'trackeval': '1.3.0', 'opencv-python': '4.14.0.94'}
- Data: motchallenge.net (official): https://motchallenge.net/data/MOT17.zip (0.524 GB in 11 s). Test sequences: MOT17-02, MOT17-04, MOT17-11 (FRCNN folder, images + gt only).
- INT8 calibration: 300 frames of MOT17-10 (not a test sequence), NNCF post-training quantization through ultralytics' OpenVINO export.
- Input: 640x384 (imgsz 640, rectangular) for all variants. Tracker: ByteTrack, ultralytics bytetrack.yaml unchanged: {'tracker_type': 'bytetrack', 'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.25, 'track_buffer': 30, 'match_thresh': 0.8, 'fuse_score': True}.
- The swept threshold is the detector confidence filter applied BEFORE the tracker. ByteTrack's own thresholds stay fixed, so boxes under its track_high_thresh only feed its second (low-score) association.
- Detect-every-N: frames without detection use 'kalman' (kalman = Kalman predict only, predicted boxes keep their IDs; hold = repeat the last boxes).
- Metrics: own implementation of the TrackEval formulas with MOTChallenge pre-processing (class 1, consider-flag 1, distractor matches removed). COMBINED = counts summed over the sequences.
- Timings are comparable only inside this kernel run. pre/inference/post = ultralytics profilers; track = ByteTrack update; JPEG decoding is excluded from fps_e2e.
- Status: complete; 32 grid points done; elapsed 4792 s.

## Exported models

| variant | export s | size MB | export args | INFERENCE_PRECISION_HINT | layers by runtime precision |
|---|---|---|---|---|---|
| PyTorch FP32 | - | 5.29 | PyTorch reference (no export) | - | - |
| OpenVINO FP32 | 4.1 | 9.6 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': False} | <Type: 'float32'> | {'f32': 306, 'i32': 10, 'u8': 2} |
| OpenVINO FP16 | 3.7 | 5.09 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': False, 'quantize': 16} | <Type: 'float32'> | {'f32': 204, 'f16': 102, 'i32': 10, 'u8': 2} |
| OpenVINO INT8 | 53.9 | 3.16 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': False, 'quantize': 8, 'fraction': 1.0} | <Type: 'float32'> | {'f32': 160, 'u8': 146, 'i32': 10} |

## 1. Tracking quality at each variant's best threshold (detect-every-1, 3 sequences combined)

| variant | best conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDSW_per_min | Frag | fps_e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.1 | 34.663 | 26.337 | 45.91 | 28.546 | 37.49 | 107 | 75.53 | 253 | 11.77 |
| OpenVINO FP32 | 0.1 | 33.774 | 24.977 | 45.976 | 26.514 | 36.725 | 132 | 93.18 | 330 | 18.68 |
| OpenVINO INT8 | 0.1 | 33.485 | 24.62 | 45.755 | 26.557 | 37.14 | 130 | 91.76 | 400 | 24.92 |
| OpenVINO FP16 | 0.1 | 33.781 | 24.962 | 46.02 | 26.512 | 36.741 | 132 | 93.18 | 328 | 18.69 |

Difference to OpenVINO FP32 (each at its own best threshold):

| variant | dHOTA | dDetA | dAssA | dMOTA | dIDF1 | dIDSW/min | speed-up (fps_e2e) |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | +0.89 | +1.36 | -0.07 | +2.03 | +0.77 | -17.65 | 0.63x |
| OpenVINO INT8 | -0.29 | -0.36 | -0.22 | +0.04 | +0.41 | -1.42 | 1.33x |
| OpenVINO FP16 | +0.01 | -0.02 | +0.04 | -0.00 | +0.02 | +0.00 | 1.00x |

## 2. Threshold drift

Full sweep (HOTA / IDF1 per threshold, detect-every-1). Reference = OpenVINO FP32.

| variant | conf 0.1 | conf 0.2 | conf 0.3 | conf 0.4 | conf 0.5 |
|---|---|---|---|---|---|
| PyTorch FP32 | 34.66 / 37.49 **best** | 33.93 / 36.16 | 33.27 / 35.08 | 31.87 / 32.51 | 29.36 / 28.20 |
| OpenVINO FP32 | 33.77 / 36.73 **best** | 32.36 / 34.10 | 31.74 / 32.96 | 30.68 / 30.79 | 29.63 / 28.90 |
| OpenVINO INT8 | 33.48 / 37.14 **best** | 32.43 / 35.20 | 31.09 / 32.84 | 29.73 / 30.22 | 28.58 / 28.50 |
| OpenVINO FP16 | 33.78 / 36.74 **best** | 32.37 / 34.12 | 31.73 / 32.95 | 30.68 / 30.79 | 29.57 / 28.83 |

What happens if OpenVINO FP32's best threshold is reused as-is, versus re-tuning the threshold for the variant:

| variant | ref best conf | own best conf | drifted? | HOTA @ref conf | HOTA @own conf | gain from re-tuning | IDF1 @ref conf | IDF1 @own conf | gain | IDSW/min @ref conf | IDSW/min @own conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.1 | 0.1 | no | 34.663 | 34.663 | +0.00 | 37.49 | 37.49 | +0.00 | 75.53 | 75.53 |
| OpenVINO INT8 | 0.1 | 0.1 | no | 33.485 | 33.485 | +0.00 | 37.14 | 37.14 | +0.00 | 91.76 | 91.76 |
| OpenVINO FP16 | 0.1 | 0.1 | no | 33.781 | 33.781 | +0.00 | 36.741 | 36.741 | +0.00 | 93.18 | 93.18 |

Note: the grid step is 0.1, so 'no drift' means the optimum moved by less than one grid step.

## 3. Detect every N frames (at each variant's best threshold)

| variant | conf | N | HOTA | DetA | AssA | IDF1 | IDSW | IDSW/min | Frag | fps_e2e | fps_wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.1 | 1 | 34.663 | 26.337 | 45.91 | 37.49 | 107 | 75.53 | 253 | 11.77 | 10.28 |
| PyTorch FP32 | 0.1 | 2 | 34.124 | 25.855 | 45.32 | 37.707 | 91 | 64.24 | 231 | 21.61 | 17.03 |
| PyTorch FP32 | 0.1 | 3 | 34.451 | 25.231 | 47.292 | 38.315 | 73 | 51.53 | 215 | 32.35 | 22.38 |
| PyTorch FP32 | 0.1 | 5 | 33.097 | 23.633 | 46.602 | 37.384 | 75 | 52.94 | 224 | 56.53 | 33.58 |
| OpenVINO FP32 | 0.1 | 1 | 33.774 | 24.977 | 45.976 | 36.725 | 132 | 93.18 | 330 | 18.68 | 15.29 |
| OpenVINO FP32 | 0.1 | 2 | 33.484 | 24.223 | 46.59 | 36.509 | 106 | 74.82 | 262 | 36.77 | 25.39 |
| OpenVINO FP32 | 0.1 | 3 | 32.393 | 23.166 | 45.575 | 35.074 | 86 | 60.71 | 221 | 55.49 | 33.85 |
| OpenVINO FP32 | 0.1 | 5 | 31.619 | 22.069 | 45.625 | 34.103 | 82 | 57.88 | 224 | 91.46 | 44.69 |
| OpenVINO INT8 | 0.1 | 1 | 33.485 | 24.62 | 45.755 | 37.14 | 130 | 91.76 | 400 | 24.92 | 19.16 |
| OpenVINO INT8 | 0.1 | 2 | 32.163 | 23.78 | 43.693 | 35.132 | 100 | 70.59 | 278 | 49.58 | 30.34 |
| OpenVINO INT8 | 0.1 | 3 | 32.228 | 23.24 | 44.854 | 35.75 | 88 | 62.12 | 249 | 73.6 | 39.85 |
| OpenVINO INT8 | 0.1 | 5 | 31.358 | 21.758 | 45.385 | 34.854 | 84 | 59.29 | 224 | 121.46 | 50.88 |
| OpenVINO FP16 | 0.1 | 1 | 33.781 | 24.962 | 46.02 | 36.741 | 132 | 93.18 | 328 | 18.69 | 15.28 |
| OpenVINO FP16 | 0.1 | 2 | 33.499 | 24.229 | 46.62 | 36.541 | 105 | 74.12 | 262 | 37.21 | 25.72 |
| OpenVINO FP16 | 0.1 | 3 | 32.41 | 23.167 | 45.619 | 35.086 | 86 | 60.71 | 220 | 55.55 | 33.86 |
| OpenVINO FP16 | 0.1 | 5 | 31.632 | 22.088 | 45.622 | 34.125 | 82 | 57.88 | 221 | 90.23 | 44.62 |

## 4. Speed split (detect-every-1, best threshold; ms per frame, median / p90)

| variant | pre-process | inference | post-process | ByteTrack | fps_e2e | fps_wall (incl. JPEG read) | peak RSS MB |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 2.48 / 2.68 | 78.61 / 83.32 | 1.24 / 1.36 | 2.53 / 3.63 | 11.77 | 10.28 | 823 |
| OpenVINO FP32 | 2.31 / 3.01 | 47.62 / 49.23 | 0.50 / 0.54 | 2.39 / 3.02 | 18.68 | 15.29 | 913 |
| OpenVINO INT8 | 2.85 / 3.12 | 34.24 / 35.44 | 0.50 / 0.54 | 2.37 / 3.00 | 24.92 | 19.16 | 871 |
| OpenVINO FP16 | 2.86 / 3.19 | 47.48 / 49.05 | 0.50 / 0.53 | 2.38 / 2.98 | 18.69 | 15.28 | 891 |

Post-process = box decoding only: the YOLO26 head is NMS-free (end-to-end).
Post-process time depends on the threshold (lower threshold = more candidate boxes), so compare it at the listed thresholds only.
Peak RSS: all variants run one after another in the same process, so it is a running maximum, not a per-variant footprint.

## 5. Per-sequence table in the style of Intel ENAB #3798 (MOTA FP32 vs INT8), extended with HOTA / IDF1 / IDSW per minute

| sequence | precision | conf | MOTA | HOTA | IDF1 | IDSW | IDSW/min | FPS | model MB |
|---|---|---|---|---|---|---|---|---|---|
| MOT17-02 | PyTorch FP32 | 0.1 | 21.204 | 29.943 | 32.658 | 32 | 96.0 | 11.86 | 5.29 |
| MOT17-02 | OpenVINO FP32 | 0.1 | 19.611 | 26.495 | 27.883 | 55 | 165.0 | 18.75 | 9.6 |
| MOT17-02 | OpenVINO INT8 | 0.1 | 19.972 | 26.451 | 28.209 | 49 | 147.0 | 25.11 | 3.16 |
| MOT17-02 | OpenVINO FP16 | 0.1 | 19.687 | 26.525 | 27.883 | 55 | 165.0 | 18.68 | 5.09 |
| MOT17-04 | PyTorch FP32 | 0.1 | 26.368 | 32.843 | 35.088 | 44 | 75.43 | 11.63 | 5.29 |
| MOT17-04 | OpenVINO FP32 | 0.1 | 24.131 | 32.819 | 35.816 | 44 | 75.43 | 18.56 | 9.6 |
| MOT17-04 | OpenVINO INT8 | 0.1 | 24.104 | 31.863 | 35.374 | 42 | 72.0 | 24.71 | 3.16 |
| MOT17-04 | OpenVINO FP16 | 0.1 | 24.081 | 32.817 | 35.834 | 44 | 75.43 | 18.63 | 5.09 |
| MOT17-11 | PyTorch FP32 | 0.1 | 53.974 | 48.545 | 54.092 | 31 | 62.0 | 11.88 | 5.29 |
| MOT17-11 | OpenVINO FP32 | 0.1 | 52.12 | 47.572 | 53.362 | 33 | 66.0 | 18.79 | 9.6 |
| MOT17-11 | OpenVINO INT8 | 0.1 | 51.886 | 49.146 | 57.25 | 39 | 78.0 | 25.03 | 3.16 |
| MOT17-11 | OpenVINO FP16 | 0.1 | 52.204 | 47.597 | 53.402 | 33 | 66.0 | 18.78 | 5.09 |
| COMBINED | PyTorch FP32 | 0.1 | 28.546 | 34.663 | 37.49 | 107 | 75.53 | 11.77 | 5.29 |
| COMBINED | OpenVINO FP32 | 0.1 | 26.514 | 33.774 | 36.725 | 132 | 93.18 | 18.68 | 9.6 |
| COMBINED | OpenVINO INT8 | 0.1 | 26.557 | 33.485 | 37.14 | 130 | 91.76 | 24.92 | 3.16 |
| COMBINED | OpenVINO FP16 | 0.1 | 26.512 | 33.781 | 36.741 | 132 | 93.18 | 18.69 | 5.09 |

## Evaluator cross-check

One run (pt_fp32 conf=0.1 N=1) scored with this script and with PyPI trackeval 1.3.0:

| metric | this script | trackeval |
|---|---|---|
| HOTA | 34.663 | 34.663 |
| DetA | 26.337 | 26.336 |
| AssA | 45.91 | 45.91 |
| MOTA | 28.546 | 28.546 |
| IDF1 | 37.49 | 37.49 |
| IDSW | 107 | 107 |
| Frag | 253 | 253 |

Plots: p2_hota_vs_threshold.png, p2_detect_every_n.png, p2_speed_split.png. Raw rows: p2_results.csv / p2_results.json.
