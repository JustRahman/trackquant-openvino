# TrackQuant-OV: yolo26n + ByteTrack on MOT17 (auto-generated)

- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4 vCPU, 31.3 GB RAM, ISA flags: avx2
- Versions: {'ultralytics': '8.4.211', 'torch': '2.11.0+cpu', 'openvino': '2026.4.1', 'nncf': '3.4.0', 'numpy': '2.1.3', 'scipy': '1.16.3', 'lap': '0.5.14', 'trackeval': '1.3.0', 'opencv-python': '4.14.0.94'}
- Data: motchallenge.net (official): https://motchallenge.net/data/MOT17.zip (0.524 GB in 16 s). Test sequences: MOT17-02, MOT17-04, MOT17-11 (FRCNN folder, images + gt only).
- INT8 calibration: 300 frames of MOT17-10 (not a test sequence), NNCF post-training quantization through ultralytics' OpenVINO export.
- Export settings: yolo26 OpenVINO models exported with nms=True and conf=0.01 (--yolo26-export-nms true): the graph contains an NMS whose built-in score filter is that conf (the lowest swept threshold); the swept conf is applied on top at predict time. PyTorch FP32 runs the stock NMS-free head. Exact arguments: table 'Exported models'.
- Input: 640x384 (imgsz 640, rectangular) for all variants. Tracker: ByteTrack, ultralytics bytetrack.yaml (stock values shown; track_low_thresh is overridden, see next line): {'tracker_type': 'bytetrack', 'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.25, 'track_buffer': 30, 'match_thresh': 0.8, 'fuse_score': True}.
- track_low_thresh: set to the swept conf in every run (--track-low conf), so ByteTrack uses every box the detector threshold lets through.
- The swept threshold is the detector confidence filter applied BEFORE the tracker. ByteTrack's other thresholds stay fixed, so boxes under its track_high_thresh only feed its second (low-score) association.
- Detect-every-N: frames without detection use 'kalman' (kalman = Kalman predict only, predicted boxes keep their IDs; hold = repeat the last boxes).
- Metrics: own implementation of the TrackEval formulas with MOTChallenge pre-processing (class 1, consider-flag 1, distractor matches removed). COMBINED = counts summed over the sequences.
- Timings are comparable only inside this kernel run. pre/inference/post = ultralytics profilers; track = ByteTrack update; JPEG decoding is excluded from fps_e2e.
- Status: complete; 32 grid points done; elapsed 4090 s.

## Exported models

| variant | export s | size MB | export args | INFERENCE_PRECISION_HINT | layers by runtime precision |
|---|---|---|---|---|---|
| PyTorch FP32 | - | 5.29 | PyTorch reference (no export) | - | - |
| OpenVINO FP32 | 11.3 | 9.64 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': True, 'conf': 0.01} | <Type: 'float32'> | {'f32': 321, 'u8': 2, 'i32': 23} |
| OpenVINO FP16 | 4.0 | 5.14 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': True, 'conf': 0.01, 'quantize': 16} | <Type: 'float32'> | {'f32': 219, 'f16': 102, 'u8': 2, 'i32': 23} |
| OpenVINO INT8 | 52.5 | 3.21 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': True, 'conf': 0.01, 'quantize': 8, 'fraction': 1.0} | <Type: 'float32'> | {'f32': 174, 'u8': 146, 'i32': 23} |

## 1. Tracking quality at each variant's best threshold (detect-every-1, 3 sequences combined)

| variant | best conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDSW_per_min | Frag | fps_e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 36.675 | 28.215 | 47.965 | 30.386 | 40.702 | 107 | 75.53 | 243 | 13.91 |
| OpenVINO FP32 | 0.01 | 36.675 | 28.215 | 47.965 | 30.386 | 40.702 | 107 | 75.53 | 243 | 21.72 |
| OpenVINO INT8 | 0.03 | 36.384 | 28.31 | 47.014 | 30.852 | 40.821 | 109 | 76.94 | 271 | 28.64 |
| OpenVINO FP16 | 0.01 | 36.668 | 28.223 | 47.931 | 30.422 | 40.708 | 109 | 76.94 | 243 | 22.18 |

Difference to OpenVINO FP32 (each at its own best threshold):

| variant | dHOTA | dDetA | dAssA | dMOTA | dIDF1 | dIDSW/min | speed-up (fps_e2e) |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | 0.64x |
| OpenVINO INT8 | -0.29 | +0.09 | -0.95 | +0.47 | +0.12 | +1.41 | 1.32x |
| OpenVINO FP16 | -0.01 | +0.01 | -0.03 | +0.04 | +0.01 | +1.41 | 1.02x |

## 2. Threshold drift

Full sweep (HOTA / IDF1 per threshold, detect-every-1). Reference = OpenVINO FP32.

| variant | conf 0.01 | conf 0.03 | conf 0.05 | conf 0.1 | conf 0.2 |
|---|---|---|---|---|---|
| PyTorch FP32 | 36.67 / 40.70 **best** | 35.59 / 39.25 | 35.05 / 38.17 | 34.66 / 37.49 | 33.93 / 36.16 |
| OpenVINO FP32 | 36.67 / 40.70 **best** | 35.59 / 39.25 | 35.05 / 38.17 | 34.66 / 37.49 | 33.93 / 36.16 |
| OpenVINO INT8 | 35.12 / 38.90 | 36.38 / 40.82 **best** | 35.34 / 39.39 | 34.67 / 37.95 | 34.44 / 37.67 |
| OpenVINO FP16 | 36.67 / 40.71 **best** | 35.59 / 39.25 | 35.04 / 38.15 | 34.65 / 37.47 | 33.92 / 36.16 |

What happens if OpenVINO FP32's best threshold is reused as-is, versus re-tuning the threshold for the variant:

| variant | ref best conf | own best conf | drifted? | HOTA @ref conf | HOTA @own conf | gain from re-tuning | IDF1 @ref conf | IDF1 @own conf | gain | IDSW/min @ref conf | IDSW/min @own conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 0.01 | no | 36.675 | 36.675 | +0.00 | 40.702 | 40.702 | +0.00 | 75.53 | 75.53 |
| OpenVINO INT8 | 0.01 | 0.03 | yes | 35.121 | 36.384 | +1.26 | 38.904 | 40.821 | +1.92 | 87.53 | 76.94 |
| OpenVINO FP16 | 0.01 | 0.01 | no | 36.668 | 36.668 | +0.00 | 40.708 | 40.708 | +0.00 | 76.94 | 76.94 |

Note: the grid is [0.01, 0.03, 0.05, 0.1, 0.2], so 'no drift' means the optimum did not move to another grid value.

## 3. Detect every N frames (at each variant's best threshold)

| variant | conf | N | HOTA | DetA | AssA | IDF1 | IDSW | IDSW/min | Frag | fps_e2e | fps_wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 1 | 36.675 | 28.215 | 47.965 | 40.702 | 107 | 75.53 | 243 | 13.91 | 12.19 |
| PyTorch FP32 | 0.01 | 2 | 35.295 | 27.256 | 46.014 | 39.554 | 94 | 66.35 | 221 | 26.91 | 21.27 |
| PyTorch FP32 | 0.01 | 3 | 35.811 | 26.439 | 48.785 | 40.162 | 76 | 53.65 | 210 | 39.37 | 28.49 |
| PyTorch FP32 | 0.01 | 5 | 33.336 | 24.64 | 45.377 | 37.512 | 77 | 54.35 | 230 | 65.73 | 38.63 |
| OpenVINO FP32 | 0.01 | 1 | 36.675 | 28.215 | 47.965 | 40.702 | 107 | 75.53 | 243 | 21.72 | 17.83 |
| OpenVINO FP32 | 0.01 | 2 | 35.295 | 27.256 | 46.014 | 39.554 | 94 | 66.35 | 221 | 42.95 | 30.27 |
| OpenVINO FP32 | 0.01 | 3 | 35.811 | 26.439 | 48.785 | 40.162 | 76 | 53.65 | 210 | 63.82 | 39.49 |
| OpenVINO FP32 | 0.01 | 5 | 33.336 | 24.64 | 45.377 | 37.512 | 77 | 54.35 | 230 | 103.89 | 50.73 |
| OpenVINO INT8 | 0.03 | 1 | 36.384 | 28.31 | 47.014 | 40.821 | 109 | 76.94 | 271 | 28.64 | 22.26 |
| OpenVINO INT8 | 0.03 | 2 | 35.124 | 26.832 | 46.271 | 39.458 | 95 | 67.06 | 250 | 55.85 | 36.05 |
| OpenVINO INT8 | 0.03 | 3 | 35.235 | 25.782 | 48.37 | 40.218 | 75 | 52.94 | 216 | 83.73 | 46.41 |
| OpenVINO INT8 | 0.03 | 5 | 32.859 | 24.524 | 44.251 | 37.197 | 85 | 60.0 | 218 | 137.39 | 57.31 |
| OpenVINO FP16 | 0.01 | 1 | 36.668 | 28.223 | 47.931 | 40.708 | 109 | 76.94 | 243 | 22.18 | 18.16 |
| OpenVINO FP16 | 0.01 | 2 | 35.288 | 27.206 | 46.076 | 39.391 | 95 | 67.06 | 218 | 42.6 | 30.06 |
| OpenVINO FP16 | 0.01 | 3 | 35.849 | 26.434 | 48.897 | 40.334 | 75 | 52.94 | 211 | 61.31 | 38.38 |
| OpenVINO FP16 | 0.01 | 5 | 33.318 | 24.672 | 45.275 | 37.445 | 77 | 54.35 | 232 | 104.23 | 50.94 |

## 4. Speed split (detect-every-1, best threshold; ms per frame, median / p90)

| variant | pre-process | inference | post-process | ByteTrack | fps_e2e | fps_wall (incl. JPEG read) | peak RSS MB |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 2.25 / 2.39 | 65.19 / 69.12 | 1.31 / 1.59 | 3.08 / 4.11 | 13.91 | 12.19 | 847 |
| OpenVINO FP32 | 2.34 / 2.68 | 39.84 / 42.35 | 0.48 / 0.58 | 2.92 / 3.57 | 21.72 | 17.83 | 932 |
| OpenVINO INT8 | 2.25 / 2.62 | 29.20 / 31.15 | 0.47 / 0.51 | 2.56 / 3.07 | 28.64 | 22.26 | 910 |
| OpenVINO FP16 | 2.43 / 2.64 | 38.82 / 41.68 | 0.47 / 0.55 | 2.86 / 3.30 | 22.18 | 18.16 | 905 |

Post-process = box decoding only for PyTorch (NMS-free head); the OpenVINO graphs were exported with nms=True, so their NMS is counted as inference.
Post-process time depends on the threshold (lower threshold = more candidate boxes), so compare it at the listed thresholds only.
Peak RSS: all variants run one after another in the same process, so it is a running maximum, not a per-variant footprint.

## 5. Per-sequence table in the style of Intel ENAB #3798 (MOTA FP32 vs INT8), extended with HOTA / IDF1 / IDSW per minute

| sequence | precision | conf | MOTA | HOTA | IDF1 | IDSW | IDSW/min | FPS | model MB |
|---|---|---|---|---|---|---|---|---|---|
| MOT17-02 | PyTorch FP32 | 0.01 | 21.818 | 29.678 | 33.531 | 35 | 105.0 | 13.73 | 5.29 |
| MOT17-02 | OpenVINO FP32 | 0.01 | 21.818 | 29.678 | 33.531 | 35 | 105.0 | 21.59 | 9.64 |
| MOT17-02 | OpenVINO INT8 | 0.03 | 21.134 | 29.525 | 33.388 | 29 | 87.0 | 28.65 | 3.21 |
| MOT17-02 | OpenVINO FP16 | 0.01 | 21.77 | 29.646 | 33.456 | 36 | 108.0 | 22.52 | 5.14 |
| MOT17-04 | PyTorch FP32 | 0.01 | 29.22 | 36.439 | 40.274 | 40 | 68.57 | 13.67 | 5.29 |
| MOT17-04 | OpenVINO FP32 | 0.01 | 29.22 | 36.438 | 40.274 | 40 | 68.57 | 21.54 | 9.64 |
| MOT17-04 | OpenVINO INT8 | 0.03 | 30.164 | 36.062 | 40.314 | 47 | 80.57 | 28.2 | 3.21 |
| MOT17-04 | OpenVINO FP16 | 0.01 | 29.314 | 36.451 | 40.339 | 40 | 68.57 | 22.36 | 5.14 |
| MOT17-11 | PyTorch FP32 | 0.01 | 53.137 | 47.918 | 53.194 | 32 | 64.0 | 14.34 | 5.29 |
| MOT17-11 | OpenVINO FP32 | 0.01 | 53.137 | 47.918 | 53.194 | 32 | 64.0 | 22.02 | 9.64 |
| MOT17-11 | OpenVINO INT8 | 0.03 | 53.455 | 47.817 | 54.043 | 33 | 66.0 | 29.17 | 3.21 |
| MOT17-11 | OpenVINO FP16 | 0.01 | 53.042 | 47.873 | 53.087 | 33 | 66.0 | 21.77 | 5.14 |
| COMBINED | PyTorch FP32 | 0.01 | 30.386 | 36.675 | 40.702 | 107 | 75.53 | 13.91 | 5.29 |
| COMBINED | OpenVINO FP32 | 0.01 | 30.386 | 36.675 | 40.702 | 107 | 75.53 | 21.72 | 9.64 |
| COMBINED | OpenVINO INT8 | 0.03 | 30.852 | 36.384 | 40.821 | 109 | 76.94 | 28.64 | 3.21 |
| COMBINED | OpenVINO FP16 | 0.01 | 30.422 | 36.668 | 40.708 | 109 | 76.94 | 22.18 | 5.14 |

## Evaluator cross-check

One run (pt_fp32 conf=0.01 N=1) scored with this script and with PyPI trackeval 1.3.0:

| metric | this script | trackeval |
|---|---|---|
| HOTA | 36.675 | 36.675 |
| DetA | 28.215 | 28.215 |
| AssA | 47.965 | 47.964 |
| MOTA | 30.386 | 30.383 |
| IDF1 | 40.702 | 40.701 |
| IDSW | 107 | 107 |
| Frag | 243 | 244 |

Plots: p2_hota_vs_threshold.png, p2_detect_every_n.png, p2_speed_split.png. Raw rows: p2_results.csv / p2_results.json.
