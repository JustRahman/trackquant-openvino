# TrackQuant-OV: yolov8n + ByteTrack on MOT17 (auto-generated)

- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4 vCPU, 31.3 GB RAM, ISA flags: avx2
- Versions: {'ultralytics': '8.4.211', 'torch': '2.11.0+cpu', 'openvino': '2026.4.1', 'nncf': '3.4.0', 'numpy': '2.1.3', 'scipy': '1.16.3', 'lap': '0.5.14', 'trackeval': '1.3.0', 'opencv-python': '4.14.0.94'}
- Data: motchallenge.net (official): https://motchallenge.net/data/MOT17.zip (0.524 GB in 23 s). Test sequences: MOT17-02, MOT17-04, MOT17-11 (FRCNN folder, images + gt only).
- INT8 calibration: 300 frames of MOT17-10 (not a test sequence), NNCF post-training quantization through ultralytics' OpenVINO export.
- Export settings: ultralytics OpenVINO export with its default head (no nms argument); NMS runs in ultralytics' post-processing at the swept conf. Exact arguments: table 'Exported models'.
- Input: 640x384 (imgsz 640, rectangular) for all variants. Tracker: ByteTrack, ultralytics bytetrack.yaml (stock values shown; track_low_thresh is overridden, see next line): {'tracker_type': 'bytetrack', 'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.25, 'track_buffer': 30, 'match_thresh': 0.8, 'fuse_score': True}.
- track_low_thresh: set to the swept conf in every run (--track-low conf), so ByteTrack uses every box the detector threshold lets through.
- The swept threshold is the detector confidence filter applied BEFORE the tracker. ByteTrack's other thresholds stay fixed, so boxes under its track_high_thresh only feed its second (low-score) association.
- Detect-every-N: frames without detection use 'kalman' (kalman = Kalman predict only, predicted boxes keep their IDs; hold = repeat the last boxes).
- Metrics: own implementation of the TrackEval formulas with MOTChallenge pre-processing (class 1, consider-flag 1, distractor matches removed). COMBINED = counts summed over the sequences.
- Timings are comparable only inside this kernel run. pre/inference/post = ultralytics profilers; track = ByteTrack update; JPEG decoding is excluded from fps_e2e.
- Status: complete; 32 grid points done; elapsed 5740 s.

## Exported models

| variant | export s | size MB | export args | INFERENCE_PRECISION_HINT | layers by runtime precision |
|---|---|---|---|---|---|
| PyTorch FP32 | - | 6.25 | PyTorch reference (no export) | - | - |
| OpenVINO FP32 | 3.4 | 12.3 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu'} | <Type: 'float32'> | {'f32': 186} |
| OpenVINO FP16 | 2.5 | 6.33 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'quantize': 16} | <Type: 'float32'> | {'f32': 122, 'f16': 64} |
| OpenVINO INT8 | 58.8 | 3.54 | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'quantize': 8, 'fraction': 1.0} | <Type: 'float32'> | {'f32': 91, 'u8': 96} |

## 1. Tracking quality at each variant's best threshold (detect-every-1, 3 sequences combined)

| variant | best conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDSW_per_min | Frag | fps_e2e |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.03 | 40.298 | 33.576 | 48.755 | 34.718 | 47.403 | 117 | 82.59 | 277 | 10.04 |
| OpenVINO FP32 | 0.03 | 40.298 | 33.576 | 48.755 | 34.718 | 47.403 | 117 | 82.59 | 277 | 13.73 |
| OpenVINO INT8 | 0.01 | 39.706 | 34.198 | 46.512 | 35.297 | 46.857 | 145 | 102.35 | 323 | 19.23 |
| OpenVINO FP16 | 0.01 | 40.477 | 33.682 | 49.051 | 34.071 | 47.636 | 131 | 92.47 | 287 | 13.66 |

Difference to OpenVINO FP32 (each at its own best threshold):

| variant | dHOTA | dDetA | dAssA | dMOTA | dIDF1 | dIDSW/min | speed-up (fps_e2e) |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | 0.73x |
| OpenVINO INT8 | -0.59 | +0.62 | -2.24 | +0.58 | -0.55 | +19.76 | 1.40x |
| OpenVINO FP16 | +0.18 | +0.11 | +0.30 | -0.65 | +0.23 | +9.88 | 0.99x |

## 2. Threshold drift

Full sweep (HOTA / IDF1 per threshold, detect-every-1). Reference = OpenVINO FP32.

| variant | conf 0.01 | conf 0.03 | conf 0.05 | conf 0.1 | conf 0.2 |
|---|---|---|---|---|---|
| PyTorch FP32 | 40.25 / 47.44 | 40.30 / 47.40 **best** | 39.65 / 46.54 | 39.17 / 45.53 | 38.93 / 45.00 |
| OpenVINO FP32 | 40.25 / 47.44 | 40.30 / 47.40 **best** | 39.65 / 46.54 | 39.17 / 45.53 | 38.93 / 45.00 |
| OpenVINO INT8 | 39.71 / 46.86 **best** | 39.60 / 46.69 | 39.15 / 45.92 | 38.96 / 45.42 | 38.83 / 45.37 |
| OpenVINO FP16 | 40.48 / 47.64 **best** | 40.30 / 47.40 | 39.66 / 46.53 | 39.25 / 45.56 | 38.93 / 44.98 |

What happens if OpenVINO FP32's best threshold is reused as-is, versus re-tuning the threshold for the variant:

| variant | ref best conf | own best conf | drifted? | HOTA @ref conf | HOTA @own conf | gain from re-tuning | IDF1 @ref conf | IDF1 @own conf | gain | IDSW/min @ref conf | IDSW/min @own conf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.03 | 0.03 | no | 40.298 | 40.298 | +0.00 | 47.403 | 47.403 | +0.00 | 82.59 | 82.59 |
| OpenVINO INT8 | 0.03 | 0.01 | yes | 39.603 | 39.706 | +0.10 | 46.686 | 46.857 | +0.17 | 103.76 | 102.35 |
| OpenVINO FP16 | 0.03 | 0.01 | yes | 40.301 | 40.477 | +0.18 | 47.404 | 47.636 | +0.23 | 86.12 | 92.47 |

Note: the grid is [0.01, 0.03, 0.05, 0.1, 0.2], so 'no drift' means the optimum did not move to another grid value.

## 3. Detect every N frames (at each variant's best threshold)

| variant | conf | N | HOTA | DetA | AssA | IDF1 | IDSW | IDSW/min | Frag | fps_e2e | fps_wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.03 | 1 | 40.298 | 33.576 | 48.755 | 47.403 | 117 | 82.59 | 277 | 10.04 | 8.89 |
| PyTorch FP32 | 0.03 | 2 | 39.536 | 32.553 | 48.348 | 46.871 | 107 | 75.53 | 268 | 21.27 | 17.25 |
| PyTorch FP32 | 0.03 | 3 | 40.228 | 31.985 | 50.896 | 48.252 | 95 | 67.06 | 244 | 28.93 | 21.64 |
| PyTorch FP32 | 0.03 | 5 | 38.547 | 30.586 | 48.868 | 46.088 | 94 | 66.35 | 246 | 50.08 | 31.61 |
| OpenVINO FP32 | 0.03 | 1 | 40.298 | 33.576 | 48.755 | 47.403 | 117 | 82.59 | 277 | 13.73 | 11.83 |
| OpenVINO FP32 | 0.03 | 2 | 39.536 | 32.553 | 48.348 | 46.871 | 107 | 75.53 | 268 | 27.7 | 21.14 |
| OpenVINO FP32 | 0.03 | 3 | 40.228 | 31.985 | 50.896 | 48.252 | 95 | 67.06 | 244 | 42.73 | 29.74 |
| OpenVINO FP32 | 0.03 | 5 | 38.547 | 30.586 | 48.868 | 46.088 | 94 | 66.35 | 246 | 67.26 | 39.31 |
| OpenVINO INT8 | 0.01 | 1 | 39.706 | 34.198 | 46.512 | 46.857 | 145 | 102.35 | 323 | 19.23 | 15.57 |
| OpenVINO INT8 | 0.01 | 2 | 39.276 | 33.337 | 46.604 | 46.1 | 127 | 89.65 | 298 | 37.86 | 26.14 |
| OpenVINO INT8 | 0.01 | 3 | 38.547 | 32.408 | 46.147 | 45.217 | 105 | 74.12 | 267 | 56.08 | 34.06 |
| OpenVINO INT8 | 0.01 | 5 | 37.292 | 30.789 | 45.435 | 43.816 | 118 | 83.29 | 272 | 91.24 | 44.79 |
| OpenVINO FP16 | 0.01 | 1 | 40.477 | 33.682 | 49.051 | 47.636 | 131 | 92.47 | 287 | 13.66 | 11.79 |
| OpenVINO FP16 | 0.01 | 2 | 39.501 | 32.671 | 48.119 | 46.595 | 114 | 80.47 | 282 | 26.47 | 20.17 |
| OpenVINO FP16 | 0.01 | 3 | 39.945 | 32.107 | 50.013 | 47.486 | 96 | 67.76 | 252 | 39.74 | 27.31 |
| OpenVINO FP16 | 0.01 | 5 | 38.217 | 30.765 | 47.777 | 45.263 | 97 | 68.47 | 252 | 64.5 | 37.66 |

## 4. Speed split (detect-every-1, best threshold; ms per frame, median / p90)

| variant | pre-process | inference | post-process | ByteTrack | fps_e2e | fps_wall (incl. JPEG read) | peak RSS MB |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 2.74 / 3.02 | 90.23 / 96.04 | 1.65 / 1.98 | 3.78 / 6.34 | 10.04 | 8.89 | 815 |
| OpenVINO FP32 | 2.62 / 2.86 | 64.74 / 67.15 | 1.95 / 2.21 | 3.14 / 5.04 | 13.73 | 11.83 | 875 |
| OpenVINO INT8 | 2.67 / 2.95 | 42.53 / 44.67 | 2.11 / 2.40 | 3.87 / 5.60 | 19.23 | 15.57 | 833 |
| OpenVINO FP16 | 2.50 / 2.96 | 63.41 / 70.01 | 1.98 / 2.32 | 3.48 / 5.42 | 13.66 | 11.79 | 846 |

Post-process = box decoding + NMS (YOLOv8).
Post-process time depends on the threshold (lower threshold = more candidate boxes), so compare it at the listed thresholds only.
Peak RSS: all variants run one after another in the same process, so it is a running maximum, not a per-variant footprint.

## 5. Per-sequence table in the style of Intel ENAB #3798 (MOTA FP32 vs INT8), extended with HOTA / IDF1 / IDSW per minute

| sequence | precision | conf | MOTA | HOTA | IDF1 | IDSW | IDSW/min | FPS | model MB |
|---|---|---|---|---|---|---|---|---|---|
| MOT17-02 | PyTorch FP32 | 0.03 | 16.646 | 26.966 | 28.374 | 35 | 105.0 | 10.02 | 6.25 |
| MOT17-02 | OpenVINO FP32 | 0.03 | 16.646 | 26.966 | 28.374 | 35 | 105.0 | 13.58 | 12.3 |
| MOT17-02 | OpenVINO INT8 | 0.01 | 17.335 | 26.293 | 28.331 | 43 | 129.0 | 19.0 | 3.54 |
| MOT17-02 | OpenVINO FP16 | 0.01 | 16.059 | 27.007 | 28.404 | 34 | 102.0 | 14.5 | 6.33 |
| MOT17-04 | PyTorch FP32 | 0.03 | 38.337 | 42.41 | 51.784 | 51 | 87.43 | 9.83 | 6.25 |
| MOT17-04 | OpenVINO FP32 | 0.03 | 38.337 | 42.41 | 51.784 | 51 | 87.43 | 13.34 | 12.3 |
| MOT17-04 | OpenVINO INT8 | 0.01 | 38.907 | 41.63 | 50.768 | 71 | 121.71 | 18.89 | 3.54 |
| MOT17-04 | OpenVINO FP16 | 0.01 | 37.664 | 42.752 | 52.156 | 67 | 114.86 | 13.72 | 6.33 |
| MOT17-11 | PyTorch FP32 | 0.03 | 52.067 | 49.851 | 56.574 | 31 | 62.0 | 10.3 | 6.25 |
| MOT17-11 | OpenVINO FP32 | 0.03 | 52.067 | 49.851 | 56.574 | 31 | 62.0 | 14.32 | 12.3 |
| MOT17-11 | OpenVINO INT8 | 0.01 | 52.469 | 49.945 | 57.116 | 31 | 62.0 | 19.82 | 3.54 |
| MOT17-11 | OpenVINO FP16 | 0.01 | 51.431 | 49.479 | 56.449 | 30 | 60.0 | 13.1 | 6.33 |
| COMBINED | PyTorch FP32 | 0.03 | 34.718 | 40.298 | 47.403 | 117 | 82.59 | 10.04 | 6.25 |
| COMBINED | OpenVINO FP32 | 0.03 | 34.718 | 40.298 | 47.403 | 117 | 82.59 | 13.73 | 12.3 |
| COMBINED | OpenVINO INT8 | 0.01 | 35.297 | 39.706 | 46.857 | 145 | 102.35 | 19.23 | 3.54 |
| COMBINED | OpenVINO FP16 | 0.01 | 34.071 | 40.477 | 47.636 | 131 | 92.47 | 13.66 | 6.33 |

## Evaluator cross-check

One run (pt_fp32 conf=0.01 N=1) scored with this script and with PyPI trackeval 1.3.0:

| metric | this script | trackeval |
|---|---|---|
| HOTA | 40.25 | 40.25 |
| DetA | 33.656 | 33.656 |
| AssA | 48.545 | 48.545 |
| MOTA | 34.102 | 34.104 |
| IDF1 | 47.444 | 47.444 |
| IDSW | 130 | 130 |
| Frag | 284 | 286 |

Plots: p2_hota_vs_threshold.png, p2_detect_every_n.png, p2_speed_split.png. Raw rows: p2_results.csv / p2_results.json.
