# TrackQuant-OV follow-up: yolo26n + ByteTrack on MOT17 (auto-generated)

- CPU: Intel(R) Xeon(R) CPU @ 2.20GHz x4 vCPU, 31.3 GB RAM, ISA flags: avx2
- Versions: {'ultralytics': '8.4.211', 'torch': '2.11.0+cpu', 'openvino': '2026.4.1', 'nncf': '3.4.0', 'numpy': '2.1.3', 'scipy': '1.16.3', 'lap': '0.5.14', 'trackeval': '-', 'opencv-python': '4.14.0.94'}
- Data: motchallenge.net (official): https://motchallenge.net/data/MOT17.zip. Sequences: MOT17-02, MOT17-04, MOT17-11. INT8 calibration: 300 frames of MOT17-10.
- Tracker: ByteTrack, ultralytics bytetrack.yaml: {'tracker_type': 'bytetrack', 'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.25, 'track_buffer': 30, 'match_thresh': 0.8, 'fuse_score': True}.
- Method: the detector runs once per model and sequence at conf 0.01 (all classes, max_det 300); each threshold is then applied to the kept person boxes (score > conf) before ByteTrack. See 'Check of the method' below.
- Status: complete; elapsed 1529 s of a 4500 s budget.

## 1. Low thresholds

Detector pass (ms per frame, median; conf 0.01, so post-process sees the most boxes):

| variant | pre | inference | post | fps (detector only, incl. JPEG read) |
|---|---|---|---|---|
| PyTorch FP32 | 2.356 | 79.348 | 1.409 | 10.45 |
| OpenVINO FP32 | 2.163 | 45.545 | 0.404 | 16.47 |
| OpenVINO INT8 | 2.227 | 33.027 | 0.403 | 20.86 |

### Arm 'stock': bytetrack.yaml untouched (track_low_thresh 0.1)

3 sequences combined, detect-every-1:

| variant | conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDSW_per_min | Frag | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 34.663 | 26.337 | 45.91 | 28.546 | 37.49 | 107 | 75.53 | 253 | 23741 | 2061 | 51833 |
| PyTorch FP32 | 0.03 | 34.663 | 26.337 | 45.91 | 28.546 | 37.49 | 107 | 75.53 | 253 | 23741 | 2061 | 51833 |
| PyTorch FP32 | 0.05 | 34.663 | 26.337 | 45.91 | 28.546 | 37.49 | 107 | 75.53 | 253 | 23741 | 2061 | 51833 |
| PyTorch FP32 | 0.1 | 34.663 | 26.337 | 45.91 | 28.546 | 37.49 | 107 | 75.53 | 253 | 23741 | 2061 | 51833 |
| OpenVINO FP32 | 0.01 | 33.774 | 24.977 | 45.976 | 26.514 | 36.725 | 132 | 93.18 | 330 | 22588 | 2418 | 52986 |
| OpenVINO FP32 | 0.03 | 33.774 | 24.977 | 45.976 | 26.514 | 36.725 | 132 | 93.18 | 330 | 22588 | 2418 | 52986 |
| OpenVINO FP32 | 0.05 | 33.774 | 24.977 | 45.976 | 26.514 | 36.725 | 132 | 93.18 | 330 | 22588 | 2418 | 52986 |
| OpenVINO FP32 | 0.1 | 33.774 | 24.977 | 45.976 | 26.514 | 36.725 | 132 | 93.18 | 330 | 22588 | 2418 | 52986 |
| OpenVINO INT8 | 0.01 | 32.758 | 24.684 | 43.721 | 26.614 | 36.058 | 141 | 99.53 | 367 | 22492 | 2238 | 53082 |
| OpenVINO INT8 | 0.03 | 32.758 | 24.684 | 43.721 | 26.614 | 36.058 | 141 | 99.53 | 367 | 22492 | 2238 | 53082 |
| OpenVINO INT8 | 0.05 | 32.758 | 24.684 | 43.721 | 26.614 | 36.058 | 141 | 99.53 | 367 | 22492 | 2238 | 53082 |
| OpenVINO INT8 | 0.1 | 32.758 | 24.684 | 43.721 | 26.614 | 36.058 | 141 | 99.53 | 367 | 22492 | 2238 | 53082 |

| variant | conf(s) with the highest HOTA | highest HOTA | HOTA at 0.1 | gain over 0.1 | best threshold below 0.1? | all metrics identical at every conf? | IDSW per conf |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01, 0.03, 0.05, 0.1 | 34.663 | 34.663 | +0.000 | no | yes | 0.01: 107, 0.03: 107, 0.05: 107, 0.1: 107 |
| OpenVINO FP32 | 0.01, 0.03, 0.05, 0.1 | 33.774 | 33.774 | +0.000 | no | yes | 0.01: 132, 0.03: 132, 0.05: 132, 0.1: 132 |
| OpenVINO INT8 | 0.01, 0.03, 0.05, 0.1 | 32.758 | 32.758 | +0.000 | no | yes | 0.01: 141, 0.03: 141, 0.05: 141, 0.1: 141 |

Threshold drift, INT8 vs OpenVINO FP32: best conf [0.01, 0.03, 0.05, 0.1] vs [0.01, 0.03, 0.05, 0.1]: **no drift**.

### Arm 'low=conf' (extra): track_low_thresh set to the swept conf, everything else stock

3 sequences combined, detect-every-1:

| variant | conf | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDSW_per_min | Frag | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 36.675 | 28.215 | 47.965 | 30.386 | 40.702 | 107 | 75.53 | 243 | 25738 | 2667 | 49836 |
| PyTorch FP32 | 0.03 | 35.595 | 27.564 | 46.29 | 29.841 | 39.247 | 107 | 75.53 | 241 | 25042 | 2383 | 50532 |
| PyTorch FP32 | 0.05 | 35.054 | 27.145 | 45.582 | 29.463 | 38.174 | 110 | 77.65 | 238 | 24614 | 2238 | 50960 |
| PyTorch FP32 | 0.1 | 34.663 | 26.337 | 45.91 | 28.546 | 37.49 | 107 | 75.53 | 253 | 23741 | 2061 | 51833 |
| OpenVINO FP32 | 0.01 | 36.204 | 27.513 | 47.974 | 28.203 | 40.279 | 129 | 91.06 | 240 | 25451 | 4008 | 50123 |
| OpenVINO FP32 | 0.03 | 35.565 | 26.361 | 48.305 | 27.658 | 39.597 | 129 | 91.06 | 251 | 24146 | 3115 | 51428 |
| OpenVINO FP32 | 0.05 | 34.833 | 25.837 | 47.281 | 27.249 | 38.375 | 140 | 98.82 | 263 | 23582 | 2849 | 51992 |
| OpenVINO FP32 | 0.1 | 33.774 | 24.977 | 45.976 | 26.514 | 36.725 | 132 | 93.18 | 330 | 22588 | 2418 | 52986 |
| OpenVINO INT8 | 0.01 | 33.912 | 27.556 | 42.078 | 28.962 | 37.718 | 142 | 100.24 | 251 | 25619 | 3589 | 49955 |
| OpenVINO INT8 | 0.03 | 33.733 | 26.367 | 43.457 | 28.244 | 37.792 | 133 | 93.88 | 270 | 24362 | 2884 | 51212 |
| OpenVINO INT8 | 0.05 | 33.332 | 25.834 | 43.283 | 27.824 | 37.085 | 143 | 100.94 | 293 | 23728 | 2557 | 51846 |
| OpenVINO INT8 | 0.1 | 32.758 | 24.684 | 43.721 | 26.614 | 36.058 | 141 | 99.53 | 367 | 22492 | 2238 | 53082 |

| variant | conf(s) with the highest HOTA | highest HOTA | HOTA at 0.1 | gain over 0.1 | best threshold below 0.1? | all metrics identical at every conf? | IDSW per conf |
|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.01 | 36.675 | 34.663 | +2.012 | yes | no | 0.01: 107, 0.03: 107, 0.05: 110, 0.1: 107 |
| OpenVINO FP32 | 0.01 | 36.204 | 33.774 | +2.430 | yes | no | 0.01: 129, 0.03: 129, 0.05: 140, 0.1: 132 |
| OpenVINO INT8 | 0.01 | 33.912 | 32.758 | +1.154 | yes | no | 0.01: 142, 0.03: 133, 0.05: 143, 0.1: 141 |

Threshold drift, INT8 vs OpenVINO FP32: best conf [0.01] vs [0.01]: **no drift**.

Reading aid: ByteTrack only uses boxes with score > track_low_thresh. In arm 'stock' that is 0.1, so a row of that arm can only differ from its conf 0.1 row if boxes under that score reach the tracker some other way; the column 'all metrics identical at every conf?' says whether they did.

### Check of the method

MOT17-02, conf 0.1: predict(conf, classes=[0]) on every frame (bench.run_sequence, as in the first run) versus the kept boxes of the conf 0.01 pass:

| variant | same tracks? | max box difference px | IDSW direct | IDSW kept boxes | HOTA direct | HOTA kept boxes |
|---|---|---|---|---|---|---|
| PyTorch FP32 | True | 0.0 | 32 | 32 | 29.943 | 29.943 |
| OpenVINO FP32 | True | 0.0 | 55 | 55 | 26.495 | 26.495 |

## 2. Why OpenVINO FP32 differs from PyTorch FP32

### Exports

| export | export args | export s | inputs | outputs | NMS nodes | TopK nodes | metadata | warnings during export |
|---|---|---|---|---|---|---|---|---|
| a: nms=False, 384x640 (current) | {'format': 'openvino', 'imgsz': [384, 640], 'batch': 1, 'device': 'cpu', 'nms': False} | 4.6 | ['[1,3,384,640]'] | ['[1,300,6]'] | 0 | 2 | {'imgsz': [384, 640], 'end2end': True} args {'nms': False, 'dynamic': False} | - |
| b: nms=True, 384x640 | {'nms': True, 'imgsz': [384, 640]} | 4.1 | ['[1,3,384,640]'] | ['[1,300,6]'] | 1 | 1 | {'imgsz': [384, 640], 'end2end': False} args {'nms': True, 'dynamic': False} | - |
| c: nms=False, 640x640 letterbox | {'nms': False, 'imgsz': [640, 640]} | 3.9 | ['[1,3,640,640]'] | ['[1,300,6]'] | 0 | 2 | {'imgsz': [640, 640], 'end2end': True} args {'nms': False, 'dynamic': False} | - |
| d: nms=False, dynamic=True | {'nms': False, 'imgsz': [384, 640], 'dynamic': True} | 4.1 | ['[?,3,?,?]'] | ['[?,300,6]'] | 0 | 2 | {'imgsz': [384, 640], 'end2end': True} args {'nms': False, 'dynamic': True} | - |

### Pre-processing, frame 0 of MOT17-02

The tensor ultralytics hands to the network (predictor.preprocess), PyTorch vs OpenVINO, and the shape of the raw network output:

| export | PyTorch predict args | PyTorch input | OpenVINO input | same shape? | max pixel diff (of 255) | mean pixel diff | values differing | PyTorch raw output | OpenVINO raw output |
|---|---|---|---|---|---|---|---|---|---|
| a: nms=False, 384x640 (current) | {'imgsz': [384, 640]} | (1, 3, 384, 640) | (1, 3, 384, 640) | True | 0.0 | 0.0 | 0 of 737280 | [(1, 84, 5040), {boxes: (1, 4, 5040), scores: (1, 80, 5040), feats: list}] | (1, 300, 6) |
| b: nms=True, 384x640 | {'imgsz': [384, 640]} | (1, 3, 384, 640) | (1, 3, 384, 640) | True | 0.0 | 0.0 | 0 of 737280 | [(1, 84, 5040), {boxes: (1, 4, 5040), scores: (1, 80, 5040), feats: list}] | (1, 300, 6) |
| c: nms=False, 640x640 letterbox | {'imgsz': [640, 640], 'rect': False} | (1, 3, 640, 640) | (1, 3, 640, 640) | True | 0.0 | 0.0 | 0 of 1228800 | [(1, 84, 8400), {boxes: (1, 4, 8400), scores: (1, 80, 8400), feats: list}] | (1, 300, 6) |
| d: nms=False, dynamic=True | {'imgsz': [384, 640]} | (1, 3, 384, 640) | (1, 3, 384, 640) | True | 0.0 | 0.0 | 0 of 737280 | [(1, 84, 5040), {boxes: (1, 4, 5040), scores: (1, 80, 5040), feats: list}] | (1, 300, 6) |

### Raw detections, PyTorch vs OpenVINO FP32

Person boxes with score > floor. Matched = one-to-one, IoU > 0.9. 'only' = boxes of that side without such a match (in brackets: without even an IoU > 0.5 partner). Coordinates in pixels of the 1920x1080 frame.

| export | sequence | floor | boxes/frame PT | boxes/frame OV | matched/frame | PT only/frame | OV only/frame | frames with any unmatched | mean abs score diff | max abs score diff | mean coord diff px | max coord diff px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a: nms=False, 384x640 (current) | MOT17-02 | 0.01 | 63.72 | 114.54 | 17.74 | 45.978 (9.045) | 96.802 (59.868) | 600 of 600 | 0.06289 | 0.84961 | 2.961 | 57.871 |
| a: nms=False, 384x640 (current) | MOT17-02 | 0.1 | 17.77 | 16.35 | 9.8 | 7.972 (4.123) | 6.557 (2.708) | 600 of 600 | 0.06962 | 0.70036 | 2.807 | 46.356 |
| a: nms=False, 384x640 (current) | MOT17-02 | 0.25 | 8.58 | 8.61 | 6.94 | 1.645 (0.837) | 1.67 (0.862) | 554 of 600 | 0.07059 | 0.60615 | 2.777 | 46.356 |
| a: nms=False, 384x640 (current) | MOT17-04 | 0.01 | 46.61 | 66.95 | 27.48 | 19.132 (8.146) | 39.471 (28.485) | 1050 of 1050 | 0.0695 | 0.81694 | 2.535 | 53.82 |
| a: nms=False, 384x640 (current) | MOT17-04 | 0.1 | 19.61 | 20.24 | 15.75 | 3.861 (2.642) | 4.492 (3.273) | 1049 of 1050 | 0.07611 | 0.67652 | 2.053 | 20.176 |
| a: nms=False, 384x640 (current) | MOT17-04 | 0.25 | 13.2 | 12.12 | 11.05 | 2.153 (1.669) | 1.076 (0.591) | 995 of 1050 | 0.07297 | 0.53522 | 1.839 | 20.176 |
| b: nms=True, 384x640 | MOT17-02 | 0.01 | 63.72 | 8.58 | 8.58 | 55.137 (55.137) | 0.0 (0.0) | 600 of 600 | 0.0 | 0.0 | 0.0 | 0.001 |
| b: nms=True, 384x640 | MOT17-02 | 0.1 | 17.77 | 8.58 | 8.58 | 9.187 (9.187) | 0.0 (0.0) | 600 of 600 | 0.0 | 0.0 | 0.0 | 0.001 |
| b: nms=True, 384x640 | MOT17-02 | 0.25 | 8.58 | 8.58 | 8.58 | 0.0 (0.0) | 0.0 (0.0) | 0 of 600 | 0.0 | 0.0 | 0.0 | 0.001 |
| b: nms=True, 384x640 | MOT17-04 | 0.01 | 46.61 | 13.2 | 13.2 | 33.41 (33.41) | 0.0 (0.0) | 1050 of 1050 | 0.0 | 0.0 | 0.0 | 0.0 |
| b: nms=True, 384x640 | MOT17-04 | 0.1 | 19.61 | 13.2 | 13.2 | 6.409 (6.409) | 0.0 (0.0) | 1048 of 1050 | 0.0 | 0.0 | 0.0 | 0.0 |
| b: nms=True, 384x640 | MOT17-04 | 0.25 | 13.2 | 13.2 | 13.2 | 0.0 (0.0) | 0.0 (0.0) | 0 of 1050 | 0.0 | 0.0 | 0.0 | 0.0 |
| c: nms=False, 640x640 letterbox | MOT17-02 | 0.01 | 65.47 | 119.02 | 17.9 | 47.577 (9.202) | 101.12 (62.745) | 600 of 600 | 0.06398 | 0.85686 | 2.951 | 49.347 |
| c: nms=False, 640x640 letterbox | MOT17-02 | 0.1 | 18.73 | 17.18 | 10.09 | 8.638 (4.422) | 7.095 (2.878) | 600 of 600 | 0.069 | 0.72397 | 2.788 | 49.347 |
| c: nms=False, 640x640 letterbox | MOT17-02 | 0.25 | 9.04 | 8.91 | 7.16 | 1.877 (0.978) | 1.752 (0.853) | 557 of 600 | 0.06954 | 0.59965 | 2.766 | 49.347 |
| c: nms=False, 640x640 letterbox | MOT17-04 | 0.01 | 47.55 | 68.42 | 26.56 | 20.99 (9.209) | 41.863 (30.081) | 1050 of 1050 | 0.07253 | 0.80848 | 2.471 | 18.005 |
| c: nms=False, 640x640 letterbox | MOT17-04 | 0.1 | 19.69 | 20.2 | 15.51 | 4.18 (2.836) | 4.694 (3.35) | 1050 of 1050 | 0.07919 | 0.71812 | 2.057 | 18.571 |
| c: nms=False, 640x640 letterbox | MOT17-04 | 0.25 | 13.35 | 12.35 | 11.17 | 2.189 (1.654) | 1.189 (0.654) | 1010 of 1050 | 0.07521 | 0.52058 | 1.831 | 18.571 |
| d: nms=False, dynamic=True | MOT17-02 | 0.01 | 63.72 | 114.54 | 17.74 | 45.978 (9.045) | 96.802 (59.868) | 600 of 600 | 0.06289 | 0.84961 | 2.961 | 57.871 |
| d: nms=False, dynamic=True | MOT17-02 | 0.1 | 17.77 | 16.35 | 9.8 | 7.972 (4.123) | 6.557 (2.708) | 600 of 600 | 0.06962 | 0.70036 | 2.807 | 46.356 |
| d: nms=False, dynamic=True | MOT17-02 | 0.25 | 8.58 | 8.61 | 6.94 | 1.645 (0.837) | 1.67 (0.862) | 554 of 600 | 0.07059 | 0.60615 | 2.777 | 46.356 |
| d: nms=False, dynamic=True | MOT17-04 | 0.01 | 46.61 | 66.95 | 27.48 | 19.132 (8.146) | 39.471 (28.485) | 1050 of 1050 | 0.0695 | 0.81694 | 2.535 | 53.82 |
| d: nms=False, dynamic=True | MOT17-04 | 0.1 | 19.61 | 20.24 | 15.75 | 3.861 (2.642) | 4.492 (3.273) | 1049 of 1050 | 0.07611 | 0.67652 | 2.053 | 20.176 |
| d: nms=False, dynamic=True | MOT17-04 | 0.25 | 13.2 | 12.12 | 11.05 | 2.153 (1.669) | 1.076 (0.591) | 995 of 1050 | 0.07297 | 0.53522 | 1.839 | 20.176 |

### The 300-box cap (conf 0.01, boxes of all classes)

| model | sequence | boxes/frame mean | max | frames at the cap (300) | lowest kept score in capped frames (median / max) | capped frames whose lowest kept score is > 0.1 | person boxes/frame | person boxes > 0.1 /frame |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | MOT17-02 | 110.0 | 159 | 0 of 600 | None / None | 0 | 63.7 | 17.8 |
| PyTorch FP32 | MOT17-04 | 95.9 | 128 | 0 of 1050 | None / None | 0 | 46.6 | 19.6 |
| PyTorch FP32 | MOT17-11 | 54.1 | 92 | 0 of 900 | None / None | 0 | 28.2 | 9.1 |
| OpenVINO FP32 | MOT17-02 | 193.0 | 264 | 0 of 600 | None / None | 0 | 114.5 | 16.4 |
| OpenVINO FP32 | MOT17-04 | 145.9 | 195 | 0 of 1050 | None / None | 0 | 66.9 | 20.2 |
| OpenVINO FP32 | MOT17-11 | 64.8 | 140 | 0 of 900 | None / None | 0 | 37.9 | 9.4 |
| OpenVINO INT8 | MOT17-02 | 192.1 | 264 | 0 of 600 | None / None | 0 | 107.8 | 15.6 |
| OpenVINO INT8 | MOT17-04 | 146.2 | 195 | 0 of 1050 | None / None | 0 | 65.3 | 20.2 |
| OpenVINO INT8 | MOT17-11 | 64.5 | 124 | 0 of 900 | None / None | 0 | 36.1 | 9.4 |
| OpenVINO FP32 export b | MOT17-02 | 10.4 | 18 | 0 of 600 | None / None | 0 | 8.6 | 8.6 |
| OpenVINO FP32 export b | MOT17-04 | 15.8 | 23 | 0 of 1050 | None / None | 0 | 13.2 | 13.2 |
| PyTorch FP32 {'imgsz': [640, 640], 'rect': False} | MOT17-02 | 111.6 | 166 | 0 of 600 | None / None | 0 | 65.5 | 18.7 |
| PyTorch FP32 {'imgsz': [640, 640], 'rect': False} | MOT17-04 | 98.3 | 128 | 0 of 1050 | None / None | 0 | 47.5 | 19.7 |
| OpenVINO FP32 export c | MOT17-02 | 200.3 | 276 | 0 of 600 | None / None | 0 | 119.0 | 17.2 |
| OpenVINO FP32 export c | MOT17-04 | 150.8 | 197 | 0 of 1050 | None / None | 0 | 68.4 | 20.2 |
| OpenVINO FP32 export d | MOT17-02 | 193.0 | 264 | 0 of 600 | None / None | 0 | 114.5 | 16.4 |
| OpenVINO FP32 export d | MOT17-04 | 145.9 | 195 | 0 of 1050 | None / None | 0 | 66.9 | 20.2 |

A capped frame can only have lost boxes that score under its lowest kept score.

Asking for more boxes (first frames of MOT17-04, conf 0.01):

| model | setting | result |
|---|---|---|
| PyTorch FP32 | max_det=300 | {'frames': 20, 'boxes_mean': 98.7, 'boxes_max': 106} |
| PyTorch FP32 | max_det=1000 | {'frames': 20, 'boxes_mean': 98.7, 'boxes_max': 106} |
| OpenVINO FP32 | max_det=300 | {'frames': 20, 'boxes_mean': 153.3, 'boxes_max': 174} |
| OpenVINO FP32 | max_det=1000 | {'frames': 20, 'boxes_mean': 153.3, 'boxes_max': 174} |

### Tracking from those detections (conf 0.1, stock ByteTrack)

| export | sequence | side | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Frag | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a: nms=False, 384x640 (current) | MOT17-02 | PyTorch | 29.943 | 19.279 | 46.804 | 21.204 | 32.658 | 32 | 82 | 4414 | 442 | 14167 |
| a: nms=False, 384x640 (current) | MOT17-04 | PyTorch | 32.843 | 24.276 | 44.725 | 26.368 | 35.088 | 44 | 117 | 13578 | 994 | 33979 |
| a: nms=False, 384x640 (current) | MOT17-02 | OpenVINO | 26.495 | 18.441 | 38.287 | 19.611 | 27.883 | 55 | 98 | 4247 | 548 | 14334 |
| a: nms=False, 384x640 (current) | MOT17-04 | OpenVINO | 32.819 | 22.612 | 47.939 | 24.131 | 35.816 | 44 | 173 | 12694 | 1174 | 34863 |
| b: nms=True, 384x640 | MOT17-02 | PyTorch | 29.943 | 19.279 | 46.804 | 21.204 | 32.658 | 32 | 82 | 4414 | 442 | 14167 |
| b: nms=True, 384x640 | MOT17-04 | PyTorch | 32.843 | 24.276 | 44.725 | 26.368 | 35.088 | 44 | 117 | 13578 | 994 | 33979 |
| b: nms=True, 384x640 | MOT17-02 | OpenVINO | 29.416 | 18.078 | 48.03 | 20.123 | 30.857 | 32 | 134 | 4064 | 293 | 14517 |
| b: nms=True, 384x640 | MOT17-04 | OpenVINO | 31.893 | 22.333 | 45.803 | 23.994 | 33.796 | 45 | 299 | 12391 | 935 | 35166 |
| c: nms=False, 640x640 letterbox | MOT17-02 | PyTorch | 29.891 | 19.962 | 45.098 | 21.791 | 32.716 | 39 | 89 | 4598 | 510 | 13983 |
| c: nms=False, 640x640 letterbox | MOT17-04 | PyTorch | 33.571 | 25.064 | 45.272 | 27.228 | 36.165 | 42 | 121 | 14037 | 1046 | 33520 |
| c: nms=False, 640x640 letterbox | MOT17-02 | OpenVINO | 28.045 | 18.773 | 42.107 | 19.735 | 30.479 | 54 | 95 | 4355 | 634 | 14226 |
| c: nms=False, 640x640 letterbox | MOT17-04 | OpenVINO | 31.78 | 22.421 | 45.375 | 23.685 | 33.809 | 54 | 169 | 12600 | 1282 | 34957 |
| d: nms=False, dynamic=True | MOT17-02 | PyTorch | 29.943 | 19.279 | 46.804 | 21.204 | 32.658 | 32 | 82 | 4414 | 442 | 14167 |
| d: nms=False, dynamic=True | MOT17-04 | PyTorch | 32.843 | 24.276 | 44.725 | 26.368 | 35.088 | 44 | 117 | 13578 | 994 | 33979 |
| d: nms=False, dynamic=True | MOT17-02 | OpenVINO | 26.495 | 18.441 | 38.287 | 19.611 | 27.883 | 55 | 98 | 4247 | 548 | 14334 |
| d: nms=False, dynamic=True | MOT17-04 | OpenVINO | 32.819 | 22.612 | 47.939 | 24.131 | 35.816 | 44 | 173 | 12694 | 1174 | 34863 |

## Conclusions (from the numbers above only)

- Thresholds, PyTorch FP32, arm 'stock': highest HOTA 34.663 at conf [0.01, 0.03, 0.05, 0.1]; conf 0.1 gives 34.663, so the best threshold is not below 0.1.
- Thresholds, OpenVINO FP32, arm 'stock': highest HOTA 33.774 at conf [0.01, 0.03, 0.05, 0.1]; conf 0.1 gives 33.774, so the best threshold is not below 0.1.
- Thresholds, OpenVINO INT8, arm 'stock': highest HOTA 32.758 at conf [0.01, 0.03, 0.05, 0.1]; conf 0.1 gives 32.758, so the best threshold is not below 0.1.
- Thresholds, PyTorch FP32, arm 'low=conf': highest HOTA 36.675 at conf [0.01]; conf 0.1 gives 34.663, so the best threshold is below 0.1.
- Thresholds, OpenVINO FP32, arm 'low=conf': highest HOTA 36.204 at conf [0.01]; conf 0.1 gives 33.774, so the best threshold is below 0.1.
- Thresholds, OpenVINO INT8, arm 'low=conf': highest HOTA 33.912 at conf [0.01]; conf 0.1 gives 32.758, so the best threshold is below 0.1.
- Export a: nms=False, 384x640 (current), MOT17-02: input tensors identical; boxes with score > 0.1 per frame: 7.972 PyTorch-only, 6.557 OpenVINO-only, 9.8 matched (mean score difference 0.06962, mean coordinate difference 2.807 px); ID switches PyTorch 32 vs OpenVINO 55, HOTA 29.943 vs 26.495. The two detectors already return different boxes before the tracker sees them.
- Export b: nms=True, 384x640, MOT17-02: input tensors identical; boxes with score > 0.1 per frame: 9.187 PyTorch-only, 0.0 OpenVINO-only, 8.58 matched (mean score difference 0.0, mean coordinate difference 0.0 px); ID switches PyTorch 32 vs OpenVINO 32, HOTA 29.943 vs 29.416. The two detectors already return different boxes before the tracker sees them.
- Export c: nms=False, 640x640 letterbox, MOT17-02: input tensors identical; boxes with score > 0.1 per frame: 8.638 PyTorch-only, 7.095 OpenVINO-only, 10.09 matched (mean score difference 0.069, mean coordinate difference 2.788 px); ID switches PyTorch 39 vs OpenVINO 54, HOTA 29.891 vs 28.045. The two detectors already return different boxes before the tracker sees them.
- Export d: nms=False, dynamic=True, MOT17-02: input tensors identical; boxes with score > 0.1 per frame: 7.972 PyTorch-only, 6.557 OpenVINO-only, 9.8 matched (mean score difference 0.06962, mean coordinate difference 2.807 px); ID switches PyTorch 32 vs OpenVINO 55, HOTA 29.943 vs 26.495. The two detectors already return different boxes before the tracker sees them.
- Smallest ID switch gap between PyTorch and OpenVINO on MOT17-02: export b: nms=True, 384x640 (gap 0); gaps of all exports: a = 23, b = 0, c = 15, d = 23.
- 300-box cap at conf 0.01: no frame of any model reached it.

Raw rows: p2f_results.csv / p2f_results.json (the json also holds every table above as data).
