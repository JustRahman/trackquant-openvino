"""Build the corrected (v2) Kaggle kernels: INTEL_GUARD (taken from kaggle/p2_yolo26/trackquant_bench.py) +
src/trackquant/bench.py with its KERNEL_ARGS line filled in. The old p2_yolo26 / p2_yolov8 kernels are not touched.

  python3 scripts/p2_embed_v2.py     (standard library only)
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "trackquant" / "bench.py"
GUARDED = ROOT / "kaggle" / "p2_yolo26" / "trackquant_bench.py"
MARK = "KERNEL_ARGS = []"
GRID = ["--track-low", "conf", "--confs", "0.01,0.03,0.05,0.1,0.2", "--budget-min", "150"]  # N grid 1, 2, 3, 5 is bench's default
KERNELS = {
    "p2_yolo26_v2": ("trackquant-ov-yolo26-v2", "TrackQuant-OV YOLO26 v2", ["--model", "yolo26n", "--yolo26-export-nms", "true"] + GRID),
    "p2_yolov8_v2": ("trackquant-ov-yolov8-v2", "TrackQuant-OV YOLOv8 v2", ["--model", "yolov8n"] + GRID),
}

guard = GUARDED.read_text().split('"""', 1)[0]
assert guard.startswith("# INTEL_GUARD") and "GenuineIntel" in guard, "INTEL_GUARD block not found"
src = SRC.read_text()
assert src.count(MARK) == 1, "KERNEL_ARGS line not found exactly once in bench.py"
for folder, (slug, title, kernel_args) in KERNELS.items():
    d = ROOT / "kaggle" / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "trackquant_bench.py").write_text(guard + src.replace(MARK, f"KERNEL_ARGS = {kernel_args!r}"))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'your-kaggle-username')}/{slug}", "title": title, "code_file": "trackquant_bench.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, indent=2) + "\n")
    print(f"kaggle/{folder}/trackquant_bench.py: {' '.join(kernel_args)}")
