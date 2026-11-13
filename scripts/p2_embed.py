"""Regenerate the Kaggle copies of src/trackquant/bench.py (one per kernel, differing only in DEFAULT_MODEL).

  python3 scripts/p2_embed.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "trackquant" / "bench.py"
KERNELS = {"p2_yolo26": "yolo26n", "p2_yolov8": "yolov8n"}
MARK = 'DEFAULT_MODEL = "yolo26n"'

src = SRC.read_text()
assert src.count(MARK) == 1, "DEFAULT_MODEL line not found exactly once in bench.py"
for folder, model in KERNELS.items():
    dst = ROOT / "kaggle" / folder / "trackquant_bench.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.replace(MARK, f'DEFAULT_MODEL = "{model}"'))
    print(f"{dst.relative_to(ROOT)}: default --model {model}, {len(src.splitlines())} lines")
