"""Build the single-file Kaggle kernel kaggle/p2_followup/trackquant_followup.py:
INTEL_GUARD (taken from kaggle/p2_yolo26/trackquant_bench.py) + src/trackquant/followup.py with its
`from bench import *` line replaced by src/trackquant/bench.py itself (minus bench's own __main__ block).

  python3 scripts/p2_followup_embed.py     (standard library only)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "src" / "trackquant" / "bench.py"
FOLLOWUP = ROOT / "src" / "trackquant" / "followup.py"
GUARDED = ROOT / "kaggle" / "p2_yolo26" / "trackquant_bench.py"
DST = ROOT / "kaggle" / "p2_followup" / "trackquant_followup.py"
BENCH_TAIL = '\n\nif __name__ == "__main__":\n    main()\n'

guard = GUARDED.read_text().split('"""', 1)[0]
assert guard.startswith("# INTEL_GUARD") and "GenuineIntel" in guard, "INTEL_GUARD block not found"
bench = BENCH.read_text()
assert bench.endswith(BENCH_TAIL), "bench.py does not end with its __main__ block"
lines = FOLLOWUP.read_text().split("\n")
star = [i for i, ln in enumerate(lines) if ln.startswith("from bench import *")]
assert len(star) == 1, "followup.py must have exactly one 'from bench import *' line"
lines[star[0]] = ("# ---- BEGIN src/trackquant/bench.py (inlined by scripts/p2_followup_embed.py; do not edit here) ----\n"
                  + bench[: -len(BENCH_TAIL)] + "\n\n# ---- END src/trackquant/bench.py ----\n")
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(guard + "\n".join(lines))
print(f"{DST.relative_to(ROOT)}: {len(DST.read_text().splitlines())} lines")
