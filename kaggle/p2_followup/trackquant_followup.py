# INTEL_GUARD: speed numbers must come from an Intel CPU. Exit at once on any other vendor so the job can be re-launched.
import sys as _s
try:
    _v = [l.split(":",1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("vendor_id")][0]
except Exception:
    _v = "unknown"
print("[guard] cpu vendor:", _v, flush=True)
if _v != "GenuineIntel":
    open("/kaggle/working/NOT_INTEL.txt", "w").write(_v)
    print("[guard] not an Intel CPU, exiting so it can be re-launched", flush=True)
    _s.exit(0)

"""TrackQuant-OV follow-up (yolo26n only): two questions left open by results/kaggle_yolo26/p2_report.md.

  1. LOW THRESHOLDS. The best confidence threshold was 0.1 = the edge of the grid. Here: PyTorch FP32, OpenVINO FP32,
     OpenVINO INT8 at conf 0.01 / 0.03 / 0.05 (and 0.1 again, same run), detect-every-1, the same 3 sequences.
     Arm "stock": ByteTrack's bytetrack.yaml untouched (as before). Arm "low=conf" (extra): track_low_thresh is set
     to the swept conf, because stock ByteTrack ignores every box at or under its track_low_thresh.
  2. WHY OpenVINO FP32 != PyTorch FP32. Raw detections per frame, PyTorch vs OpenVINO FP32, for four exports:
     (a) nms=False 384x640 (the one used so far), (b) nms=True, (c) 640x640 letterbox on both sides,
     (d) dynamic=True. Matched boxes (IoU > 0.9): |score diff|, box coordinate diff; unmatched boxes per frame;
     frames that hit the 300-box cap; HOTA / IDSW of both sides; input tensor shape + pixel diff for frame 0.

How it is measured: the detector runs ONCE per model and sequence at conf 0.01 (all classes, max_det 300) and the
boxes are kept. Every threshold / tracker setting is then "keep person boxes with score > conf -> ByteTrack" on
those boxes. One direct run per model (bench.run_sequence at conf 0.1 on MOT17-02) checks that this gives the
same tracks as calling predict(conf=0.1, classes=[0]) frame by frame; the result of that check is in the report.

Smoke test (60 frames per sequence):  python followup.py --smoke [--data-dir /path/to/MOT17/train]
"""
# ---- BEGIN src/trackquant/bench.py (inlined by scripts/p2_followup_embed.py; do not edit here) ----
"""TrackQuant-OV (project 2): what OpenVINO FP16 / INT8 quantization does to multi-object TRACKING.

Kaggle "script" kernel (CPU, internet on). One file, one detector per kernel (--model yolo26n | yolov8n).

What it does
  1. pip installs pinned deps (Kaggle only), records versions / lscpu / RAM -> p2_env.txt
  2. downloads 4 MOT17 train sequences (HTTP range requests into MOT17.zip, so ~1 GB instead of 5.9 GB):
     3 TEST sequences (02 static camera, 04 crowded high static view, 11 moving camera) and 1 CALIBRATION
     sequence (10, never used for scoring). Only images + gt of the -FRCNN folder; its detections are ignored.
  3. exports the detector with ultralytics: OpenVINO FP32, FP16, INT8 (NNCF post-training quantization,
     calibrated on ~300 frames of MOT17-10). PyTorch FP32 is the reference. Records export time + size.
  4. for every variant runs detector -> ByteTrack (ultralytics bytetrack.yaml, untouched) on the 3 sequences:
       phase A: confidence threshold sweep 0.1..0.5 at detect-every-1
       phase B: detect-every-N sweep (N = 2, 3, 5) at that variant's best threshold (N = 1 comes from phase A)
  5. scores every run with HOTA / DetA / AssA, MOTA, IDF1, ID switches (+ per minute), fragmentation
     (own implementation of the TrackEval formulas, cross-checked once against the PyPI trackeval package)
  6. writes p2_results.csv/.json (after every run), plots, p2_report.md (measured values only)

How one frame is timed
  pre / inference / post come from ultralytics' own profilers (Results.speed): letterbox+normalise, the
  network, and decode (+ NMS for YOLOv8; YOLO26's head is NMS-free). "track" is our own timer around
  ByteTrack's update(). Reading the JPEG from disk is NOT part of the split (it is in fps_wall only).

Input size: imgsz 640 on the long side. MOT17 frames are 1920x1080, so PyTorch letterboxes them to 384x640
("rect" inference). The OpenVINO models are exported at exactly that static shape (384x640), so all four
variants see the same pixels and the timing comparison is fair.

Detect-every-N: on frames without detection the tracker is advanced with the Kalman filter only (predict
step, no update) and the predicted boxes of the live tracks are written out with their IDs. tracker.frame_id
is still advanced, so ByteTrack's track_buffer stays 30 VIDEO frames. If the tracker internals ever change
and this fails, the code falls back to repeating the last boxes ("hold") and says so in the results.

Smoke test (same code path, 60 frames per sequence, 2 thresholds, N = 1 and 2):
  python bench.py --smoke [--data-dir /path/to/MOT17/train]
The full grid refuses to run outside Kaggle.
"""
import argparse
import configparser
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

DEFAULT_MODEL = "yolo26n"  # scripts/p2_embed.py rewrites this line for each kernel
PINS = ["ultralytics==8.4.211", "openvino==2026.4.1", "nncf==3.4.0", "lap>=0.5.12", "psutil", "pandas", "matplotlib"]
NOPIN = ["ultralytics", "openvino", "nncf", "lap", "psutil", "pandas", "matplotlib"]
MOT_URLS = [  # tried in order; the report says which one was used
    ("motchallenge.net (official)", "https://motchallenge.net/data/MOT17.zip"),
    ("Hugging Face mirror Lekim89/MOT17 (unofficial)", "hf://Lekim89/MOT17"),
]
TEST_SEQS = ["MOT17-02", "MOT17-04", "MOT17-11"]  # static / crowded static high view / moving camera
CALIB_SEQ = "MOT17-10"  # moving camera, 1080p, never scored
CALIB_FRAMES = 300
IMGSZ = [384, 640]  # (h, w): what imgsz=640 rect inference gives on 1920x1080
CONFS = [0.1, 0.2, 0.3, 0.4, 0.5]
EVERY = [1, 2, 3, 5]
VARIANTS = ["pt_fp32", "ov_fp32", "ov_int8", "ov_fp16"]  # run order: INT8 before FP16 (more important if time runs out)
LABEL = {"pt_fp32": "PyTorch FP32", "ov_fp32": "OpenVINO FP32", "ov_fp16": "OpenVINO FP16", "ov_int8": "OpenVINO INT8"}
BUDGET_S = 2 * 3600 + 45 * 60  # hard stop for the whole kernel
RESERVE_S = 300  # kept for the report / plots
DISTRACTORS = (2, 7, 8, 12)  # MOT17 gt classes: person_on_vehicle, static_person, distractor, reflection
T0 = time.time()
SKIP_MODE = "kalman"  # becomes "hold" if the Kalman-only step fails (see module docstring)


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<{e}>"


def pip(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    if r.returncode:
        log(f"pip {pkgs} failed: {r.stderr[-400:]}")
    return r.returncode == 0


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "-"


def write_env(out, meta):
    """Everything needed to interpret the timings: CPU model, ISA flags that decide INT8 speed, RAM, versions."""
    import psutil

    flags = set(sh("grep -m1 flags /proc/cpuinfo").split())
    isa = [f for f in ("avx2", "avx_vnni", "avx512f", "avx512_vnni", "avx512_bf16", "amx_int8", "amx_bf16") if f in flags]
    meta.update(
        cpu=sh("lscpu | grep 'Model name' | sed 's/.*: *//'") or platform.processor(),
        vcpus=os.cpu_count(),
        ram_gb=round(psutil.virtual_memory().total / 2**30, 1),
        isa=isa,
        python=platform.python_version(),
        versions={p: version(p) for p in ("ultralytics", "torch", "openvino", "nncf", "numpy", "scipy", "lap", "trackeval", "opencv-python")},
    )
    txt = [f"{k}: {meta[k]}" for k in ("cpu", "vcpus", "ram_gb", "isa", "python", "versions")]
    txt += ["", "--- lscpu ---", sh("lscpu"), "", "--- free -h ---", sh("free -h"), "", "--- df -h ---", sh("df -h /kaggle/working /tmp 2>/dev/null")]
    (out / "p2_env.txt").write_text("\n".join(txt))
    log(f"cpu: {meta['cpu']} x{meta['vcpus']}, {meta['ram_gb']} GB, isa: {isa}")


# ----------------------------------------------------------------------------------------------- data


def curl(url, *opts, tries=4):
    """All HTTP goes through curl: it retries, follows redirects and does not hang on a dead IPv6 route like urllib."""
    cmd = ["curl", "-sS", "-L", "--fail", "--connect-timeout", "30", "--retry", "3", *opts, url]
    for attempt in range(tries):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            return r.stdout
        log(f"  curl attempt {attempt + 1} failed: {r.stderr.decode(errors='replace')[-200:]}")
        time.sleep(5)
    raise IOError(f"curl failed for {url} {opts}")


class HttpRangeFile:
    """Read-only seekable 'file' over HTTP Range requests. zipfile only needs it to read the central directory."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = curl(url, "-I").decode(errors="replace").lower()
        self.size = int(head.rsplit("content-length:", 1)[1].split()[0])  # last response if there were redirects

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        data = curl(self.url, "-r", f"{self.pos}-{end - 1}")
        if len(data) != end - self.pos:
            raise IOError(f"range request returned {len(data)} bytes, wanted {end - self.pos} (no Range support?)")
        self.pos += len(data)
        return data


def fetch_range(url, path, lo, hi):
    """Download bytes [lo, hi) of url and write them at the same offsets of a local (sparse) file."""
    part = Path(f"{path}.{lo}.part")
    curl(url, "-r", f"{lo}-{hi - 1}", "-o", str(part))
    if part.stat().st_size != hi - lo:
        raise IOError(f"range {lo}-{hi}: got {part.stat().st_size} bytes")
    with open(part, "rb") as src, open(path, "r+b") as dst:
        dst.seek(lo)
        shutil.copyfileobj(src, dst, 1 << 22)
    part.unlink()


def download_mot17(url, root, seqs, max_frames):
    """Fetch only the needed sequences out of the 5.9 GB MOT17.zip.

    The zip's central directory (a few MB at the end of the file) lists where every member starts. The files of
    one sequence sit next to each other, so each sequence is ONE byte range. The ranges are written at their
    original offsets into a sparse local file, which zipfile can then open like the real archive.
    """
    remote = HttpRangeFile(url)
    zf = zipfile.ZipFile(remote)
    want = {}
    for zi in zf.infolist():
        p = zi.filename.split("/")  # MOT17/train/MOT17-02-FRCNN/img1/000001.jpg
        if len(p) < 4 or p[1] != "train" or not p[2].endswith("-FRCNN") or p[2][:8] not in seqs or zi.is_dir():
            continue
        is_img = p[3] == "img1" and p[-1].endswith(".jpg") and (not max_frames or int(p[-1][:-4]) <= max_frames)
        if is_img or zi.filename.endswith(("gt/gt.txt", "seqinfo.ini")):
            want.setdefault(p[2], []).append(zi)
    missing = [s for s in seqs if not any(k.startswith(s) for k in want)]
    if missing:
        raise IOError(f"sequences {missing} not found in {url}")

    cd_start = zf.start_dir  # offset of the central directory
    ranges = [(cd_start, remote.size)]
    for members in want.values():
        lo = min(z.header_offset for z in members)
        hi = min(cd_start, max(z.header_offset + z.compress_size for z in members) + (1 << 16))  # + local header slack
        ranges.append((lo, hi))
    need = sum(hi - lo for lo, hi in ranges)
    if need > 0.8 * remote.size:  # members not contiguous after all: just take the whole file
        ranges, need = [(0, remote.size)], remote.size
    log(f"MOT17.zip is {remote.size / 2**30:.2f} GB; downloading {need / 2**30:.2f} GB in {len(ranges)} ranges")

    sparse = root / "MOT17_partial.zip"
    with open(sparse, "wb") as f:
        f.truncate(remote.size)
    errors = []

    def job(lo, hi):
        try:
            fetch_range(url, sparse, lo, hi)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=job, args=r) for r in ranges]
    [t.start() for t in threads]
    [t.join() for t in threads]
    if errors:
        raise errors[0]
    with zipfile.ZipFile(sparse) as local:
        for members in want.values():
            for zi in members:
                local.extract(zi, root)
    sparse.unlink()
    return need


def download_mot17_hf(repo, root, seqs, max_frames):
    """Fallback: an unofficial Hugging Face copy that stores the sequences as plain files (train/<SEQ>-FRCNN/...)."""
    pip("huggingface_hub")
    from huggingface_hub import snapshot_download

    img = "0000[0-5]?.jpg" if max_frames else "*.jpg"  # smoke: frames 1..59
    pats = [f"train/{s}-FRCNN/{x}" for s in seqs for x in (f"img1/{img}", "gt/gt.txt", "seqinfo.ini")]
    snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=str(root / "MOT17"))
    return sum(f.stat().st_size for f in (root / "MOT17" / "train").rglob("*") if f.is_file())


def get_mot17(args, work, meta):
    """Returns the MOT17 'train' folder that holds <SEQ>-FRCNN/{img1,gt,seqinfo.ini}."""
    seqs = TEST_SEQS + [CALIB_SEQ]
    max_frames = 60 if args.smoke else 0
    if args.data_dir:
        meta["data_source"] = f"local folder {args.data_dir}"
        return Path(args.data_dir)
    root = work / "data"
    root.mkdir(parents=True, exist_ok=True)
    last = None
    for name, url in MOT_URLS:
        try:
            t = time.time()
            fn = download_mot17_hf if url.startswith("hf://") else download_mot17
            nbytes = fn(url.replace("hf://", ""), root, seqs, max_frames)
            for q in seqs:  # a source only counts if every sequence is really there
                d = root / "MOT17" / "train" / f"{q}-FRCNN"
                if not (d / "gt" / "gt.txt").exists() or not (d / "seqinfo.ini").exists() or not any((d / "img1").glob("*.jpg")):
                    raise IOError(f"{q} incomplete after download")
            meta.update(data_source=f"{name}: {url}", data_download_gb=round(nbytes / 2**30, 3), data_download_s=round(time.time() - t))
            log(f"MOT17 from {name}: {nbytes / 2**30:.2f} GB in {time.time() - t:.0f} s")
            return root / "MOT17" / "train"
        except Exception as e:
            last = e
            log(f"MOT17 download from {name} failed: {e}")
    raise SystemExit(f"no MOT17 source worked ({last})")


def load_sequence(train_dir, seq, max_frames=0):
    """Frames, fps and ground truth of one sequence. gt[frame] = array of rows (id, x, y, w, h, flag, class)."""
    import numpy as np

    d = train_dir / f"{seq}-FRCNN"
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    info = ini["Sequence"]
    frames = sorted((d / info.get("imDir", "img1")).glob("*.jpg"))
    if max_frames:
        frames = frames[:max_frames]
    raw = np.loadtxt(d / "gt" / "gt.txt", delimiter=",", ndmin=2)  # frame,id,x,y,w,h,flag,class,visibility
    gt = {}
    for t in range(1, len(frames) + 1):
        gt[t] = raw[raw[:, 0] == t][:, 1:8]
    return dict(name=seq, dir=d, frames=frames, fps=float(info["frameRate"]), gt=gt, raw_gt=raw,
                w=int(info["imWidth"]), h=int(info["imHeight"]))


def make_calib_dataset(seq, work, n):
    """YOLO-format dataset (images + person labels) from the calibration sequence, for ultralytics' INT8 export."""
    root = work / "calib"
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    step = max(1, len(seq["frames"]) // n)
    picked = seq["frames"][::step][:n]  # evenly spread over the sequence
    for p in picked:
        shutil.copy(p, root / "images" / "val" / p.name)
        g = seq["raw_gt"]
        g = g[(g[:, 0] == int(p.stem)) & (g[:, 6] == 1) & (g[:, 7] == 1)]
        lines = []
        for _, _, x, y, w, h, *_ in g:
            x1, y1, x2, y2 = max(0, x), max(0, y), min(seq["w"], x + w), min(seq["h"], y + h)
            if x2 > x1 and y2 > y1:
                lines.append(f"0 {(x1 + x2) / 2 / seq['w']:.6f} {(y1 + y2) / 2 / seq['h']:.6f} {(x2 - x1) / seq['w']:.6f} {(y2 - y1) / seq['h']:.6f}")
        (root / "labels" / "val" / f"{p.stem}.txt").write_text("\n".join(lines))
    yaml = root / "calib.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/val\nval: images/val\nnames:\n  0: person\n")
    return yaml, len(picked)


# ----------------------------------------------------------------------------------------------- export


def dir_size_mb(p):
    p = Path(p)
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and f.suffix in (".xml", ".bin")]
    return round(sum(f.stat().st_size for f in files) / 2**20, 2)


def export_variant(variant, weights, work, calib_yaml, model_name):
    """Export one OpenVINO variant with ultralytics. Returns (path, seconds, size MB, export kwargs used)."""
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    d = work / "models" / variant  # own folder per variant: FP32 and FP16 exports would otherwise overwrite each other
    d.mkdir(parents=True, exist_ok=True)
    w = d / Path(weights).name
    shutil.copy(weights, w)
    kw = dict(format="openvino", imgsz=IMGSZ, batch=1, device="cpu")
    if model_name.startswith("yolo26"):
        kw["nms"] = False  # keep YOLO26's NMS-free (end-to-end) head in the exported graph
    new_api = "quantize" in DEFAULT_CFG_DICT  # ultralytics >= 8.4.2xx: quantize=16/8 replaces half= / int8=
    if variant == "ov_fp16":
        kw.update(quantize=16) if new_api else kw.update(half=True)
    if variant == "ov_int8":
        kw.update(quantize=8) if new_api else kw.update(int8=True)
        kw.update(data=str(calib_yaml), fraction=1.0)
    t = time.time()
    path = YOLO(str(w)).export(**kw)
    secs = round(time.time() - t, 1)
    log(f"export {variant}: {secs} s -> {path} ({dir_size_mb(path)} MB) kwargs={kw}")
    return str(path), secs, dir_size_mb(path), {k: v for k, v in kw.items() if k != "data"}


def runtime_precisions(model_dir):
    """Which precision each layer REALLY ran in on this CPU (OpenVINO exec graph), e.g. {'u8': 60, 'f32': 12}."""
    import openvino as ov

    xml = next(Path(model_dir).glob("*.xml"))
    compiled = ov.Core().compile_model(str(xml), "CPU")
    counts = {}
    for op in compiled.get_runtime_model().get_ordered_ops():
        rt = op.get_rt_info()
        if rt["layerType"].astype(str) in ("Input", "Output", "Const", "Constant"):
            continue
        p = rt["runtimePrecision"].astype(str).lower()
        counts[p] = counts.get(p, 0) + 1
    hint = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    return counts, hint


# ----------------------------------------------------------------------------------------------- metrics
# Own implementation of the TrackEval formulas (HOTA: Luiten et al. 2020, CLEAR MOT, Identity/IDF1) so the
# kernel does not depend on an evaluator that pins numpy. Boxes are (x, y, w, h), similarity is IoU.


def iou_matrix(a, b):
    import numpy as np

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def preprocess(seq, tracks):
    """MOTChallenge pre-processing (same rules as TrackEval's MotChallenge2DBox):
    - tracker boxes that match (IoU >= 0.5) a gt box of a distractor class are removed (not counted as FP)
    - only gt boxes of class 1 (pedestrian) with the 'consider' flag = 1 are kept
    (the gt visibility column is NOT used by the official protocol, so it is not used here either)
    Returns per-frame (gt_ids, tracker_ids, IoU matrix) with ids renumbered 0..n-1, plus the id counts.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    frames, all_g, all_t = [], [], []
    empty = np.zeros((0, 5))
    for t in range(1, len(seq["frames"]) + 1):
        g, d = seq["gt"][t], tracks.get(t, empty)
        sim = iou_matrix(g[:, 1:5], d[:, 1:5])
        if len(g) and len(d):
            m = sim.copy()
            m[m < 0.5 - 1e-9] = 0
            rows, cols = linear_sum_assignment(-m)
            ok = m[rows, cols] > 1e-9
            drop = cols[ok & np.isin(g[rows, 6], DISTRACTORS)]
            d, sim = np.delete(d, drop, axis=0), np.delete(sim, drop, axis=1)
        keep = (g[:, 5] != 0) & (g[:, 6] == 1)
        g, sim = g[keep], sim[keep]
        frames.append((g[:, 0].astype(int), d[:, 0].astype(int), sim))
        all_g.append(frames[-1][0])
        all_t.append(frames[-1][1])
    gmap = {v: i for i, v in enumerate(np.unique(np.concatenate(all_g)))}
    tmap = {v: i for i, v in enumerate(np.unique(np.concatenate(all_t)))}
    frames = [(np.array([gmap[i] for i in g], int), np.array([tmap[i] for i in d], int), s) for g, d, s in frames]
    return frames, len(gmap), len(tmap)


ALPHAS = [0.05 * i for i in range(1, 20)]  # HOTA localisation thresholds 0.05 .. 0.95


def hota_counts(frames, G, T):
    """HOTA per alpha: TP/FN/FP (detection part) and AssA (association part) for one sequence."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    A = len(ALPHAS)
    tp, fn, fp = np.zeros(A), np.zeros(A), np.zeros(A)
    if G == 0 or T == 0:
        for g, d, _ in frames:
            fn += len(g)
            fp += len(d)
        return dict(tp=tp, fn=fn, fp=fp, assa=np.zeros(A))
    # pass 1: how well does each (gt id, tracker id) pair overlap over the WHOLE sequence -> global alignment score
    potential, g_count, t_count = np.zeros((G, T)), np.zeros((G, 1)), np.zeros((1, T))
    for g, d, sim in frames:
        if len(g) and len(d):
            denom = sim.sum(0)[None] + sim.sum(1)[:, None] - sim
            potential[g[:, None], d[None]] += np.where(denom > 1e-9, sim / np.maximum(denom, 1e-9), 0)
        g_count[g] += 1
        t_count[0, d] += 1
    align = potential / (g_count + t_count - potential)
    # pass 2: per frame Hungarian matching that prefers pairs that are consistent over time
    matches = np.zeros((A, G, T))
    for g, d, sim in frames:
        if len(g) == 0 or len(d) == 0:
            fn += len(g)
            fp += len(d)
            continue
        rows, cols = linear_sum_assignment(-(align[g[:, None], d[None]] * sim))
        for a, alpha in enumerate(ALPHAS):
            ok = sim[rows, cols] >= alpha - 1e-9
            n = ok.sum()
            tp[a] += n
            fn[a] += len(g) - n
            fp[a] += len(d) - n
            matches[a, g[rows[ok]], d[cols[ok]]] += 1
    # association accuracy: for every TP, the Jaccard of its (gt id, tracker id) pair over the sequence
    assa = np.zeros(A)
    for a in range(A):
        jac = matches[a] / np.maximum(1, g_count + t_count - matches[a])
        assa[a] = (matches[a] * jac).sum() / max(1, tp[a])
    return dict(tp=tp, fn=fn, fp=fp, assa=assa)


def clear_counts(frames, G):
    """CLEAR MOT at IoU 0.5: TP/FN/FP, ID switches, fragmentations. A match that keeps last frame's ID is preferred."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tp = fn = fp = idsw = 0
    prev_id = np.full(G, np.nan)  # last tracker id ever matched to each gt id
    prev_step = np.full(G, np.nan)  # tracker id matched in the previous frame (nan = was not tracked)
    frag = np.zeros(G)
    for g, d, sim in frames:
        if len(g) == 0:
            fp += len(d)
            continue
        if len(d) == 0:
            fn += len(g)
            continue
        score = 1000 * (d[None] == prev_step[g[:, None]]) + sim
        score[sim < 0.5 - 1e-9] = 0
        rows, cols = linear_sum_assignment(-score)
        ok = score[rows, cols] > 1e-9
        mg, mt = g[rows[ok]], d[cols[ok]]
        idsw += int((~np.isnan(prev_id[mg]) & (prev_id[mg] != mt)).sum())
        untracked_before = np.isnan(prev_step)
        prev_id[mg] = mt
        prev_step[:] = np.nan
        prev_step[mg] = mt
        frag += untracked_before & ~np.isnan(prev_step)
        tp += len(mg)
        fn += len(g) - len(mg)
        fp += len(d) - len(mg)
    return dict(clr_tp=tp, clr_fn=fn, clr_fp=fp, idsw=idsw, frag=int((frag[frag > 0] - 1).sum()))


def identity_counts(frames, G, T):
    """IDF1: one global gt-id <-> tracker-id assignment that minimises identity FN + FP (IoU 0.5)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_gt = sum(len(g) for g, _, _ in frames)
    n_tr = sum(len(d) for _, d, _ in frames)
    if G == 0 or T == 0:
        return dict(idtp=0, idfn=n_gt, idfp=n_tr)
    potential, g_count, t_count = np.zeros((G, T)), np.zeros(G), np.zeros(T)
    for g, d, sim in frames:
        if len(g) and len(d):
            potential[g[:, None], d[None]] += sim >= 0.5 - 1e-9
        g_count[g] += 1
        t_count[d] += 1
    fn_mat, fp_mat = np.zeros((G + T, G + T)), np.zeros((G + T, G + T))
    fp_mat[G:, :T] = 1e10
    fn_mat[:G, T:] = 1e10
    for i in range(G):
        fn_mat[i, :T] = g_count[i]
        fn_mat[i, T + i] = g_count[i]
    for j in range(T):
        fp_mat[:G, j] = t_count[j]
        fp_mat[G + j, j] = t_count[j]
    fn_mat[:G, :T] -= potential
    fp_mat[:G, :T] -= potential
    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    idfn, idfp = fn_mat[rows, cols].sum(), fp_mat[rows, cols].sum()
    return dict(idtp=n_gt - idfn, idfn=idfn, idfp=idfp)


def score_sequence(seq, tracks):
    frames, G, T = preprocess(seq, tracks)
    c = dict(hota=hota_counts(frames, G, T), n_frames=len(frames), minutes=len(frames) / seq["fps"] / 60)
    c.update(clear_counts(frames, G))
    c.update(identity_counts(frames, G, T))
    return c


def combine(counts):
    """Sum the raw counts of several sequences (AssA is TP-weighted, as in TrackEval's COMBINED_SEQ)."""
    import numpy as np

    tot = {k: sum(c[k] for c in counts) for k in ("clr_tp", "clr_fn", "clr_fp", "idsw", "frag", "idtp", "idfn", "idfp", "n_frames", "minutes")}
    tp = sum(c["hota"]["tp"] for c in counts)
    tot["hota"] = dict(tp=tp, fn=sum(c["hota"]["fn"] for c in counts), fp=sum(c["hota"]["fp"] for c in counts),
                       assa=sum(c["hota"]["assa"] * c["hota"]["tp"] for c in counts) / np.maximum(1, tp))
    return tot


def final_metrics(c):
    """Raw counts -> the numbers in the tables (percent)."""
    import numpy as np

    h = c["hota"]
    deta = h["tp"] / np.maximum(1, h["tp"] + h["fn"] + h["fp"])
    return dict(
        HOTA=round(100 * float(np.mean(np.sqrt(deta * h["assa"]))), 3),
        DetA=round(100 * float(np.mean(deta)), 3),
        AssA=round(100 * float(np.mean(h["assa"])), 3),
        MOTA=round(100 * (c["clr_tp"] - c["clr_fp"] - c["idsw"]) / max(1, c["clr_tp"] + c["clr_fn"]), 3),
        IDF1=round(100 * c["idtp"] / max(1, c["idtp"] + 0.5 * c["idfp"] + 0.5 * c["idfn"]), 3),
        IDSW=int(c["idsw"]),
        IDSW_per_min=round(c["idsw"] / c["minutes"], 2),
        Frag=int(c["frag"]),
        TP=int(c["clr_tp"]), FP=int(c["clr_fp"]), FN=int(c["clr_fn"]),
        frames=int(c["n_frames"]), video_min=round(c["minutes"], 3),
    )


def trackeval_crosscheck(train_dir, seqs, tracks_by_seq, work):
    """Score one run with the PyPI 'trackeval' package too, to prove the formulas above match the reference."""
    import numpy as np
    import trackeval

    tdir = work / "te" / "p2" / "data"
    tdir.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        lines = [f"{t},{int(r[0])},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},1,-1,-1,-1" for t, a in sorted(tracks_by_seq[s["name"]].items()) for r in a]
        (tdir / f"{s['dir'].name}.txt").write_text("\n".join(lines))
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update(USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False, OUTPUT_SUMMARY=False,
                OUTPUT_DETAILED=False, PLOT_CURVES=False, OUTPUT_EMPTY_CLASSES=False, DISPLAY_LESS_PROGRESS=True)
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=str(train_dir), TRACKERS_FOLDER=str(work / "te"), TRACKERS_TO_EVAL=["p2"], SKIP_SPLIT_FOL=True,
                SEQ_INFO={s["dir"].name: None for s in seqs}, BENCHMARK="MOT17", SPLIT_TO_EVAL="train", PRINT_CONFIG=False)
    res, _ = trackeval.Evaluator(ecfg).evaluate([trackeval.datasets.MotChallenge2DBox(dcfg)],
                                                [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    c = res["MotChallenge2DBox"]["p2"]["COMBINED_SEQ"]["pedestrian"]
    return dict(HOTA=round(100 * float(np.mean(c["HOTA"]["HOTA"])), 3), DetA=round(100 * float(np.mean(c["HOTA"]["DetA"])), 3),
                AssA=round(100 * float(np.mean(c["HOTA"]["AssA"])), 3), MOTA=round(100 * float(c["CLEAR"]["MOTA"]), 3),
                IDF1=round(100 * float(c["Identity"]["IDF1"]), 3), IDSW=int(c["CLEAR"]["IDSW"]), Frag=int(c["CLEAR"]["Frag"]))


# ----------------------------------------------------------------------------------------------- tracking


def make_tracker():
    """A fresh ByteTrack with ultralytics' stock bytetrack.yaml (same settings for every variant)."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    try:
        from ultralytics.utils import YAML

        cfg = YAML.load(check_yaml("bytetrack.yaml"))
    except ImportError:  # older ultralytics
        from ultralytics.utils import yaml_load

        cfg = yaml_load(check_yaml("bytetrack.yaml"))
    return BYTETracker(args=IterableSimpleNamespace(**cfg)), cfg


def coast(tracker, last):
    """A frame WITHOUT detection: Kalman predict only, output the predicted boxes of the live tracks."""
    global SKIP_MODE
    import numpy as np

    if SKIP_MODE == "kalman":
        try:
            tracker.multi_predict(list(tracker.tracked_stracks) + list(tracker.lost_stracks))
            tracker.frame_id += 1  # keeps track_buffer counted in video frames
            live = [t for t in tracker.tracked_stracks if t.is_activated]
            return np.array([[*t.xyxy, t.track_id] for t in live], dtype=float).reshape(-1, 5)
        except Exception as e:
            SKIP_MODE = "hold"
            log(f"Kalman-only step failed ({e}); falling back to repeating the last boxes")
    return last


def run_sequence(model, seq, conf, every):
    """Detector + ByteTrack over one sequence. Returns tracks {frame: rows (id, x, y, w, h)} and per-frame timings."""
    import cv2
    import numpy as np
    import psutil

    tracker, _ = make_tracker()
    proc = psutil.Process()
    tracks, tm = {}, dict(pre=[], inf=[], post=[], track=[])
    last = np.zeros((0, 5))
    rss = proc.memory_info().rss
    wall0 = time.perf_counter()
    for k, path in enumerate(seq["frames"]):
        img = cv2.imread(str(path))
        if k % every == 0:
            r = model.predict(img, imgsz=IMGSZ, conf=conf, classes=[0], device="cpu", verbose=False)[0]
            for key, name in (("pre", "preprocess"), ("inf", "inference"), ("post", "postprocess")):
                tm[key].append(float(r.speed[name]))
            t = time.perf_counter()
            out = tracker.update(r.boxes.cpu().numpy(), img)  # rows: x1, y1, x2, y2, track id, score, class, det index
            last = np.asarray(out, dtype=float)[:, :5] if len(out) else np.zeros((0, 5))
        else:
            t = time.perf_counter()
            last = coast(tracker, last)
        tm["track"].append((time.perf_counter() - t) * 1e3)
        if len(last):
            x1, y1, x2, y2, tid = last.T
            tracks[k + 1] = np.stack([tid, x1, y1, x2 - x1, y2 - y1], axis=1)
        if k % 50 == 0:
            rss = max(rss, proc.memory_info().rss)
    tm["wall_s"] = time.perf_counter() - wall0
    tm["rss"] = max(rss, proc.memory_info().rss)
    return tracks, tm


def speed_stats(tms):
    """Timing columns from per-frame lists (pooled over sequences). pre/inf/post: detection frames only."""
    import numpy as np

    row, total_ms, n = {}, 0.0, sum(len(t["track"]) for t in tms)
    for key in ("pre", "inf", "post", "track"):
        v = np.concatenate([np.asarray(t[key], dtype=float) for t in tms])
        row[f"{key}_ms_med"] = round(float(np.median(v)), 3)
        row[f"{key}_ms_p90"] = round(float(np.percentile(v, 90)), 3)
        row[f"{key}_ms_per_frame"] = round(float(v.sum()) / n, 3)  # averaged over ALL frames (skipped frames cost 0)
        total_ms += float(v.sum())
    row["fps_e2e"] = round(n / (total_ms / 1e3), 2)  # pre + inference + post + track
    row["fps_wall"] = round(n / sum(t["wall_s"] for t in tms), 2)  # also includes reading the JPEG from disk
    row["rss_peak_mb"] = round(max(t["rss"] for t in tms) / 2**20)
    return row


def run_config(model, seqs, variant, conf, every, base):
    """One grid point = all test sequences. Returns rows (one per sequence + COMBINED) and the raw tracks."""
    counts, tms, rows, tracks_by_seq = [], [], [], {}
    for s in seqs:
        tracks, tm = run_sequence(model, s, conf, every)
        c = score_sequence(s, tracks)
        counts.append(c)
        tms.append(tm)
        tracks_by_seq[s["name"]] = tracks
        rows.append(dict(base, variant=variant, conf=conf, det_every=every, seq=s["name"], skip_mode=SKIP_MODE if every > 1 else "-",
                         **final_metrics(c), **speed_stats([tm])))
    rows.append(dict(base, variant=variant, conf=conf, det_every=every, seq="COMBINED", skip_mode=SKIP_MODE if every > 1 else "-",
                     **final_metrics(combine(counts)), **speed_stats(tms)))
    return rows, tracks_by_seq


# ----------------------------------------------------------------------------------------------- outputs


def save_results(out, rows, meta):
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "p2_results.csv", index=False)
    (out / "p2_results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=str))


def combined(rows, variant=None, every=None, conf=None):
    return [r for r in rows if r["seq"] == "COMBINED" and (variant is None or r["variant"] == variant)
            and (every is None or r["det_every"] == every) and (conf is None or r["conf"] == conf)]


def best_row(rows, variant):
    """The variant's best threshold = highest combined HOTA at detect-every-1."""
    c = combined(rows, variant, every=1)
    return max(c, key=lambda r: r["HOTA"]) if c else None


def plot(out, rows, meta):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [v for v in VARIANTS if combined(rows, v)]
    title = f"{meta['model']} on {meta.get('cpu', '?')} ({meta.get('vcpus', '?')} vCPU)"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("HOTA", "IDF1")):
        for v in variants:
            c = sorted(combined(rows, v, every=1), key=lambda r: r["conf"])
            ax.plot([r["conf"] for r in c], [r[metric] for r in c], marker="o", label=LABEL[v])
        ax.set(xlabel="detector confidence threshold", ylabel=f"{metric} (3 sequences combined)", title=f"{metric} vs threshold")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "p2_hota_vs_threshold.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for v in variants:
        b = best_row(rows, v)
        c = sorted(combined(rows, v, conf=b["conf"]), key=lambda r: r["det_every"])
        for ax, metric in zip(axes, ("HOTA", "IDSW_per_min", "fps_e2e")):
            ax.plot([r["det_every"] for r in c], [r[metric] for r in c], marker="o", label=f"{LABEL[v]} (conf {b['conf']})")
    for ax, name in zip(axes, ("HOTA", "ID switches per minute", "end-to-end FPS")):
        ax.set(xlabel="detect every N frames", ylabel=name, title=f"{name} vs N")
        ax.set_xticks(EVERY)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "p2_detect_every_n.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(variants)
    best = [best_row(rows, v) for v in variants]
    for key, name in (("pre", "pre-process"), ("inf", "inference"), ("post", "post-process"), ("track", "ByteTrack")):
        vals = [b[f"{key}_ms_per_frame"] for b in best]
        ax.bar([LABEL[v] for v in variants], vals, bottom=bottom, label=name)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set(ylabel="ms per frame (mean, detect-every-1)", title=f"Where the time goes: {title}")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "p2_speed_split.png", dpi=130)
    plt.close(fig)


def md_table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in body] + [""]


def write_report(out, rows, meta):
    """Markdown report. Every number is read from `rows` / `meta`; nothing is typed in by hand."""
    variants = [v for v in VARIANTS if combined(rows, v)]
    best = {v: best_row(rows, v) for v in variants}
    ex = meta.get("exports", {})
    L = [f"# TrackQuant-OV: {meta['model']} + ByteTrack on MOT17 (auto-generated)", ""]
    L += [f"- CPU: {meta.get('cpu')} x{meta.get('vcpus')} vCPU, {meta.get('ram_gb')} GB RAM, ISA flags: {', '.join(meta.get('isa', [])) or 'none of avx2/avx512/vnni/amx found'}",
          f"- Versions: {meta.get('versions')}",
          f"- Data: {meta.get('data_source')} ({meta.get('data_download_gb', '-')} GB in {meta.get('data_download_s', '-')} s). Test sequences: {', '.join(meta['test_seqs'])} (FRCNN folder, images + gt only).",
          f"- INT8 calibration: {meta.get('calib_frames', '-')} frames of {CALIB_SEQ} (not a test sequence), NNCF post-training quantization through ultralytics' OpenVINO export.",
          f"- Input: {IMGSZ[1]}x{IMGSZ[0]} (imgsz 640, rectangular) for all variants. Tracker: ByteTrack, ultralytics bytetrack.yaml unchanged: {meta.get('tracker_cfg')}.",
          "- The swept threshold is the detector confidence filter applied BEFORE the tracker. ByteTrack's own thresholds stay fixed, so boxes under its track_high_thresh only feed its second (low-score) association.",
          f"- Detect-every-N: frames without detection use '{SKIP_MODE}' (kalman = Kalman predict only, predicted boxes keep their IDs; hold = repeat the last boxes).",
          "- Metrics: own implementation of the TrackEval formulas with MOTChallenge pre-processing (class 1, consider-flag 1, distractor matches removed). COMBINED = counts summed over the sequences.",
          "- Timings are comparable only inside this kernel run. pre/inference/post = ultralytics profilers; track = ByteTrack update; JPEG decoding is excluded from fps_e2e.",
          f"- Status: {meta.get('status', 'running')}; {len(combined(rows))} grid points done; elapsed {meta.get('elapsed_s', '-')} s.", ""]
    if meta.get("skipped"):
        L += ["**Not run (time budget or error):** " + "; ".join(meta["skipped"]), ""]
    if meta.get("smoke"):
        L += ["**SMOKE RUN: 60 frames per sequence, reduced grid. Numbers are a code-path check, not results.**", ""]

    L += ["## Exported models", ""]
    L += md_table(["variant", "export s", "size MB", "export args", "INFERENCE_PRECISION_HINT", "layers by runtime precision"],
                  [[LABEL[v], e.get("export_s", "-"), e.get("size_mb", "-"), e.get("kwargs", "-"), e.get("precision_hint", "-"), e.get("runtime_precisions", "-")]
                   for v, e in ex.items()])
    if not variants:
        (out / "p2_report.md").write_text("\n".join(L + ["No tracking run finished."]))
        return

    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "IDSW_per_min", "Frag", "fps_e2e"]
    L += ["## 1. Tracking quality at each variant's best threshold (detect-every-1, 3 sequences combined)", ""]
    L += md_table(["variant", "best conf"] + cols, [[LABEL[v], best[v]["conf"]] + [best[v][c] for c in cols] for v in variants])
    ref = "ov_fp32" if "ov_fp32" in best else variants[0]
    L += [f"Difference to {LABEL[ref]} (each at its own best threshold):", ""]
    L += md_table(["variant", "dHOTA", "dDetA", "dAssA", "dMOTA", "dIDF1", "dIDSW/min", "speed-up (fps_e2e)"],
                  [[LABEL[v]] + [f"{best[v][c] - best[ref][c]:+.2f}" for c in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW_per_min")]
                   + [f"{best[v]['fps_e2e'] / best[ref]['fps_e2e']:.2f}x"] for v in variants if v != ref])

    L += ["## 2. Threshold drift", "", f"Full sweep (HOTA / IDF1 per threshold, detect-every-1). Reference = {LABEL[ref]}.", ""]
    confs = sorted({r["conf"] for r in combined(rows, every=1)})
    body = []
    for v in variants:
        by = {r["conf"]: r for r in combined(rows, v, every=1)}
        body.append([LABEL[v]] + [f"{by[c]['HOTA']:.2f} / {by[c]['IDF1']:.2f}" + (" **best**" if c == best[v]["conf"] else "") if c in by else "-" for c in confs])
    L += md_table(["variant"] + [f"conf {c}" for c in confs], body)
    body = []
    for v in variants:
        if v == ref:
            continue
        at_ref = combined(rows, v, every=1, conf=best[ref]["conf"])
        if not at_ref:
            continue
        a = at_ref[0]
        body.append([LABEL[v], best[ref]["conf"], best[v]["conf"], "yes" if best[v]["conf"] != best[ref]["conf"] else "no",
                     a["HOTA"], best[v]["HOTA"], f"{best[v]['HOTA'] - a['HOTA']:+.2f}", a["IDF1"], best[v]["IDF1"], f"{best[v]['IDF1'] - a['IDF1']:+.2f}",
                     a["IDSW_per_min"], best[v]["IDSW_per_min"]])
    L += [f"What happens if {LABEL[ref]}'s best threshold is reused as-is, versus re-tuning the threshold for the variant:", ""]
    L += md_table(["variant", "ref best conf", "own best conf", "drifted?", "HOTA @ref conf", "HOTA @own conf", "gain from re-tuning",
                   "IDF1 @ref conf", "IDF1 @own conf", "gain", "IDSW/min @ref conf", "IDSW/min @own conf"], body)
    L += ["Note: the grid step is 0.1, so 'no drift' means the optimum moved by less than one grid step.", ""]

    L += ["## 3. Detect every N frames (at each variant's best threshold)", ""]
    body = []
    for v in variants:
        for r in sorted(combined(rows, v, conf=best[v]["conf"]), key=lambda r: r["det_every"]):
            body.append([LABEL[v], r["conf"], r["det_every"], r["HOTA"], r["DetA"], r["AssA"], r["IDF1"], r["IDSW"], r["IDSW_per_min"], r["Frag"], r["fps_e2e"], r["fps_wall"]])
    L += md_table(["variant", "conf", "N", "HOTA", "DetA", "AssA", "IDF1", "IDSW", "IDSW/min", "Frag", "fps_e2e", "fps_wall"], body)

    L += ["## 4. Speed split (detect-every-1, best threshold; ms per frame, median / p90)", ""]
    L += md_table(["variant", "pre-process", "inference", "post-process", "ByteTrack", "fps_e2e", "fps_wall (incl. JPEG read)", "peak RSS MB"],
                  [[LABEL[v]] + [f"{best[v][k + '_ms_med']:.2f} / {best[v][k + '_ms_p90']:.2f}" for k in ("pre", "inf", "post", "track")]
                   + [best[v]["fps_e2e"], best[v]["fps_wall"], best[v]["rss_peak_mb"]] for v in variants])
    L += ["Post-process = box decoding" + (" only: the YOLO26 head is NMS-free (end-to-end)." if meta["model"].startswith("yolo26") else " + NMS (YOLOv8)."),
          "Post-process time depends on the threshold (lower threshold = more candidate boxes), so compare it at the listed thresholds only.",
          "Peak RSS: all variants run one after another in the same process, so it is a running maximum, not a per-variant footprint.", ""]

    L += ["## 5. Per-sequence table in the style of Intel ENAB #3798 (MOTA FP32 vs INT8), extended with HOTA / IDF1 / IDSW per minute", ""]
    body = []
    for s in meta["test_seqs"] + ["COMBINED"]:
        for v in variants:
            r = [x for x in rows if x["variant"] == v and x["seq"] == s and x["det_every"] == 1 and x["conf"] == best[v]["conf"]]
            if r:
                r = r[0]
                body.append([s, LABEL[v], r["conf"], r["MOTA"], r["HOTA"], r["IDF1"], r["IDSW"], r["IDSW_per_min"], r["fps_e2e"], ex.get(v, {}).get("size_mb", "-")])
    L += md_table(["sequence", "precision", "conf", "MOTA", "HOTA", "IDF1", "IDSW", "IDSW/min", "FPS", "model MB"], body)

    if meta.get("trackeval_check"):
        L += ["## Evaluator cross-check", "", f"One run ({meta['trackeval_check']['run']}) scored with this script and with PyPI trackeval {meta['versions'].get('trackeval')}:", ""]
        mine, ref_te = meta["trackeval_check"]["this_script"], meta["trackeval_check"]["trackeval"]
        L += md_table(["metric", "this script", "trackeval"], [[k, mine[k], ref_te[k]] for k in ref_te])
    elif meta.get("trackeval_error"):
        L += ["## Evaluator cross-check", "", f"Not done: {meta['trackeval_error']}", ""]
    L += ["Plots: p2_hota_vs_threshold.png, p2_detect_every_n.png, p2_speed_split.png. Raw rows: p2_results.csv / p2_results.json.", ""]
    (out / "p2_report.md").write_text("\n".join(L))


def finish(out, rows, meta, status=None):
    if status:
        meta["status"] = status
    meta["elapsed_s"] = round(time.time() - T0)
    save_results(out, rows, meta)
    try:
        write_report(out, rows, meta)
        if combined(rows):
            plot(out, rows, meta)
    except Exception as e:
        log(f"report/plot failed: {e!r}")


# ----------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=["yolo26n", "yolov8n"])
    ap.add_argument("--smoke", action="store_true", help="60 frames per sequence, reduced grid (code-path check)")
    ap.add_argument("--data-dir", help="existing MOT17 'train' folder (skips the download)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--budget-min", type=float, default=BUDGET_S / 60, help="hard stop for the whole run")
    args, _ = ap.parse_known_args()  # Kaggle may pass its own argv

    on_kaggle = os.path.isdir("/kaggle/working")
    if not on_kaggle and not args.smoke:
        raise SystemExit("the full grid only runs on Kaggle; use --smoke locally")
    out = Path(args.out or ("/kaggle/working" if on_kaggle else "p2_out"))
    work = Path("/kaggle/tmp/p2" if on_kaggle else out / "work")  # scratch: data + models, not kept as kernel output
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    deadline = T0 + args.budget_min * 60 - RESERVE_S

    if on_kaggle:
        if not pip(*PINS):
            log("pinned install failed, retrying without pins")
            pip(*NOPIN)
        pip("--no-deps", "trackeval==1.3.0")  # optional cross-check only; --no-deps so it cannot move numpy

    meta = dict(model=args.model, smoke=args.smoke, test_seqs=TEST_SEQS, calib_seq=CALIB_SEQ, imgsz=IMGSZ, status="running",
                exports={}, skipped=[], started=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    write_env(out, meta)
    rows = []
    finish(out, rows, meta)

    # ---- data
    train_dir = get_mot17(args, work, meta)
    nf = 60 if args.smoke else 0
    seqs = [load_sequence(train_dir, s, nf) for s in TEST_SEQS]
    for s in seqs:
        log(f"{s['name']}: {len(s['frames'])} frames, {s['fps']} fps, {s['w']}x{s['h']}")
    calib_yaml, n_calib = make_calib_dataset(load_sequence(train_dir, CALIB_SEQ, nf), work, 20 if args.smoke else CALIB_FRAMES)
    meta["calib_frames"] = n_calib
    _, meta["tracker_cfg"] = make_tracker()

    # ---- models: PyTorch weights, then the three OpenVINO exports
    from ultralytics import YOLO

    os.chdir(work)  # ultralytics downloads the weights into the current folder
    weights = f"{args.model}.pt"
    YOLO(weights)
    paths = {"pt_fp32": str(work / weights)}
    meta["exports"]["pt_fp32"] = dict(size_mb=dir_size_mb(work / weights), kwargs="PyTorch reference (no export)")
    for v in ("ov_fp32", "ov_fp16", "ov_int8"):
        try:
            paths[v], secs, mb, kw = export_variant(v, work / weights, work, calib_yaml, args.model)
            meta["exports"][v] = dict(export_s=secs, size_mb=mb, kwargs=str(kw))
            try:
                counts, hint = runtime_precisions(paths[v])
                meta["exports"][v].update(runtime_precisions=str(counts), precision_hint=hint)
            except Exception as e:
                log(f"runtime precision table for {v} failed: {e!r}")
        except Exception as e:
            log(f"EXPORT {v} FAILED: {e!r}")
            meta["skipped"].append(f"{v}: export failed ({e!r})")
        finish(out, rows, meta)

    # ---- the grid
    confs = [0.1, 0.3] if args.smoke else CONFS
    everys = [1, 2] if args.smoke else EVERY
    base = dict(model=args.model)
    models = {}

    def load(v):
        """Load a variant and warm it up, so the first timed frame does not pay for compilation."""
        import cv2

        if v not in models:
            models.clear()
            gc.collect()
            models[v] = YOLO(paths[v], task="detect")
            img = cv2.imread(str(seqs[0]["frames"][0]))
            for _ in range(10):
                models[v].predict(img, imgsz=IMGSZ, conf=0.25, classes=[0], device="cpu", verbose=False)
        return models[v]

    def run(v, conf, every):
        """One grid point with the budget guard. Returns False when the time is up."""
        if time.time() > deadline:
            meta["skipped"].append(f"{v} conf={conf} N={every}: time budget")
            return False
        t = time.time()
        try:
            new_rows, tracks = run_config(load(v), seqs, v, conf, every, base)
        except Exception as e:
            log(f"RUN {v} conf={conf} N={every} FAILED: {e!r}")
            meta["skipped"].append(f"{v} conf={conf} N={every}: {e!r}")
            return True
        rows.extend(new_rows)
        c = new_rows[-1]
        log(f"{v} conf={conf} N={every}: HOTA {c['HOTA']} MOTA {c['MOTA']} IDF1 {c['IDF1']} IDSW/min {c['IDSW_per_min']} "
            f"fps {c['fps_e2e']} ({time.time() - t:.0f} s)")
        if "trackeval_check" not in meta and "trackeval_error" not in meta and not args.smoke:
            try:  # once, on the first finished run
                keys = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag")
                meta["trackeval_check"] = dict(run=f"{v} conf={conf} N={every}", this_script={k: c[k] for k in keys},
                                               trackeval=trackeval_crosscheck(train_dir, seqs, tracks, work))
                log(f"trackeval cross-check: {meta['trackeval_check']}")
            except Exception as e:
                meta["trackeval_error"] = repr(e)
                log(f"trackeval cross-check skipped: {e!r}")
        finish(out, rows, meta)
        return True

    todo = [v for v in VARIANTS if v in paths]
    alive = True
    for v in todo:  # phase A: threshold sweep at detect-every-1
        for conf in confs:
            alive = alive and run(v, conf, 1)
    for v in todo:  # phase B: detect-every-N at the variant's best threshold
        b = best_row(rows, v)
        for every in everys[1:]:
            if b is None:
                meta["skipped"].append(f"{v} N={every}: no threshold sweep result")
            else:
                alive = alive and run(v, b["conf"], every)

    finish(out, rows, meta, status="complete" if not meta["skipped"] else "partial (see 'Not run')")
    log(f"done: {len(combined(rows))} grid points, status {meta['status']}, outputs in {out}")


# ---- END src/trackquant/bench.py ----

import logging

FU_MODEL = "yolo26n"
FU_BUDGET_S = 75 * 60  # hard stop for the whole kernel
FU_RESERVE_S = 240  # kept for the report
FU_VARIANTS = ["pt_fp32", "ov_fp32", "ov_int8"]
RAW_CONF = 0.01  # the detector pass keeps every box above this
LOW_CONFS = [0.01, 0.03, 0.05, 0.1]
DIFF_SEQS = ["MOT17-02", "MOT17-04"]  # part 2: where the ID switch gap was (55 vs 32), and the most crowded one
FLOORS = [0.01, 0.1, 0.25]  # score floors for the box comparison (0.1 / 0.25 = ByteTrack low / high threshold)
TRACK_CONF = 0.1  # part 2 tracking threshold (the best one of the first run)
MAX_DET = 300  # ultralytics default
SQUARE = [640, 640]
EXPORTS2 = [  # name, export kwargs, PyTorch predict kwargs, OpenVINO predict kwargs, key of the PyTorch pass
    ("a: nms=False, 384x640 (current)", dict(nms=False, imgsz=IMGSZ), dict(imgsz=IMGSZ), dict(imgsz=IMGSZ), "rect"),
    ("b: nms=True, 384x640", dict(nms=True, imgsz=IMGSZ), dict(imgsz=IMGSZ), dict(imgsz=IMGSZ), "rect"),
    ("c: nms=False, 640x640 letterbox", dict(nms=False, imgsz=SQUARE), dict(imgsz=SQUARE, rect=False), dict(imgsz=SQUARE), "square"),
    ("d: nms=False, dynamic=True", dict(nms=False, imgsz=IMGSZ, dynamic=True), dict(imgsz=IMGSZ), dict(imgsz=IMGSZ), "rect"),
]
SAME_KEYS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "TP", "FP", "FN")


# ----------------------------------------------------------------------------------------------- detection


def fu_load(path, img, kw):
    """Load a model and warm it up with the predict arguments it will be used with."""
    from ultralytics import YOLO

    m = YOLO(path, task="detect")
    for _ in range(5):
        m.predict(img, conf=0.25, device="cpu", verbose=False, **kw)
    return m


def detect_pass(model, seq, kw, max_det=MAX_DET, max_frames=0):
    """Detector only, conf RAW_CONF, all classes. Keeps the person boxes (x1, y1, x2, y2, score, class) per frame
    and, for the cap question, how many boxes of ANY class came back and the lowest score among them."""
    import cv2
    import numpy as np

    c = dict(dets=[], n_all=[], min_all=[], pre=[], inf=[], post=[])
    t0 = time.perf_counter()
    for path in seq["frames"][:max_frames] if max_frames else seq["frames"]:
        r = model.predict(cv2.imread(str(path)), conf=RAW_CONF, max_det=max_det, device="cpu", verbose=False, **kw)[0]
        d = r.boxes.data.cpu().numpy().astype(np.float32)[:, :6]
        c["n_all"].append(len(d))
        c["min_all"].append(float(d[:, 4].min()) if len(d) else float("nan"))
        c["dets"].append(d[d[:, 5] == 0])
        for key, name in (("pre", "preprocess"), ("inf", "inference"), ("post", "postprocess")):
            c[key].append(float(r.speed[name]))
    c["wall_s"] = time.perf_counter() - t0
    return c


def pass_speed(passes):
    import numpy as np

    s = {f"{k}_ms_med": round(float(np.median(np.concatenate([p[k] for p in passes]))), 3) for k in ("pre", "inf", "post")}
    s["fps_detector_wall"] = round(sum(len(p["dets"]) for p in passes) / sum(p["wall_s"] for p in passes), 2)
    return s


def cap_stats(c):
    """Does the 300-box cap bite? Frames that returned MAX_DET boxes, and the lowest score kept in those frames."""
    import numpy as np

    n, mins = np.array(c["n_all"]), np.array(c["min_all"])
    hit = n >= MAX_DET
    return dict(frames=len(n), boxes_all_classes_mean=round(float(n.mean()), 1), boxes_all_classes_max=int(n.max()),
                frames_at_cap=int(hit.sum()),
                lowest_kept_score_at_cap_median=round(float(np.median(mins[hit])), 4) if hit.any() else None,
                lowest_kept_score_at_cap_max=round(float(mins[hit].max()), 4) if hit.any() else None,
                frames_at_cap_with_lowest_score_over_0p1=int((mins[hit] > 0.1).sum()),
                person_boxes_mean=round(float(np.mean([len(d) for d in c["dets"]])), 1),
                person_boxes_over_0p1_mean=round(float(np.mean([(d[:, 4] > 0.1).sum() for d in c["dets"]])), 1))


def track_cached(seq, dets, conf, low=None):
    """ByteTrack over kept detections: boxes with score > conf, as predict(conf=conf, classes=[0]) would return."""
    import numpy as np
    from ultralytics.engine.results import Boxes

    tracker, _ = make_tracker()
    if low is not None:
        tracker.args.track_low_thresh = low
    tracks, ms = {}, []
    for k, d in enumerate(dets):
        t = time.perf_counter()
        out = tracker.update(Boxes(d[d[:, 4] > conf], (seq["h"], seq["w"])), None)
        ms.append((time.perf_counter() - t) * 1e3)
        if len(out):
            x1, y1, x2, y2, tid = np.asarray(out, dtype=float)[:, :5].T
            tracks[k + 1] = np.stack([tid, x1, y1, x2 - x1, y2 - y1], axis=1)
    return tracks, ms


def tracked_rows(seqs, cache, conf, low, base):
    """One tracker setting over the kept detections of several sequences -> rows per sequence + COMBINED."""
    import numpy as np

    counts, rows, ms = [], [], []
    for s in seqs:
        tracks, m = track_cached(s, cache[s["name"]]["dets"], conf, low)
        counts.append(score_sequence(s, tracks))
        ms += m
        rows.append(dict(base, conf=conf, seq=s["name"], **final_metrics(counts[-1]), track_ms_med=round(float(np.median(m)), 3)))
    rows.append(dict(base, conf=conf, seq="COMBINED", **final_metrics(combine(counts)), track_ms_med=round(float(np.median(ms)), 3)))
    return rows


def tracks_equal(a, b):
    import numpy as np

    if set(a) != set(b) or any(a[t].shape != b[t].shape for t in a):
        return False, None
    worst = max([float(np.abs(a[t] - b[t]).max()) for t in a if len(a[t])] or [0.0])
    return worst == 0.0, worst


def match_stats(P, O, floor):
    """PyTorch boxes P vs OpenVINO boxes O (lists per frame), person boxes with score > floor. One-to-one matching
    that maximises the summed IoU over pairs with IoU > 0.5; a pair counts as matched when its IoU > 0.9."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    xywh = lambda d: np.stack([d[:, 0], d[:, 1], d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]], axis=1)
    n = min(len(P), len(O))
    n_pt = n_ov = m90 = m50 = bad = 0
    ds, dc, ious = [np.zeros(0)], [np.zeros(0)], [np.zeros(0)]
    for t in range(n):
        p, o = P[t][P[t][:, 4] > floor], O[t][O[t][:, 4] > floor]
        n_pt += len(p)
        n_ov += len(o)
        k = 0
        if len(p) and len(o):
            iou = iou_matrix(xywh(p), xywh(o))
            rows, cols = linear_sum_assignment(-np.where(iou > 0.5, iou, 0.0))
            v = iou[rows, cols]
            ok = v > 0.9
            k = int(ok.sum())
            m50 += int((v > 0.5).sum())
            ds.append(np.abs(p[rows[ok], 4] - o[cols[ok], 4]))
            dc.append(np.abs(p[rows[ok], :4] - o[cols[ok], :4]).max(1))
            ious.append(v[ok])
        m90 += k
        bad += k < max(len(p), len(o))
    ds, dc, ious = np.concatenate(ds), np.concatenate(dc), np.concatenate(ious)
    f = lambda x, fn, nd: round(float(fn(x)), nd) if len(x) else None
    return dict(frames=n, pt_boxes_per_frame=round(n_pt / n, 2), ov_boxes_per_frame=round(n_ov / n, 2),
                matched_per_frame=round(m90 / n, 2), pt_only_per_frame=round((n_pt - m90) / n, 3),
                ov_only_per_frame=round((n_ov - m90) / n, 3),
                pt_only_iou50_per_frame=round((n_pt - m50) / n, 3), ov_only_iou50_per_frame=round((n_ov - m50) / n, 3),
                frames_with_unmatched=int(bad), score_absdiff_mean=f(ds, np.mean, 5), score_absdiff_max=f(ds, np.max, 5),
                coord_absdiff_mean_px=f(dc, np.mean, 3), coord_absdiff_max_px=f(dc, np.max, 3), iou_matched_mean=f(ious, np.mean, 5))


# ----------------------------------------------------------------------------------------------- export + input checks


class WarnCatcher(logging.Handler):
    """Keeps ultralytics' WARNING lines of an export (e.g. an argument that was overridden)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage()[:300])


def export_custom(name, weights, work, kw):
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    d = work / "models" / ("fu_" + name[0])  # own folder per export
    d.mkdir(parents=True, exist_ok=True)
    w = d / Path(weights).name
    shutil.copy(weights, w)
    h = WarnCatcher()
    LOGGER.addHandler(h)
    t = time.time()
    try:
        path = YOLO(str(w)).export(format="openvino", batch=1, device="cpu", **kw)
    finally:
        LOGGER.removeHandler(h)
    return str(path), round(time.time() - t, 1), h.msgs


def ov_model_info(model_dir):
    """What was really exported: input / output shapes, NMS and TopK nodes in the graph, ultralytics' metadata."""
    import openvino as ov
    import yaml

    m = ov.Core().read_model(str(next(Path(model_dir).glob("*.xml"))))
    types = [op.get_type_name() for op in m.get_ops()]
    info = dict(inputs=[str(i.get_partial_shape()) for i in m.inputs], outputs=[str(o.get_partial_shape()) for o in m.outputs],
                nms_nodes=sum("nms" in t.lower() or "nonmax" in t.lower() for t in types), topk_nodes=types.count("TopK"))
    try:
        md = yaml.safe_load((Path(model_dir) / "metadata.yaml").read_text())
        info["metadata"] = {k: md[k] for k in ("imgsz", "end2end", "dynamic") if k in md}
        info["metadata_args"] = {k: v for k, v in (md.get("args") or {}).items() if k in ("nms", "dynamic", "imgsz", "half", "int8", "end2end")}
    except Exception as e:
        info["metadata"] = f"<{e!r}>"
    return info


def describe(x, depth=0):
    if hasattr(x, "shape"):
        return str(tuple(x.shape))
    if isinstance(x, (list, tuple)) and depth < 2:
        return "[" + ", ".join(describe(v, depth + 1) for v in x) + "]"
    if isinstance(x, dict) and depth < 2:
        return "{" + ", ".join(f"{k}: {describe(v, depth + 1)}" for k, v in x.items()) + "}"
    return type(x).__name__


def pre_tensor(model, img, kw):
    """The exact tensor ultralytics feeds the network for `img` with these predict arguments, + the raw output shape."""
    model.predict(img, conf=RAW_CONF, device="cpu", verbose=False, **kw)  # sets up the predictor for these arguments
    p = model.predictor
    im = p.preprocess([img])
    try:
        raw = describe(p.inference(im))
    except Exception as e:
        raw = f"<{e!r}>"
    return im.detach().cpu().float().numpy(), raw


def pre_diff(a, b):
    """Pixel difference of two pre-processed tensors (values are 0..1; reported in 1/255 steps)."""
    import numpy as np

    if a.shape != b.shape:
        return dict(same_shape=False)
    d = np.abs(a - b) * 255
    return dict(same_shape=True, max_absdiff_255=round(float(d.max()), 4), mean_absdiff_255=round(float(d.mean()), 6),
                values_differing=int((d > 0).sum()), values_total=int(d.size))


def maxdet_probe(model, seq, kw, n=20):
    """Boxes per frame (all classes, conf RAW_CONF) with max_det 300 vs 1000 on the first frames of a crowded sequence."""
    import numpy as np

    res = {}
    for md in (MAX_DET, 1000):
        try:
            c = detect_pass(model, seq, kw, max_det=md, max_frames=n)
            res[f"max_det={md}"] = dict(frames=len(c["n_all"]), boxes_mean=round(float(np.mean(c["n_all"])), 1), boxes_max=int(max(c["n_all"])))
        except Exception as e:
            res[f"max_det={md}"] = f"<{e!r}>"
    return res


# ----------------------------------------------------------------------------------------------- outputs


def p1_rows(rows, variant, arm, stock_low):
    """{conf: COMBINED row} of one variant and tracker arm. In arm "low=conf" the settings with conf >= the stock
    track_low_thresh are the stock rows (identical by construction, not run twice)."""
    by = {}
    for r in rows:
        if r["part"] == "1" and r["seq"] == "COMBINED" and r["variant"] == variant:
            if r["arm"] == arm or (arm != "stock" and r["arm"] == "stock" and r["conf"] >= stock_low):
                by[r["conf"]] = r
    return dict(sorted(by.items()))


def fu_report(out, rows, meta):
    """Markdown report. Every number and every yes/no is computed from `rows` / `meta` of this run."""
    cfg = meta.get("tracker_cfg") or {}
    stock_low = cfg.get("track_low_thresh", 0.1)
    L = [f"# TrackQuant-OV follow-up: {meta['model']} + ByteTrack on MOT17 (auto-generated)", ""]
    L += [f"- CPU: {meta.get('cpu')} x{meta.get('vcpus')} vCPU, {meta.get('ram_gb')} GB RAM, ISA flags: {', '.join(meta.get('isa', [])) or '-'}",
          f"- Versions: {meta.get('versions')}",
          f"- Data: {meta.get('data_source')}. Sequences: {', '.join(meta['test_seqs'])}. INT8 calibration: {meta.get('calib_frames', '-')} frames of {CALIB_SEQ}.",
          f"- Tracker: ByteTrack, ultralytics bytetrack.yaml: {cfg}.",
          f"- Method: the detector runs once per model and sequence at conf {RAW_CONF} (all classes, max_det {MAX_DET}); each threshold "
          "is then applied to the kept person boxes (score > conf) before ByteTrack. See 'Check of the method' below.",
          f"- Status: {meta.get('status', 'running')}; elapsed {meta.get('elapsed_s', '-')} s of a {meta.get('budget_s')} s budget.", ""]
    if meta.get("skipped"):
        L += ["**Not run (time budget or error):** " + "; ".join(meta["skipped"]), ""]
    if meta.get("smoke"):
        L += ["**SMOKE RUN: 60 frames per sequence. Numbers are a code-path check, not results.**", ""]

    # ---- part 1
    L += ["## 1. Low thresholds", ""]
    if meta.get("detector"):
        L += ["Detector pass (ms per frame, median; conf 0.01, so post-process sees the most boxes):", ""]
        L += md_table(["variant", "pre", "inference", "post", "fps (detector only, incl. JPEG read)"],
                      [[LABEL[v], d["pre_ms_med"], d["inf_ms_med"], d["post_ms_med"], d["fps_detector_wall"]] for v, d in meta["detector"].items()])
    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "IDSW_per_min", "Frag", "TP", "FP", "FN"]
    variants = [v for v in FU_VARIANTS if p1_rows(rows, v, "stock", stock_low)]
    best = {}
    for arm, title in (("stock", f"Arm 'stock': bytetrack.yaml untouched (track_low_thresh {stock_low})"),
                       ("low=conf", "Arm 'low=conf' (extra): track_low_thresh set to the swept conf, everything else stock")):
        tabs = {v: p1_rows(rows, v, arm, stock_low) for v in variants}
        if not any(r["arm"] == arm for r in rows if r["part"] == "1"):
            continue
        L += [f"### {title}", "", "3 sequences combined, detect-every-1:", ""]
        L += md_table(["variant", "conf"] + cols, [[LABEL[v], c] + [r[k] for k in cols] for v in variants for c, r in tabs[v].items()])
        body = []
        for v in variants:
            by = tabs[v]
            top = max(r["HOTA"] for r in by.values())
            winners = [c for c, r in by.items() if r["HOTA"] == top]
            best[(arm, v)] = winners
            edge = by.get(0.1)
            if edge is None:
                below, same, gain = "conf 0.1 not measured", "-", "-"
            else:
                below = "yes" if edge["HOTA"] < top else "no"
                same = "yes" if all(all(r[k] == edge[k] for k in SAME_KEYS) for r in by.values()) else "no"
                gain = f"{top - edge['HOTA']:+.3f}"
            body.append([LABEL[v], ", ".join(map(str, winners)), top, edge["HOTA"] if edge else "-", gain, below, same,
                         ", ".join(f"{c}: {r['IDSW']}" for c, r in by.items())])
        L += md_table(["variant", "conf(s) with the highest HOTA", "highest HOTA", "HOTA at 0.1", "gain over 0.1",
                       "best threshold below 0.1?", "all metrics identical at every conf?", "IDSW per conf"], body)
        if ("ov_fp32" in variants) and ("ov_int8" in variants):
            a, b = best[(arm, "ov_fp32")], best[(arm, "ov_int8")]
            L += [f"Threshold drift, INT8 vs OpenVINO FP32: best conf {b} vs {a}: " + ("**no drift**" if a == b else "**the best threshold moved**") + ".", ""]
    if variants:
        L += [f"Reading aid: ByteTrack only uses boxes with score > track_low_thresh. In arm 'stock' that is {stock_low}, so a row of that arm "
              "can only differ from its conf 0.1 row if boxes under that score reach the tracker some other way; the column "
              "'all metrics identical at every conf?' says whether they did.", ""]
    if meta.get("equiv"):
        L += ["### Check of the method", "",
              f"MOT17-02, conf {TRACK_CONF}: predict(conf, classes=[0]) on every frame (bench.run_sequence, as in the first run) versus the kept boxes of the conf {RAW_CONF} pass:", ""]
        L += md_table(["variant", "same tracks?", "max box difference px", "IDSW direct", "IDSW kept boxes", "HOTA direct", "HOTA kept boxes"],
                      [[LABEL[v], e["same_tracks"], e["max_absdiff"], e["direct"]["IDSW"], e["cached"]["IDSW"], e["direct"]["HOTA"], e["cached"]["HOTA"]]
                       for v, e in meta["equiv"].items()])

    # ---- part 2
    L += ["## 2. Why OpenVINO FP32 differs from PyTorch FP32", ""]
    ex = meta.get("exports2", {})
    if ex:
        L += ["### Exports", ""]
        L += md_table(["export", "export args", "export s", "inputs", "outputs", "NMS nodes", "TopK nodes", "metadata", "warnings during export"],
                      [[n, e.get("kwargs"), e.get("export_s", "-"), e.get("inputs"), e.get("outputs"), e.get("nms_nodes"), e.get("topk_nodes"),
                        f"{e.get('metadata')} args {e.get('metadata_args')}", " / ".join(e.get("warnings") or []) or "-"] for n, e in ex.items()])
    if meta.get("pre"):
        L += ["### Pre-processing, frame 0 of MOT17-02", "",
              "The tensor ultralytics hands to the network (predictor.preprocess), PyTorch vs OpenVINO, and the shape of the raw network output:", ""]
        L += md_table(["export", "PyTorch predict args", "PyTorch input", "OpenVINO input", "same shape?", "max pixel diff (of 255)",
                       "mean pixel diff", "values differing", "PyTorch raw output", "OpenVINO raw output"],
                      [[p["export"], p["pt_kwargs"], p["pt_shape"], p["ov_shape"], p["same_shape"], p.get("max_absdiff_255", "-"),
                        p.get("mean_absdiff_255", "-"), f"{p['values_differing']} of {p['values_total']}" if p["same_shape"] else "-",
                        p["pt_raw_out"], p["ov_raw_out"]] for p in meta["pre"]])
    diff = meta.get("diff", [])
    if diff:
        L += ["### Raw detections, PyTorch vs OpenVINO FP32", "",
              "Person boxes with score > floor. Matched = one-to-one, IoU > 0.9. 'only' = boxes of that side without such a match "
              "(in brackets: without even an IoU > 0.5 partner). Coordinates in pixels of the 1920x1080 frame.", ""]
        L += md_table(["export", "sequence", "floor", "boxes/frame PT", "boxes/frame OV", "matched/frame", "PT only/frame", "OV only/frame",
                       "frames with any unmatched", "mean abs score diff", "max abs score diff", "mean coord diff px", "max coord diff px"],
                      [[d["export"], d["seq"], d["floor"], d["pt_boxes_per_frame"], d["ov_boxes_per_frame"], d["matched_per_frame"],
                        f"{d['pt_only_per_frame']} ({d['pt_only_iou50_per_frame']})", f"{d['ov_only_per_frame']} ({d['ov_only_iou50_per_frame']})",
                        f"{d['frames_with_unmatched']} of {d['frames']}", d["score_absdiff_mean"], d["score_absdiff_max"],
                        d["coord_absdiff_mean_px"], d["coord_absdiff_max_px"]] for d in diff])
    caps = meta.get("cap", [])
    if caps:
        L += [f"### The {MAX_DET}-box cap (conf {RAW_CONF}, boxes of all classes)", ""]
        L += md_table(["model", "sequence", "boxes/frame mean", "max", f"frames at the cap ({MAX_DET})", "lowest kept score in capped frames (median / max)",
                       "capped frames whose lowest kept score is > 0.1", "person boxes/frame", "person boxes > 0.1 /frame"],
                      [[c["model"], c["seq"], c["boxes_all_classes_mean"], c["boxes_all_classes_max"], f"{c['frames_at_cap']} of {c['frames']}",
                        f"{c['lowest_kept_score_at_cap_median']} / {c['lowest_kept_score_at_cap_max']}", c["frames_at_cap_with_lowest_score_over_0p1"],
                        c["person_boxes_mean"], c["person_boxes_over_0p1_mean"]] for c in caps])
        L += ["A capped frame can only have lost boxes that score under its lowest kept score.", ""]
    if meta.get("maxdet"):
        L += [f"Asking for more boxes (first frames of {DIFF_SEQS[-1]}, conf {RAW_CONF}):", ""]
        L += md_table(["model", "setting", "result"], [[LABEL[v], k, r] for v, p in meta["maxdet"].items() for k, r in p.items()])
    t2 = [r for r in rows if r["part"] == "2"]
    if t2:
        L += [f"### Tracking from those detections (conf {TRACK_CONF}, stock ByteTrack)", ""]
        L += md_table(["export", "sequence", "side", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "TP", "FP", "FN"],
                      [[r["export"], r["seq"], r["side"]] + [r[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag", "TP", "FP", "FN")]
                       for r in t2 if r["seq"] != "COMBINED"])

    # ---- conclusions: sentences picked by the measured values
    L += ["## Conclusions (from the numbers above only)", ""]
    for arm in ("stock", "low=conf"):
        for v in variants:
            if (arm, v) in best:
                by = p1_rows(rows, v, arm, stock_low)
                top = max(r["HOTA"] for r in by.values())
                edge = by.get(0.1)
                L.append(f"- Thresholds, {LABEL[v]}, arm '{arm}': highest HOTA {top} at conf {best[(arm, v)]}"
                         + (f"; conf 0.1 gives {edge['HOTA']}, so the best threshold is " + ("below 0.1." if edge["HOTA"] < top else "not below 0.1.") if edge else "; conf 0.1 was not measured."))
    gaps = []
    for name, *_ in EXPORTS2:
        pt = next((r for r in t2 if r["export"] == name and r["seq"] == DIFF_SEQS[0] and r["side"] == "PyTorch"), None)
        ov = next((r for r in t2 if r["export"] == name and r["seq"] == DIFF_SEQS[0] and r["side"] == "OpenVINO"), None)
        d = next((x for x in diff if x["export"] == name and x["seq"] == DIFF_SEQS[0] and x["floor"] == 0.1), None)
        p = next((x for x in meta.get("pre", []) if x["export"] == name), None)
        if not (pt and ov and d):
            continue
        gaps.append((abs(ov["IDSW"] - pt["IDSW"]), name))
        pre = "input not checked" if not p else ("input shapes differ: " + p["pt_shape"] + " vs " + p["ov_shape"]) if not p["same_shape"] \
            else "input tensors identical" if p["max_absdiff_255"] == 0 else f"input tensors differ (max {p['max_absdiff_255']} of 255)"
        L.append(f"- Export {name}, {DIFF_SEQS[0]}: {pre}; boxes with score > 0.1 per frame: {d['pt_only_per_frame']} PyTorch-only, "
                 f"{d['ov_only_per_frame']} OpenVINO-only, {d['matched_per_frame']} matched (mean score difference {d['score_absdiff_mean']}, "
                 f"mean coordinate difference {d['coord_absdiff_mean_px']} px); ID switches PyTorch {pt['IDSW']} vs OpenVINO {ov['IDSW']}, "
                 f"HOTA {pt['HOTA']} vs {ov['HOTA']}. "
                 + ("The detections are the same boxes, so the tracking result is the same." if d["pt_only_per_frame"] == 0 and d["ov_only_per_frame"] == 0 and pt["IDSW"] == ov["IDSW"]
                    else "The two detectors already return different boxes before the tracker sees them." if d["pt_only_per_frame"] or d["ov_only_per_frame"]
                    else "The boxes match one-to-one; the remaining differences are in scores / coordinates."))
    if gaps:
        g, name = min(gaps)
        L.append(f"- Smallest ID switch gap between PyTorch and OpenVINO on {DIFF_SEQS[0]}: export {name} (gap {g}); gaps of all exports: "
                 + ", ".join(f"{n.split(':')[0]} = {x}" for x, n in gaps) + ".")
    if caps:
        hit = [c for c in caps if c["frames_at_cap"]]
        L.append(f"- {MAX_DET}-box cap at conf {RAW_CONF}: " + ("no frame of any model reached it." if not hit else
                 "; ".join(f"{c['model']} {c['seq']}: {c['frames_at_cap']} of {c['frames']} frames, {c['frames_at_cap_with_lowest_score_over_0p1']} of them with all kept scores > 0.1" for c in hit)
                 + ". Only frames whose kept scores are all > 0.1 can have lost a box that stock ByteTrack would have used."))
    L += ["", "Raw rows: p2f_results.csv / p2f_results.json (the json also holds every table above as data).", ""]
    (out / "p2f_report.md").write_text("\n".join(L))


def fu_save(out, rows, meta, status=None):
    import pandas as pd

    if status:
        meta["status"] = status
    meta["elapsed_s"] = round(time.time() - T0)
    pd.DataFrame(rows).to_csv(out / "p2f_results.csv", index=False)
    (out / "p2f_results.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=str))
    try:
        fu_report(out, rows, meta)
    except Exception as e:
        log(f"report failed: {e!r}")


# ----------------------------------------------------------------------------------------------- main


def followup_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="60 frames per sequence (code-path check)")
    ap.add_argument("--data-dir", help="existing MOT17 'train' folder (skips the download)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--budget-min", type=float, default=FU_BUDGET_S / 60, help="hard stop for the whole run")
    args, _ = ap.parse_known_args()  # Kaggle may pass its own argv

    on_kaggle = os.path.isdir("/kaggle/working")
    if not on_kaggle and not args.smoke:
        raise SystemExit("the full run only works on Kaggle; use --smoke locally")
    out = Path(args.out or ("/kaggle/working" if on_kaggle else "p2f_out"))
    work = Path("/kaggle/tmp/p2f" if on_kaggle else out / "work")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    deadline = T0 + args.budget_min * 60 - FU_RESERVE_S
    if on_kaggle and not pip(*PINS):
        log("pinned install failed, retrying without pins")
        pip(*NOPIN)

    meta = dict(model=FU_MODEL, smoke=args.smoke, test_seqs=TEST_SEQS, imgsz=IMGSZ, raw_conf=RAW_CONF, confs=LOW_CONFS, floors=FLOORS,
                budget_s=round(args.budget_min * 60), status="running", exports={}, exports2={}, skipped=[], detector={}, equiv={},
                cap=[], maxdet={}, pre=[], diff=[], started=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    write_env(out, meta)
    (out / "p2_env.txt").replace(out / "p2f_env.txt")
    rows = []
    fu_save(out, rows, meta)

    import cv2
    from ultralytics import YOLO

    train_dir = get_mot17(args, work, meta)
    nf = 60 if args.smoke else 0
    seqs = {s: load_sequence(train_dir, s, nf) for s in TEST_SEQS}
    calib_yaml, meta["calib_frames"] = make_calib_dataset(load_sequence(train_dir, CALIB_SEQ, nf), work, 20 if args.smoke else CALIB_FRAMES)
    _, meta["tracker_cfg"] = make_tracker()
    stock_low = meta["tracker_cfg"].get("track_low_thresh", 0.1)
    img0 = cv2.imread(str(seqs[DIFF_SEQS[0]]["frames"][0]))
    kw_rect = dict(imgsz=IMGSZ)

    os.chdir(work)  # ultralytics downloads the weights into the current folder
    weights = work / f"{FU_MODEL}.pt"
    YOLO(weights.name)
    paths = {"pt_fp32": str(weights)}
    for v in ("ov_fp32", "ov_int8"):
        try:
            paths[v], secs, mb, kw = export_variant(v, weights, work, calib_yaml, FU_MODEL)
            meta["exports"][v] = dict(export_s=secs, size_mb=mb, kwargs=str(kw))
        except Exception as e:
            log(f"EXPORT {v} FAILED: {e!r}")
            meta["skipped"].append(f"{v}: export failed ({e!r})")

    def time_up(what):
        if time.time() > deadline:
            meta["skipped"].append(f"{what}: time budget")
            return True
        return False

    def add_caps(label, cache, names):
        for s in names:
            meta["cap"].append(dict(model=label, seq=s, **cap_stats(cache[s])))

    # ---- part 1: one detector pass per variant, then the thresholds on the kept boxes
    caches = {}

    def sweep(v, arm):
        for conf in LOW_CONFS:
            if arm != "stock" and conf >= stock_low:
                continue  # same as the stock row
            if time_up(f"part 1 {v} conf={conf} arm={arm}"):
                return
            t = time.time()
            new = tracked_rows(list(seqs.values()), caches[v], conf, None if arm == "stock" else conf,
                               dict(part="1", model=FU_MODEL, variant=v, arm=arm, track_low_thresh=stock_low if arm == "stock" else conf))
            rows.extend(new)
            c = new[-1]
            log(f"part 1 {v} conf={conf} arm={arm}: HOTA {c['HOTA']} MOTA {c['MOTA']} IDF1 {c['IDF1']} IDSW {c['IDSW']} ({time.time() - t:.0f} s)")
            fu_save(out, rows, meta)

    for v in [v for v in FU_VARIANTS if v in paths]:
        if time_up(f"part 1 {v} detector pass"):
            continue
        try:
            model = fu_load(paths[v], img0, kw_rect)
            caches[v] = {}
            for name, s in seqs.items():
                caches[v][name] = detect_pass(model, s, kw_rect)
                log(f"detector pass {v} {name}: {len(s['frames'])} frames in {caches[v][name]['wall_s']:.0f} s")
            meta["detector"][v] = pass_speed(list(caches[v].values()))
            add_caps(LABEL[v], caches[v], TEST_SEQS)
            if v != "ov_int8":
                s = seqs[DIFF_SEQS[0]]
                direct, _ = run_sequence(model, s, TRACK_CONF, 1)
                cached, _ = track_cached(s, caches[v][s["name"]]["dets"], TRACK_CONF)
                same, worst = tracks_equal(direct, cached)
                meta["equiv"][v] = dict(same_tracks=same, max_absdiff=worst, direct=final_metrics(score_sequence(s, direct)),
                                        cached=final_metrics(score_sequence(s, cached)))
                log(f"method check {v}: same tracks {same}, max diff {worst}")
                meta["maxdet"][v] = maxdet_probe(model, seqs[DIFF_SEQS[-1]], kw_rect)
            del model
            gc.collect()
            fu_save(out, rows, meta)
            sweep(v, "stock")
        except Exception as e:
            log(f"PART 1 {v} FAILED: {e!r}")
            meta["skipped"].append(f"part 1 {v}: {e!r}")

    # ---- part 2: PyTorch vs OpenVINO FP32 raw detections for four exports
    dseqs = [seqs[s] for s in DIFF_SEQS]
    pt_model, pt_cache, pt_pre, pt_tracked = None, {}, {}, set()
    if "pt_fp32" in caches:
        pt_cache["rect"] = caches["pt_fp32"]
    for name, ekw, pkw_pt, pkw_ov, key in EXPORTS2:
        if time_up(f"part 2 export {name}"):
            continue
        try:
            if name.startswith("a") and "ov_fp32" in paths:
                path, info = paths["ov_fp32"], dict(meta["exports"]["ov_fp32"], warnings=[])
            else:
                path, secs, warns = export_custom(name, weights, work, ekw)
                info = dict(kwargs=str(ekw), export_s=secs, warnings=warns)
            meta["exports2"][name] = dict(info, **ov_model_info(path))
            log(f"part 2 {name}: {meta['exports2'][name]}")
            if pt_model is None:
                pt_model = fu_load(paths["pt_fp32"], img0, kw_rect)
            if key not in pt_cache:
                pt_cache[key] = {s["name"]: detect_pass(pt_model, s, pkw_pt) for s in dseqs}
                add_caps(f"PyTorch FP32 {pkw_pt}", pt_cache[key], DIFF_SEQS)
            if key not in pt_pre:
                pt_pre[key] = pre_tensor(pt_model, img0, pkw_pt)
            ov_model = fu_load(path, img0, pkw_ov)
            if name.startswith("a") and "ov_fp32" in caches:
                ov_cache = caches["ov_fp32"]
            else:
                ov_cache = {s["name"]: detect_pass(ov_model, s, pkw_ov) for s in dseqs}
                add_caps(f"OpenVINO FP32 export {name.split(':')[0]}", ov_cache, DIFF_SEQS)
            (t_pt, raw_pt), (t_ov, raw_ov) = pt_pre[key], pre_tensor(ov_model, img0, pkw_ov)
            meta["pre"].append(dict(export=name, pt_kwargs=str(pkw_pt), ov_kwargs=str(pkw_ov), pt_shape=str(t_pt.shape), ov_shape=str(t_ov.shape),
                                    pt_raw_out=raw_pt, ov_raw_out=raw_ov, **pre_diff(t_pt, t_ov)))
            for s in dseqs:
                for floor in FLOORS:
                    meta["diff"].append(dict(export=name, seq=s["name"], floor=floor,
                                             **match_stats(pt_cache[key][s["name"]]["dets"], ov_cache[s["name"]]["dets"], floor)))
            base = dict(part="2", model=FU_MODEL, export=name, arm="stock", track_low_thresh=stock_low)
            rows.extend(tracked_rows(dseqs, pt_cache[key], TRACK_CONF, None, dict(base, variant="pt_fp32", side="PyTorch")))
            rows.extend(tracked_rows(dseqs, ov_cache, TRACK_CONF, None, dict(base, variant="ov_fp32", side="OpenVINO")))
            d = next(x for x in meta["diff"] if x["export"] == name and x["floor"] == 0.1)
            log(f"part 2 {name} {d['seq']}: PT-only {d['pt_only_per_frame']}/frame, OV-only {d['ov_only_per_frame']}/frame, score diff {d['score_absdiff_mean']}")
            del ov_model
            gc.collect()
        except Exception as e:
            log(f"PART 2 {name} FAILED: {e!r}")
            meta["skipped"].append(f"part 2 export {name}: {e!r}")
        fu_save(out, rows, meta)

    # ---- part 1, extra arm (cheap: tracker only), last so it cannot cost part 2 its time
    for v in caches:
        try:
            sweep(v, "low=conf")
        except Exception as e:
            log(f"PART 1 {v} arm low=conf FAILED: {e!r}")
            meta["skipped"].append(f"part 1 {v} arm low=conf: {e!r}")

    fu_save(out, rows, meta, status="complete" if not meta["skipped"] else "partial (see 'Not run')")
    log(f"done: {len(rows)} rows, status {meta['status']}, outputs in {out}")


if __name__ == "__main__":
    followup_main()
