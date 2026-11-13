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
from bench import *  # noqa: F401,F403  (scripts/p2_followup_embed.py puts bench.py itself here in the Kaggle copy)
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
