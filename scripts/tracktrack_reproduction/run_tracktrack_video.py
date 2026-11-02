#!/usr/bin/env python3
"""Run the author's TrackTrack models/core on one custom, already-undistorted video.

This adapter fixes the generic MOT17 parameters, not dataset/video-name heuristics.
No benchmark data, ground truth, AFLink, interpolation or smoothing is used.
The two local model checkpoints must be the trusted author-provided checkpoints:
loading them explicitly uses torch.load(weights_only=False). Never use untrusted
pickle/checkpoint files with this runner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import torch


INPUT_SIZE = (800, 1440)  # H, W, matching the author's MOT17 experiment.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SEQUENCE_NAME = "aria_video"  # Fixed safe name; never enters dataset-name branches.
FEATURE_DIM = 2048
SEED = 10000


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True, help="Official TrackTrack checkout")
    p.add_argument("--video", type=Path, help="Undistorted input video; process every frame from its beginning")
    p.add_argument("--output", type=Path, help="New or empty output directory; never overwrite an existing run")
    p.add_argument("--cmc", type=Path, help="TAB-separated index,a00,a01,tx,a10,a11,ty; row zero must be identity")
    p.add_argument("--max-frames", type=int, default=None, help="Optional prefix length for smoke tests")
    p.add_argument("--reid-batch-size", type=int, default=32)
    p.add_argument("--self-check", action="store_true", help="Import official modules and check configuration; no checkpoint loading")
    p.add_argument("--load-models-only", action="store_true", help="Load both trusted author checkpoints, write metadata, then exit")
    return p


def fixed_tracker_args(fps, width, height):
    return SimpleNamespace(
        det_thr=0.60, init_thr=0.70, match_thr=0.70,
        penalty_p=0.20, penalty_q=0.40, reduce_step=0.05, tai_thr=0.55,
        min_len=3, min_box_area=100,
        # Author run.py uses integer seqinfo FPS * 2. Preserve two seconds for
        # fractional-FPS custom video, with an explicit rounding rule.
        max_time_lost=max(1, int(round(2 * fps))),
        img_w=width, img_h=height,
        data_path="MOT17_custom_video/",  # Semantic marker only; no file IO here.
    )


def load_official_api(repo):
    for subdir in ("1. YOLOX", "2. FastReID", "3. Tracker"):
        directory = repo / subdir
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        sys.path.insert(0, str(directory))
    # Only import inference modules. The benchmark runner/TrackEval/AFLink are
    # deliberately not imported.
    from yolox.exp import get_exp
    from yolox.data.data_augment import preproc
    from yolox.utils import fuse_model, postprocess
    from fastreid.emb_computer import EmbeddingComputer
    from fastreid.utils.checkpoint import Checkpointer
    from trackers.tracker import Tracker
    exp = get_exp(str(repo / "1. YOLOX/exps/yolox_x_mot17_test.py"))
    if tuple(exp.test_size) != INPUT_SIZE or exp.num_classes != 1 or exp.test_conf != 0.1:
        raise RuntimeError("Official MOT17 configuration changed; review the adapter before running")
    if not hasattr(np, "float_"):
        raise RuntimeError("Author CMC requires NumPy 1.x; use the pinned NumPy 1.26.4 runtime")
    return SimpleNamespace(get_exp=get_exp, preproc=preproc, fuse_model=fuse_model,
                           postprocess=postprocess, EmbeddingComputer=EmbeddingComputer,
                           Checkpointer=Checkpointer, Tracker=Tracker, exp=exp)


def file_info(path, digest=False):
    result = {"path": str(path), "bytes": path.stat().st_size,
              "mtime_ns": path.stat().st_mtime_ns}
    if digest:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        result["sha256"] = h.hexdigest()
    return result


def git_info(repo):
    def git(*args):
        # Windows-created checkout is read by WSL. Normalize CRLF per command,
        # without mutating repository config/files or recording a giant false diff.
        proc = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", str(repo), *args], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short", "--untracked-files=no"),
            "diff": git("diff", "--stat", "--", "1. YOLOX", "2. FastReID", "3. Tracker"),
            "diff_format": "stat", "status_excludes_untracked": True,
            "line_ending_normalization": "Per-command git -c core.autocrlf=true; repository config/files unchanged"}


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def validate_cmc(path):
    """Validate without silently shifting, sorting, filling or changing values."""
    count = 0
    first_index = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            tokens = line.rstrip("\r\n").split("\t")
            while tokens and tokens[-1] == "":
                tokens.pop()
            if len(tokens) != 7:
                raise ValueError(f"CMC line {line_number}: expected 7 TAB-separated columns")
            index_float = float(tokens[0])
            if not math.isfinite(index_float) or not index_float.is_integer():
                raise ValueError(f"CMC line {line_number}: index must be an integer")
            index = int(index_float)
            values = np.asarray([float(x) for x in tokens[1:]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"CMC line {line_number}: non-finite matrix")
            if first_index is None:
                first_index = index
                if index not in (0, 1):
                    raise ValueError("CMC indices must start at 0 or 1 for this complete video prefix")
                if not np.allclose(values, [1, 0, 0, 0, 1, 0], rtol=0, atol=1e-6):
                    raise ValueError("First CMC row must be identity, aligned to the first input frame")
            if index != first_index + count:
                raise ValueError(f"CMC line {line_number}: non-contiguous frame indices")
            count += 1
    if not count:
        raise ValueError("CMC file is empty")
    return {"rows": count, "first_index": first_index,
            "consumption": "one row per input frame, including empty detection frames; index column ignored by author core"}


@contextmanager
def official_tracker(api, args, output, cmc_path):
    """Use author CMC's relative path unchanged, in an adapter-owned directory."""
    work = output / "tracker_work"
    target = work / "trackers" / "cmc" / f"GMC-{SEQUENCE_NAME}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cmc_path, target)
    previous = Path.cwd()
    tracker = None
    try:
        os.chdir(work)
        tracker = api.Tracker(args, SEQUENCE_NAME)
        yield tracker
    finally:
        if tracker is not None:
            tracker.cmc.gmcFile.close()
        os.chdir(previous)


def load_models(api, repo, batch_size, report):
    if not torch.cuda.is_available():
        raise RuntimeError("The author's detector and EmbeddingComputer require CUDA")
    torch.cuda.set_device(0)
    detector_path = repo / "1. YOLOX/weights/mot17.pth.tar"
    reid_path = repo / "2. FastReID/weights/mot17_sbs_S50.pth"
    config_path = repo / "2. FastReID/configs/MOT17/sbs_S50.yml"
    for path in (detector_path, reid_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report["checkpoints"] = {"yolox": file_info(detector_path, True), "reid": file_info(reid_path, True)}
    report["reid_config"] = file_info(config_path, True)
    report["checkpoint_trust"] = "Explicit weights_only=False for these two author-provided local checkpoints only"
    report["compatibility"] = [
        "YOLOX torch.load explicitly sets weights_only=False for trusted author checkpoint",
        "FastReID Checkpointer._load_file is temporarily overridden during initialization only to set weights_only=False; restored immediately",
    ]
    print("Loading author YOLOX-X checkpoint...", flush=True)
    detector = api.exp.get_model().eval()
    checkpoint = torch.load(str(detector_path), map_location="cpu", weights_only=False)
    detector.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    # Load/fuse on CPU, then convert to FP16 before the GPU transfer. This avoids
    # keeping a full-precision model or training checkpoint optimizer on the GPU.
    detector = api.fuse_model(detector).eval().half().cuda()

    print("Loading author FastReID checkpoint...", flush=True)
    embedder = api.EmbeddingComputer(str(config_path), str(reid_path), max_batch=batch_size)
    original_load = api.Checkpointer._load_file
    original_log = api.Checkpointer._log_incompatible_keys
    report["reid_checkpoint_incompatibilities"] = []

    def trusted_reid_load(self, filename):
        if Path(filename).resolve() != reid_path.resolve():
            raise RuntimeError("Refusing an unexpected checkpoint path in the local compatibility override")
        return torch.load(filename, map_location=torch.device("cpu"), weights_only=False)

    def record_incompatible(self, incompatible):
        record = {"missing_keys": list(incompatible.missing_keys),
                  "unexpected_keys": list(incompatible.unexpected_keys),
                  "incorrect_shapes": list(incompatible.incorrect_shapes)}
        report["reid_checkpoint_incompatibilities"].append(record)
        original_log(self, incompatible)
        # The training-only classifier is unused in eval forward. Missing actual
        # embedding/backbone weights would invalidate reproduction, so fail early.
        bad = [key for key in record["missing_keys"] if key != "heads.weight"]
        bad += [item[0] for item in record["incorrect_shapes"] if item[0] != "heads.weight"]
        if bad:
            raise RuntimeError(f"ReID checkpoint has incompatible inference weights: {bad}")

    api.Checkpointer._load_file = trusted_reid_load
    api.Checkpointer._log_incompatible_keys = record_incompatible
    try:
        embedder.initialize_model()
    finally:
        api.Checkpointer._load_file = original_load
        api.Checkpointer._log_incompatible_keys = original_log
    torch.cuda.synchronize()
    return detector, embedder


def decode_six_columns(output, scale, width, height):
    """Exactly the author's DetEvaluator conversion, including its output dtype."""
    if output is None:
        return np.empty((0, 6), dtype=np.float16)
    # postprocess can return an inference tensor. A clone here (outside the
    # inference context) permits the author's in-place confidence conversion.
    output = output.clone()
    output[:, 4] *= output[:, 5]
    output[:, 5] = output[:, 6]
    result = output[:, :6].detach().cpu().numpy().copy()
    result[:, :4] /= scale
    valid = (np.minimum(result[:, 2], width - 1) - np.maximum(result[:, 0], 0)) > 0
    valid &= (np.minimum(result[:, 3], height - 1) - np.maximum(result[:, 1], 0)) > 0
    result = result[valid]
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite detector output")
    return result


def crop_valid_mask(dets, width, height):
    """The exact rounded/clipped crop geometry used by EmbeddingComputer."""
    boxes = np.round(dets[:, :4]).astype(np.int32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


def add_embeddings(embedder, image, dets, batch_size):
    # The official implementation puts every crop on GPU before its own batching.
    # Bound that allocation using repeated calls to the SAME official function.
    if not len(dets):
        return np.empty((0, 6 + FEATURE_DIM), dtype=np.float16)
    chunks = []
    for start in range(0, len(dets), batch_size):
        part = dets[start:start + batch_size]
        feat = embedder.compute_embedding(image, part[:, :4])
        if feat.shape != (len(part), FEATURE_DIM) or not np.isfinite(feat).all():
            raise RuntimeError(f"Invalid ReID embeddings, shape={feat.shape}")
        if np.any(np.linalg.norm(feat.astype(np.float32), axis=1) < 0.5):
            raise RuntimeError("ReID returned zero/unnormalized embeddings")
        chunks.append(np.concatenate([part, feat], axis=1))
    return np.concatenate(chunks, axis=0)


def step_tracker(tracker, det80, det95):
    if len(det80):
        # Empty arrays are valid for the 0.95 input but not the 0.80 reduction.
        return tracker.update(det80, det95)
    return tracker.update_without_detections()


def filtered_tracks(tracks, min_area=100):
    kept = []
    rejected = {"aspect_ratio": 0, "area_or_id": 0}
    for track in tracks:
        box = track.x1y1wh
        if not np.isfinite(box).all():
            raise RuntimeError("Non-finite tracker box")
        x, y, w, h = map(float, box)
        if h == 0 or w / h > 1.6:
            rejected["aspect_ratio"] += 1
            continue
        if track.track_id > 0 and w * h > min_area:
            kept.append((int(track.track_id), x, y, w, h, float(track.score)))
        else:
            rejected["area_or_id"] += 1
    return kept, rejected


def render_preview(frame, tracks, frame_id):
    font_scale = max(0.7, frame.shape[1] / 1800)
    line_width = max(2, int(round(frame.shape[1] / 900)))
    for track_id, x, y, w, h, score in tracks:
        color = (50 + (track_id * 73) % 206, 50 + (track_id * 151) % 206,
                 50 + (track_id * 199) % 206)
        p1, p2 = (round(x), round(y)), (round(x + w), round(y + h))
        cv2.rectangle(frame, p1, p2, color, line_width)
        label_origin = (max(0, p1[0]), max(round(25 * font_scale), p1[1] - 7))
        cv2.putText(frame, f"ID {track_id} {score:.2f}", label_origin,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_width, cv2.LINE_AA)
    cv2.putText(frame, f"TrackTrack | frame {frame_id} | {len(tracks)} tracks", (12, round(28 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_width, cv2.LINE_AA)
    return frame


def run_video(api, detector, embedder, args, report):
    cmc_meta = validate_cmc(args.cmc)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nominal_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video must have valid FPS and image dimensions")
    expected = min(nominal_count, args.max_frames) if args.max_frames else nominal_count
    if expected > 0 and cmc_meta["rows"] < expected:
        cap.release()
        raise ValueError(f"CMC has {cmc_meta['rows']} rows, but {expected} video frames are requested")
    tracker_args = fixed_tracker_args(fps, width, height)
    report.update({"video": file_info(args.video), "video_metadata": {"fps": fps, "width": width,
                  "height": height, "nominal_frames": nominal_count}, "cmc": {**file_info(args.cmc, True), **cmc_meta},
                  "tracker_parameters": vars(tracker_args)})
    write_json(args.output / "runtime_parameters.json", report)
    cache_dir = args.output / "detections"
    cache_dir.mkdir()
    # Some OpenCV encoders truncate odd dimensions: pad ONLY the preview canvas.
    preview_width, preview_height = width + width % 2, height + height % 2
    writer = cv2.VideoWriter(str(args.output / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (preview_width, preview_height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not create the mp4v preview writer")
    report["preview"] = {"codec": "mp4v", "width": preview_width, "height": preview_height,
                          "audio": False, "padding_right_bottom_only": [preview_width-width, preview_height-height]}
    totals = {"frames": 0, "frames_without_primary_detections": 0,
              "raw_det80": 0, "raw_det95": 0, "reid_crop_rejected80": 0, "reid_crop_rejected95": 0,
              "output_boxes": 0, "output_aspect_rejected": 0, "output_area_or_id_rejected": 0}
    durations = {"detector_seconds": 0.0, "reid_seconds": 0.0, "tracker_seconds": 0.0}
    ids = set()
    started = time.perf_counter()
    try:
        with official_tracker(api, tracker_args, args.output, args.cmc) as tracker, \
                (args.output / "tracks.txt").open("w", encoding="utf-8") as mot, \
                (args.output / "frames.jsonl").open("w", encoding="utf-8") as frame_log:
            while args.max_frames is None or totals["frames"] < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_id = totals["frames"] + 1
                if frame_id > cmc_meta["rows"]:
                    raise ValueError(f"CMC exhausted before decoded video frame {frame_id}")
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video dimensions changed; CMC and coordinates would be invalid")
                tick = time.perf_counter()
                inp, scale = api.preproc(frame, INPUT_SIZE, MEAN, STD)
                tensor = torch.from_numpy(inp).unsqueeze(0).cuda().half()
                with torch.inference_mode():
                    raw = detector(tensor)
                    # postprocess mutates cxcywh into xyxy: each call needs a clone.
                    out80 = api.postprocess(raw.clone(), 1, 0.1, 0.80)[0]
                    out95 = api.postprocess(raw.clone(), 1, 0.1, 0.95)[0]
                det80 = decode_six_columns(out80, scale, width, height)
                det95 = decode_six_columns(out95, scale, width, height)
                durations["detector_seconds"] += time.perf_counter() - tick
                mask80, mask95 = crop_valid_mask(det80, width, height), crop_valid_mask(det95, width, height)
                tick = time.perf_counter()
                feat80 = add_embeddings(embedder, frame, det80[mask80], args.reid_batch_size)
                feat95 = add_embeddings(embedder, frame, det95[mask95], args.reid_batch_size)
                durations["reid_seconds"] += time.perf_counter() - tick
                # Cache untouched six-column detector results AND the exact tracker
                # inputs. Rounded crop rejection can therefore be independently audited.
                np.savez_compressed(cache_dir / f"{frame_id:06d}.npz", frame_id=np.int64(frame_id),
                                    det80=det80, det95=det95, crop_valid80=mask80, crop_valid95=mask95,
                                    tracker_input80=feat80, tracker_input95=feat95)
                tick = time.perf_counter()
                tracks = step_tracker(tracker, feat80, feat95)
                durations["tracker_seconds"] += time.perf_counter() - tick
                kept, rejected = filtered_tracks(tracks, tracker_args.min_box_area)
                for track_id, x, y, w, h, score in kept:
                    # Match author output precision/format.
                    mot.write(f"{frame_id},{track_id},{round(x, 1)},{round(y, 1)},{round(w, 1)},{round(h, 1)},{round(score, 2)},-1,-1,-1\n")
                    ids.add(track_id)
                nbad80, nbad95 = int((~mask80).sum()), int((~mask95).sum())
                frame_record = {"frame": frame_id, "timestamp_seconds": (frame_id - 1) / fps,
                                "det80": len(det80), "det95": len(det95),
                                "reid_crop_rejected80": nbad80, "reid_crop_rejected95": nbad95,
                                "output_tracks": len(kept), "output_rejected": rejected}
                frame_log.write(json.dumps(frame_record) + "\n")
                display = render_preview(frame.copy(), kept, frame_id)
                if preview_width != width or preview_height != height:
                    display = cv2.copyMakeBorder(display, 0, preview_height-height, 0, preview_width-width,
                                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
                writer.write(display)
                totals["frames"] += 1
                totals["frames_without_primary_detections"] += int(not len(feat80))
                totals["raw_det80"] += len(det80)
                totals["raw_det95"] += len(det95)
                totals["reid_crop_rejected80"] += nbad80
                totals["reid_crop_rejected95"] += nbad95
                totals["output_boxes"] += len(kept)
                totals["output_aspect_rejected"] += rejected["aspect_ratio"]
                totals["output_area_or_id_rejected"] += rejected["area_or_id"]
                if frame_id == 1 or frame_id % 25 == 0:
                    print(json.dumps({**frame_record, "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)
    finally:
        cap.release()
        writer.release()
        report["result"] = {**totals, **durations, "unique_output_ids": len(ids), "track_ids": sorted(ids),
                            "processing_wall_seconds": time.perf_counter() - started}
    if not totals["frames"]:
        raise RuntimeError("No video frames decoded")
    if expected > 0 and totals["frames"] != expected:
        raise RuntimeError(f"Decoded {totals['frames']} frames, expected {expected}; refusing to report a complete run")
    preview = cv2.VideoCapture(str(args.output / "preview.mp4"))
    preview_ok, _ = preview.read()
    preview_count = int(preview.get(cv2.CAP_PROP_FRAME_COUNT))
    preview.release()
    report["preview"].update({"readback_first_frame_ok": bool(preview_ok), "encoded_frames": preview_count})
    if not preview_ok or preview_count != totals["frames"]:
        raise RuntimeError("Preview video readback/count verification failed")
    report["status"] = "complete"


def main(argv=None):
    args = parser().parse_args(argv)
    args.repo = args.repo.resolve()
    if args.reid_batch_size <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise ValueError("Batch size and max-frames must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    api = load_official_api(args.repo)
    if args.self_check:
        print(json.dumps({"status": "imports_ok", "torch": torch.__version__, "numpy": np.__version__,
                          "opencv": cv2.__version__, "cuda_available": torch.cuda.is_available(),
                          "input_size_hw": api.exp.test_size, "num_classes": api.exp.num_classes,
                          "conf": api.exp.test_conf, "tracker_parameters_at_30fps": vars(fixed_tracker_args(30, 1408, 1408))}, indent=2))
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --self-check is used")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {args.output}")
    if not args.load_models_only:
        for name in ("video", "cmc"):
            value = getattr(args, name)
            if value is None or not value.is_file():
                raise FileNotFoundError(f"--{name} must name an existing file")
            setattr(args, name, value.resolve())
        validate_cmc(args.cmc)
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # Author detect.py setting.
    report = {"status": "initializing", "created_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "repository": git_info(args.repo), "runner": file_info(Path(__file__).resolve(), True),
              "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                           "opencv": cv2.__version__, "torchvision": importlib.metadata.version("torchvision"),
                           "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
              "seed": SEED, "cudnn_benchmark": True, "max_frames": args.max_frames,
              "detector": {"input_size_hw": INPUT_SIZE, "confidence": 0.1, "nms": [0.80, 0.95],
                           "fp16": True, "fuse": True, "mean": MEAN, "std": STD,
                           "one_forward_two_cloned_postprocess_inputs": True},
              "reid": {"config": "MOT17/sbs_S50.yml", "fp16": True, "feature_dim": FEATURE_DIM,
                       "external_crop_batch_size": args.reid_batch_size},
              "adaptations": ["Custom video IO; explicit fixed generic MOT17 parameter profile",
                              "Author CMC reader uses a copied input file in output/tracker_work, not the repository",
                              "No AFLink, GBI, interpolation, smoothing or benchmark evaluation",
                              "Reject only boxes whose author rounded/clipped ReID crop has zero area; count each branch and cache originals",
                              "Call official EmbeddingComputer in bounded chunks to limit GPU crop allocation",
                              "max_time_lost=round(2*video_fps); all video frames retained; first CMC row identity",
                              "Output area>100 and aspect ratio<=1.6 explicitly match the MOT17 runner"],
              "cache_schema": {"det80/det95": "Nx6 [x1,y1,x2,y2,obj_conf*class_conf,class_id] before crop rejection",
                               "crop_valid80/crop_valid95": "N boolean mask for valid rounded/clipped ReID crop",
                               "tracker_input80/tracker_input95": "Mx2054: exact accepted det6 + L2-normalized 2048D feature",
                               "frame_id": "one-based input video frame, including empty frames"}}
    write_json(args.output / "runtime_parameters.json", report)
    try:
        if torch.cuda.is_available():
            report["gpu"] = {"name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)}
        detector, embedder = load_models(api, args.repo, args.reid_batch_size, report)
        write_json(args.output / "runtime_parameters.json", report)
        if args.load_models_only:
            report["status"] = "models_loaded"
        else:
            run_video(api, detector, embedder, args, report)
        write_json(args.output / "runtime_parameters.json", report)
        write_json(args.output / "summary.json", report)
        print(json.dumps({"status": report["status"], "output": str(args.output), "result": report.get("result")}, indent=2), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(args.output / "summary.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
