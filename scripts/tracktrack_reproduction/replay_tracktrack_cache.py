#!/usr/bin/env python3
"""CPU-only replay of exact TrackTrack tracker inputs saved by the video runner.

No detector, ReID model, model checkpoint, video decoder, or CUDA operation is
used. Same-CMC replay must reproduce tracks.txt byte-for-byte. Identity/alternate
CMC runs are sensitivity checks, not accuracy evaluations without ground truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

import run_tracktrack_video as runner


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Completed video-run directory with summary.json and detections/*.npz")
    p.add_argument("--repo", type=Path, required=True, help="Same official TrackTrack checkout used by the source run")
    p.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--identity-cmc", action="store_true", help="Replace every CMC matrix with identity")
    mode.add_argument("--cmc", type=Path, help="Explicit actual/alternate CMC file, one row per processed frame")
    return p


def load_tracker_api(repo):
    """Import only the NumPy/SciPy/lap author tracker, not model inference APIs."""
    directory = repo / "3. Tracker"
    if not (directory / "trackers/tracker.py").is_file():
        raise FileNotFoundError(directory / "trackers/tracker.py")
    if not hasattr(np, "float_"):
        raise RuntimeError("Author CMC requires the pinned NumPy 1.26.4 runtime")
    sys.path.insert(0, str(directory))
    from trackers.tracker import Tracker
    return SimpleNamespace(Tracker=Tracker)


def read_source(run_dir):
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("status") != "complete":
        raise ValueError("Source video run must be complete; partial or currently running caches are not replayed")
    if not (run_dir / "tracks.txt").is_file():
        raise FileNotFoundError(run_dir / "tracks.txt")
    video = source["video_metadata"]
    expected = vars(runner.fixed_tracker_args(video["fps"], video["width"], video["height"]))
    recorded = source["tracker_parameters"]
    if recorded != expected:
        raise ValueError(f"Source tracker parameters differ from the current fixed MOT17 profile: source={recorded}, expected={expected}")
    count = source["result"]["frames"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Source summary has an invalid processed frame count")
    cache_dir = run_dir / "detections"
    actual_names = {p.name for p in cache_dir.glob("*.npz")}
    expected_names = {f"{frame:06d}.npz" for frame in range(1, count + 1)}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Cache is not a complete consecutive frame sequence: missing={missing[:10]}, extra={extra[:10]}")
    return source, SimpleNamespace(**recorded), [cache_dir / f"{i:06d}.npz" for i in range(1, count + 1)]


def read_cache(path, expected_frame):
    """Load numeric arrays without pickle; never rescale/re-normalize cached data."""
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as cache:
        frame_id = cache["frame_id"]
        if frame_id.shape != () or int(frame_id) != expected_frame:
            raise ValueError(f"Frame ID mismatch in {path}")
        arrays = {}
        counts = {}
        for branch in ("80", "95"):
            det = cache[f"det{branch}"]
            mask = cache[f"crop_valid{branch}"]
            exact_input = cache[f"tracker_input{branch}"]
            if det.ndim != 2 or det.shape[1] != 6:
                raise ValueError(f"Bad det{branch} shape in {path}")
            if mask.dtype != np.bool_ or mask.shape != (len(det),):
                raise ValueError(f"Bad crop_valid{branch} mask in {path}")
            if exact_input.shape != (int(mask.sum()), 6 + runner.FEATURE_DIM):
                raise ValueError(f"Bad tracker_input{branch} shape in {path}")
            if not np.issubdtype(exact_input.dtype, np.floating) or not np.isfinite(exact_input).all():
                raise ValueError(f"Bad tracker_input{branch} dtype/values in {path}")
            if not np.array_equal(exact_input[:, :6], det[mask]):
                raise ValueError(f"Cached detector/feature row alignment mismatch in {path}, branch {branch}")
            arrays[branch] = exact_input
            counts[f"raw_det{branch}"] = len(det)
            counts[f"reid_crop_rejected{branch}"] = int((~mask).sum())
    return arrays["80"], arrays["95"], counts, hashlib.sha256(data).hexdigest()


def cmc_prefix(path, frames):
    metadata = runner.validate_cmc(path)
    if metadata["rows"] < frames:
        raise ValueError(f"CMC contains {metadata['rows']} rows, need {frames}")
    values = np.loadtxt(path, delimiter="\t", usecols=range(1, 7), max_rows=frames, ndmin=2)
    return values, metadata


def make_identity(path, frames):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in range(frames):
            handle.write(f"{frame}\t1\t0\t0\t0\t1\t0\n")


def compare_mot_files(source_path, replay_path):
    source = source_path.read_bytes()
    replay = replay_path.read_bytes()

    def group(data):
        frames = {}
        ids = set()
        rows = 0
        for row in data.splitlines():
            if not row:
                continue
            tokens = row.split(b",")
            frame, track_id = int(tokens[0]), int(tokens[1])
            frames.setdefault(frame, []).append(row)
            ids.add(track_id)
            rows += 1
        return frames, ids, rows

    left, left_ids, left_rows = group(source)
    right, right_ids, right_rows = group(replay)
    different = [frame for frame in sorted(left.keys() | right.keys()) if left.get(frame, []) != right.get(frame, [])]
    return {"tracks_byte_identical": source == replay,
            "source_tracks_sha256": hashlib.sha256(source).hexdigest(),
            "replay_tracks_sha256": hashlib.sha256(replay).hexdigest(),
            "source_output_boxes": left_rows, "replay_output_boxes": right_rows,
            "source_unique_output_ids": len(left_ids), "replay_unique_output_ids": len(right_ids),
            "frames_with_different_output_rows": len(different), "different_frame_ids": different,
            "interpretation": "Literal row/count differences only; not ID-switch counts, accuracy, recall, HOTA or IDF1"}


def replay(args):
    args.run_dir, args.repo, args.output = args.run_dir.resolve(), args.repo.resolve(), args.output.resolve()
    if args.output == args.run_dir or args.output in args.run_dir.parents:
        raise ValueError("Output cannot be the source run directory or its ancestor")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {args.output}")
    source, tracker_args, caches = read_source(args.run_dir)
    source_cmc = args.run_dir / "tracker_work/trackers/cmc" / f"GMC-{runner.SEQUENCE_NAME}.txt"
    if not source_cmc.is_file():
        raise FileNotFoundError(f"Missing immutable CMC copy from source run: {source_cmc}")
    original_warps, _ = cmc_prefix(source_cmc, len(caches))
    actual_cmc = None
    if not args.identity_cmc:
        if args.cmc is None or not args.cmc.is_file():
            raise FileNotFoundError("--cmc must name an existing file")
        actual_cmc = args.cmc.resolve()
        warps, cmc_metadata = cmc_prefix(actual_cmc, len(caches))
    else:
        warps = np.tile([1., 0., 0., 0., 1., 0.], (len(caches), 1))
        cmc_metadata = {"rows": len(caches), "first_index": 0, "consumption": "one identity per cached frame"}
    same_warps = bool(np.array_equal(warps, original_warps))
    api = load_tracker_api(args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.identity_cmc:
        actual_cmc = args.output / "identity_cmc.txt"
        make_identity(actual_cmc, len(caches))
    random.seed(source.get("seed", runner.SEED))
    np.random.seed(source.get("seed", runner.SEED))
    report = {"status": "initializing", "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Cache/adaptor reproducibility and CMC sensitivity; no ground-truth accuracy claim",
              "source_run": str(args.run_dir), "source_summary": runner.file_info(args.run_dir / "summary.json", True),
              "replay_script": runner.file_info(Path(__file__).resolve(), True),
              "helper_runner": runner.file_info(Path(runner.__file__).resolve(), True),
              "repository": runner.git_info(args.repo), "tracker_parameters": vars(tracker_args),
              "video_metadata": source["video_metadata"], "source_processed_frames": len(caches),
              "cmc_mode": "identity" if args.identity_cmc else "supplied",
              "cmc": {**runner.file_info(actual_cmc, True), **cmc_metadata},
              "cmc_matrices_equal_to_source_for_processed_frames": same_warps,
              "require_byte_identical_tracks": same_warps,
              "versions": {"python": sys.version, "numpy": np.__version__},
              "models_loaded": False, "cuda_used": False,
              "no_postprocessing": True, "cache_values_modified": False}
    report["repository_commit_matches_source"] = report["repository"]["commit"] == source.get("repository", {}).get("commit")
    source_git = source.get("repository", {})
    report["repository_tracked_diff_matches_source"] = (
        report["repository"]["diff"] == source_git.get("diff") if source_git.get("diff_format") == "stat" else None)
    report["source_provenance_note"] = (
        "Comparable CRLF-normalized diff --stat" if source_git.get("diff_format") == "stat"
        else "Legacy source recorded full diff without CRLF normalization; its diff text is not comparable to current normalized stat")
    report["helper_runner_matches_source_hash"] = report["helper_runner"]["sha256"] == source.get("runner", {}).get("sha256")
    runner.write_json(args.output / "summary.json", report)
    totals = {"frames": 0, "frames_without_primary_detections": 0, "raw_det80": 0, "raw_det95": 0,
              "reid_crop_rejected80": 0, "reid_crop_rejected95": 0, "output_boxes": 0,
              "output_aspect_rejected": 0, "output_area_or_id_rejected": 0}
    ids = set()
    cache_digest = hashlib.sha256()
    cache_dtypes = set()
    started = time.perf_counter()
    tracker_seconds = 0.0
    try:
        with runner.official_tracker(api, tracker_args, args.output, actual_cmc) as tracker, \
                (args.output / "tracks.txt").open("w", encoding="utf-8", newline="\n") as mot:
            for frame_id, path in enumerate(caches, start=1):
                det80, det95, counts, cache_hash = read_cache(path, frame_id)
                cache_digest.update(f"{frame_id}:{cache_hash}\n".encode("ascii"))
                cache_dtypes.update([str(det80.dtype), str(det95.dtype)])
                tick = time.perf_counter()
                tracks = runner.step_tracker(tracker, det80, det95)
                tracker_seconds += time.perf_counter() - tick
                kept, rejected = runner.filtered_tracks(tracks, tracker_args.min_box_area)
                for track_id, x, y, w, h, score in kept:
                    # Same format/precision/order as the existing video runner.
                    mot.write(f"{frame_id},{track_id},{round(x, 1)},{round(y, 1)},{round(w, 1)},{round(h, 1)},{round(score, 2)},-1,-1,-1\n")
                    ids.add(track_id)
                totals["frames"] += 1
                totals["frames_without_primary_detections"] += int(not len(det80))
                totals["output_boxes"] += len(kept)
                totals["output_aspect_rejected"] += rejected["aspect_ratio"]
                totals["output_area_or_id_rejected"] += rejected["area_or_id"]
                for key, value in counts.items():
                    totals[key] += value
                if frame_id == 1 or frame_id % 100 == 0:
                    print(json.dumps({"frame": frame_id, "total_frames": len(caches), "output_tracks": len(kept),
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
        report["result"] = {**totals, "unique_output_ids": len(ids), "track_ids": sorted(ids),
                            "tracker_seconds": tracker_seconds, "processing_wall_seconds": time.perf_counter() - started,
                            "cmc_rows_consumed": totals["frames"]}
        report["cache"] = {"frames": len(caches), "tracker_input_dtypes": sorted(cache_dtypes),
                           "ordered_frame_id_and_npz_sha256_digest": cache_digest.hexdigest(),
                           "values": "Exact cached tracker_input80/95 arrays; no feature or box recomputation"}
        report["comparison_to_source"] = compare_mot_files(args.run_dir / "tracks.txt", args.output / "tracks.txt")
        if same_warps and not report["comparison_to_source"]["tracks_byte_identical"]:
            raise RuntimeError("Same-CMC replay failed byte-identical tracks.txt check; inspect summary and provenance before using results")
        report["status"] = "complete"
        runner.write_json(args.output / "summary.json", report)
        print(json.dumps({"status": "complete", "output": str(args.output),
                          "same_cmc": same_warps, "comparison": report["comparison_to_source"]}, indent=2), flush=True)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        report.setdefault("partial_result", totals)
        runner.write_json(args.output / "summary.json", report)
        raise


def main(argv=None):
    replay(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
