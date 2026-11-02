"""Reusable phase-1 annotation pipeline orchestrator.

Commands:
  prepare  Run one-pass detection, parallel trackers, stitching, fusion, event
           generation, review clips and a CVAT-ready XML.
  review   Audit a CVAT-corrected XML and optionally evaluate it against gold.
  status   Print completed stages and important artifact paths.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "tracking"
DEFAULT_PYTHON = ROOT / ".conda-envs" / "phase1-py310" / "Scripts" / "python.exe"


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--force", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    common(prepare)
    prepare.add_argument("--video", type=Path, required=True)
    prepare.add_argument("--model", type=Path, default=ROOT / "models" / "tracking" / "yolo11l.pt")
    prepare.add_argument("--current-tracker", type=Path, default=TRACKING / "botsort_reid.yaml")
    prepare.add_argument("--reid", type=Path, default=ROOT / "models" / "tracking" / "osnet_x0_25_msmt17.pth")
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--imgsz", type=int, default=1280)
    prepare.add_argument("--conf", type=float, default=0.15)
    prepare.add_argument("--iou", type=float, default=0.70)
    prepare.add_argument("--max-frames", type=int)
    prepare.add_argument("--max-review-clips", type=int, default=50)

    review = subparsers.add_parser("review")
    common(review)
    review.add_argument("--annotations", type=Path, required=True, help="CVAT XML exported after human correction.")
    review.add_argument("--video", type=Path)
    review.add_argument("--ground-truth", type=Path)

    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def state_path(run_dir: Path) -> Path:
    return run_dir / "pipeline_state.json"


def read_state(run_dir: Path) -> dict:
    path = state_path(run_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"stages": {}}


def save_state(run_dir: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path(run_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_step(
    name: str,
    command: list[str],
    run_dir: Path,
    state: dict,
    sentinel: Path,
    force: bool,
) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if sentinel.is_file() and not force:
        state["stages"][name] = {"status": "reused", "sentinel": str(sentinel.resolve())}
        save_state(run_dir, state)
        print(f"[{name}] reuse {sentinel}")
        return
    print(f"[{name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    (logs / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.stdout:
        print(completed.stdout[-4000:])
    if completed.returncode != 0:
        state["stages"][name] = {"status": "failed", "returncode": completed.returncode}
        save_state(run_dir, state)
        raise SystemExit(f"Stage {name!r} failed; see {logs / f'{name}.stderr.log'}")
    if not sentinel.is_file():
        raise SystemExit(f"Stage {name!r} succeeded but did not create {sentinel}")
    state["stages"][name] = {"status": "complete", "sentinel": str(sentinel.resolve())}
    save_state(run_dir, state)


def prepare(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    for path in (args.video, args.model, args.current_tracker, args.reid):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = read_state(run_dir)
    state.update(
        {
            "schema_version": "1.0",
            "run_dir": str(run_dir),
            "video": str(args.video.resolve()),
            "command": "prepare",
        }
    )
    state.setdefault("stages", {})
    save_state(run_dir, state)

    proposals = run_dir / "01_proposals"
    command = [
        str(python), "-u", str(TRACKING / "phase1_generate_proposals.py"),
        "--video", str(args.video.resolve()),
        "--output-dir", str(proposals),
        "--model", str(args.model.resolve()),
        "--current-tracker", str(args.current_tracker.resolve()),
        "--deepoc-reid", "osnet-x0-25-msmt17",
        "--device", args.device,
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    run_step("01_generate_proposals", command, run_dir, state, proposals / "proposal_manifest.json", args.force)

    stitched = run_dir / "02_current_stitched"
    command = [
        str(python), "-u", str(TRACKING / "stitch_cvat_tracklets.py"),
        "--annotations", str(proposals / "current_raw" / "annotations.xml"),
        "--video", str(args.video.resolve()),
        "--output-dir", str(stitched),
        "--reid-model", str(args.reid.resolve()),
        "--reid-backend", "torchreid",
        "--reid-model-name", "osnet_x0_25",
        "--device", args.device,
        "--endpoint-samples", "4",
        "--max-gap", "5",
        "--min-similarity", "0.75",
        "--max-size-ratio", "2.5",
        "--base-spatial-gate", "2.0",
        "--spatial-gate-per-frame", "0.30",
        "--appearance-weight", "0.60",
        "--spatial-weight", "0.25",
        "--gap-weight", "0.10",
        "--size-weight", "0.05",
        "--use-gmc",
        "--gmc-weight", "0.20",
        "--gmc-base-gate", "0.80",
        "--gmc-gate-per-frame", "0.12",
        "--max-cost", "0.50",
    ]
    run_step("02_stitch_current", command, run_dir, state, stitched / "stitch_manifest.json", args.force)

    fused = run_dir / "03_fused_review"
    command = [
        str(python), "-u", str(TRACKING / "phase1_fuse_and_review.py"),
        "--video", str(args.video.resolve()),
        "--current", str(stitched / "annotations.xml"),
        "--deepoc", str(proposals / "deepocsort" / "annotations.xml"),
        "--output-dir", str(fused),
        "--max-review-clips", str(args.max_review_clips),
    ]
    run_step("03_fuse_and_review", command, run_dir, state, fused / "fusion_manifest.json", args.force)

    handoff = f"""# Ready for CVAT review

Upload `{(fused / 'annotations.xml').resolve()}` with the source video `{args.video.resolve()}`.

Review queue: `{(fused / 'REVIEW_CHECKLIST.md').resolve()}`  
Conflict clips: `{(fused / 'review_clips').resolve()}`  
Raw YOLO cache: `{(proposals / 'detections.parquet').resolve()}`

After correction, export CVAT XML and run:

```powershell
tracking\\run_phase1.ps1 review --run-dir "{run_dir}" --annotations "PATH_TO_EXPORTED_XML" --ground-truth "OPTIONAL_GOLD_XML"
```
"""
    (run_dir / "READY_FOR_CVAT.md").write_text(handoff, encoding="utf-8")
    state["stages"]["04_cvat_handoff"] = {
        "status": "complete",
        "fused_xml": str((fused / "annotations.xml").resolve()),
        "review_checklist": str((fused / "REVIEW_CHECKLIST.md").resolve()),
    }
    save_state(run_dir, state)
    print(handoff)


def review(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    python = args.python.resolve()
    state = read_state(run_dir)
    state.setdefault("stages", {})
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    video = args.video or (Path(state["video"]) if state.get("video") else None)
    if video is None or not video.is_file():
        raise FileNotFoundError("Video is required and was not found in pipeline_state.json")
    fused = run_dir / "03_fused_review" / "annotations.xml"
    qa = run_dir / "04_human_review_qa"
    command = [
        str(python), "-u", str(TRACKING / "audit_cvat_video_annotations.py"),
        str(args.annotations.resolve()),
        "--output-dir", str(qa),
        "--baseline-xml", str(fused),
        "--video", str(video.resolve()),
    ]
    run_step("05_structure_audit", command, run_dir, state, qa / "qa_summary.json", args.force)

    metrics_dir = run_dir / "05_metrics"
    if args.ground_truth:
        if not args.ground_truth.is_file():
            raise FileNotFoundError(args.ground_truth)
        proposals = run_dir / "01_proposals"
        stitched = run_dir / "02_current_stitched" / "annotations.xml"
        command = [
            str(python), "-u", str(TRACKING / "evaluate_cvat_mot.py"),
            "--ground-truth", str(args.ground_truth.resolve()),
            "--prediction", f"human_reviewed={args.annotations.resolve()}",
            "--prediction", f"fused_before_review={fused.resolve()}",
            "--prediction", f"current_stitched={stitched.resolve()}",
            "--prediction", f"deepocsort={proposals / 'deepocsort' / 'annotations.xml'}",
            "--output-dir", str(metrics_dir),
        ]
        run_step("06_gold_evaluation", command, run_dir, state, metrics_dir / "metrics.json", args.force)

    qa_payload = json.loads((qa / "qa_summary.json").read_text(encoding="utf-8"))["current"]
    structural_clean = not any(
        qa_payload.get(key, 0)
        for key in ("invalid_boxes", "out_of_bounds_boxes", "duplicate_track_frame_keyframes")
    )
    lines = [
        "# Phase 1 post-review report",
        "",
        f"Reviewed annotations: `{args.annotations.resolve()}`",
        "",
        f"Structural gate: **{'PASS' if structural_clean else 'FAIL'}**",
        "",
        "| Check | Count |",
        "|---|---:|",
        f"| Invalid boxes | {qa_payload.get('invalid_boxes', 0)} |",
        f"| Out-of-bounds boxes | {qa_payload.get('out_of_bounds_boxes', 0)} |",
        f"| Duplicate track/frame keyframes | {qa_payload.get('duplicate_track_frame_keyframes', 0)} |",
        f"| Large jump candidates | {qa_payload.get('large_one_frame_jump_candidates', 0)} |",
        f"| High-IoU overlap runs | {qa_payload.get('high_iou_overlap_runs', 0)} |",
        "",
    ]
    if (metrics_dir / "REPORT.md").is_file():
        lines.extend(["## Gold evaluation", "", (metrics_dir / "REPORT.md").read_text(encoding="utf-8")])
    else:
        lines.append("Gold evaluation was skipped because `--ground-truth` was not supplied.")
    report = run_dir / "PHASE1_FINAL_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state["stages"]["06_final_report"] = {"status": "complete", "report": str(report.resolve())}
    save_state(run_dir, state)
    print(report)


def status(args: argparse.Namespace) -> None:
    state = read_state(args.run_dir.resolve())
    print(json.dumps(state, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "review":
        review(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
