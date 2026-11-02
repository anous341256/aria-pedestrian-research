"""Extract an exact frame-aligned prefix from a video for CVAT import."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path)
    parser.add_argument("output_video", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()

    if not args.input_video.is_file():
        raise FileNotFoundError(args.input_video)
    if args.output_video.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_video}")
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    import cv2

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create {args.output_video}")

    written = 0
    while written < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()
    if written != args.frames:
        raise RuntimeError(f"Requested {args.frames} frames but wrote {written}")
    print(f"wrote {written} frames at {fps:g} FPS: {args.output_video}")


if __name__ == "__main__":
    main()
