"""Fine-tune a person detector for one human-in-the-loop flywheel round."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=int, default=10)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        single_cls=True,
        freeze=args.freeze,
        patience=10,
        close_mosaic=5,
        cos_lr=True,
        amp=True,
        cache="disk",
        plots=True,
        save=True,
        save_period=5,
        verbose=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    metrics = {}
    for key, value in getattr(model.trainer, "metrics", {}).items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = str(value)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Round 1 Aria person-detector fine-tune",
        "model_input": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
        "data_yaml": {"path": str(args.data.resolve()), "sha256": sha256(args.data)},
        "configuration": {
            "epochs_requested": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
            "freeze": args.freeze,
            "patience": 10,
            "close_mosaic": 5,
            "cos_lr": True,
            "single_cls": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "outputs": {
            "save_dir": str(save_dir.resolve()),
            "best": str(best.resolve()) if best.is_file() else None,
            "best_sha256": sha256(best) if best.is_file() else None,
            "last": str(last.resolve()) if last.is_file() else None,
        },
        "sanity_validation_metrics": metrics,
        "metric_warning": (
            "Validation frames are from the same recording as training and are not an "
            "independent scientific evaluation. Use a held-out recording for claims."
        ),
    }
    (save_dir / "flywheel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
