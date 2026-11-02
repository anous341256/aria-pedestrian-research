"""Local-only integration smoke test, NOT a gold annotation or accuracy benchmark."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["HF_HOME"] = "/opt/cvat-sam2-cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"

import cv2
import numpy as np
from PIL import Image
import torch
from cvat_sdk.auto_annotation import TrackableShape
from cvat_sdk.masks import decode_mask, encode_mask
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/cvat_sam2_setup/local_smoke"
VIDEO = ROOT / "outputs/tracktrack_reproduction/aria_164557_first60s_v1/input/aria_rgb_rectified_first60s.mp4"
ADAPTER = ROOT / "tools/cvat-models-official/tracker/sam2/func.py"


class Context:
    original_shape_type = "mask"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("official_cvat_sam2", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert torch.cuda.is_available(), "CUDA unavailable"
    tracker = module.create(model_id="facebook/sam2.1-hiera-large", device="cuda")
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 75)
    ok, bgr = cap.read()
    assert ok
    height, width = bgr.shape[:2]
    # Visually located white-shirt pedestrian at frame 75. This is a test prompt,
    # not a human-verified segmentation label. Coordinates in the original video.
    seed_box = [1674, 1140, 1764, 1312]
    image_predictor = SAM2ImagePredictor(tracker._predictor)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_predictor.set_image(rgb)
    masks, scores, _ = image_predictor.predict(box=np.array(seed_box), multimask_output=False)
    mask = masks[0].astype(bool)
    assert mask.any(), "Empty seed mask"
    seed = TrackableShape(type="mask", points=encode_mask(mask))
    pp = tracker.preprocess_image(Context(), Image.fromarray(rgb))
    state = tracker.init_tracking_state(Context(), pp, seed)
    del image_predictor, pp
    writer = cv2.VideoWriter(str(OUT / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (1280, 960))
    assert writer.isOpened()
    records, tiles = [], []
    started = time.perf_counter()
    for offset in range(51):
        if offset:
            ok, bgr = cap.read()
            assert ok
            pp = tracker.preprocess_image(Context(), Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            shape = tracker.track(Context(), pp, state)
            mask = (decode_mask(shape.points, image_width=width, image_height=height)
                    if shape is not None else np.zeros((height, width), dtype=bool))
            del pp
        ys, xs = np.where(mask)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        records.append({"frame_0based": 75 + offset, "mask_pixels": int(mask.sum()), "bbox": box})
        overlay = bgr.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([50, 240, 50])).astype(np.uint8)
        if box:
            cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (50, 240, 50), 4)
        preview = cv2.resize(overlay, (1280, 960))
        cv2.putText(preview, f"LOCAL SMOKE ONLY | frame {75+offset} | not reviewed labels",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        writer.write(preview)
        if offset % 10 == 0:
            tiles.append(cv2.resize(preview, (640, 480)))
            print(f"frame={75+offset}, mask_pixels={int(mask.sum())}", flush=True)
    writer.release()
    cap.release()
    torch.cuda.synchronize()
    result = {
        "purpose": "local adapter/GPU smoke test, not CVAT end-to-end or accuracy evaluation",
        "video": str(VIDEO), "seed_frame_0based": 75, "seed_box": seed_box,
        "seed_mask_score_not_accuracy": float(scores[0]), "frames": len(records),
        "nonempty_masks": sum(r["mask_pixels"] > 0 for r in records),
        "seconds_for_loop_including_render": time.perf_counter() - started,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "optional_connected_components_cuda_extension": False,
        "records": records,
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cv2.imwrite(str(OUT / "contact_sheet.jpg"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    (OUT.parent / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
