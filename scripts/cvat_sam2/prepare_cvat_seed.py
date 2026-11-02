"""Create ONE auto-generated seed polygon, never a propagated track result.

Import only into the new isolated SAM2 smoke-test task. The outside marker
ends this seed at frame 76; subsequent propagation must be run in CVAT.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ["HF_HOME"] = "/opt/cvat-sam2-cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

root = Path(__file__).resolve().parents[2]
out = root / "outputs/cvat_sam2_setup/cvat_seed"
out.mkdir(parents=True, exist_ok=True)
video = root / "outputs/tracktrack_reproduction/aria_164557_first60s_v1/input/aria_rgb_rectified_first60s.mp4"
cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_FRAMES, 75)
ok, bgr = cap.read()
cap.release()
assert ok
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device="cuda")
    predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    masks, _, _ = predictor.predict(box=np.array([1674, 1140, 1764, 1312]), multimask_output=False)
mask = masks[0].astype(np.uint8)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
polygon = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True).reshape(-1, 2)
assert len(polygon) >= 3
doc = ET.Element("annotations")
ET.SubElement(doc, "version").text = "1.1"
track = ET.SubElement(doc, "track", id="0", label="person", source="auto")
points = ";".join(f"{x:.2f},{y:.2f}" for x, y in polygon)
for frame, outside in [(75, 0), (76, 1)]:
    ET.SubElement(track, "polygon", frame=str(frame), outside=str(outside),
                  occluded="0", keyframe="1", points=points, z_order="0")
ET.indent(doc)
ET.ElementTree(doc).write(out / "seed_only_frame75.xml", encoding="utf-8", xml_declaration=True)
preview = bgr.copy()
cv2.polylines(preview, [polygon], True, (0, 255, 0), 4)
cv2.imwrite(str(out / "seed_preview.jpg"), cv2.resize(preview, (1280, 960)))
print(f"One seed polygon: {len(polygon)} vertices, frame 75 only; no tracking results exported.")
