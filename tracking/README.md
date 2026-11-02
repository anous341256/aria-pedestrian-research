# JapanWalk pedestrian pre-annotation

This pipeline uses a COCO-pretrained pedestrian detector and BoT-SORT to make
track proposals for manual correction. The generated identities are **not**
ground truth until reviewed in CVAT.

## Recommended pilot workflow

Create a Python 3.10 environment and install the pinned runtime:

```powershell
pip install -r tracking/requirements.txt
```

Run a short 300-frame smoke test first:

```powershell
python tracking/auto_annotate_people.py `
  --video D:/path/to/japanwalk.mp4 `
  --output-dir E:/New_research/outputs/japanwalk_pilot_smoke `
  --max-frames 300 `
  --preview
```

Then process the selected pilot clip:

```powershell
python tracking/auto_annotate_people.py `
  --video D:/path/to/japanwalk.mp4 `
  --output-dir E:/New_research/outputs/japanwalk_pilot `
  --preview
```

The default is the locally cached COCO checkpoint at
`models/tracking/yolo11l.pt`. On limited VRAM, use
`--model yolo11m.pt --imgsz 960`; Ultralytics downloads that alternative on
first use. Only COCO class 0 (`person`) is retained.

Import `annotations.xml` together with the source video into a CVAT video task.
Correct missed boxes, false boxes, ID switches, fragments, occlusion and
outside states. Do not train the trajectory model from the raw automatic file.

After correction, run the existing ID audit against the exported tracks and
require the acceptance gates in `reports/JapanWalk_ID_audit/ID_QUALITY_AUDIT.md`.

## License note

Ultralytics code and tracker configuration are AGPL-3.0. COCO images and model
weights have their own terms. Review licensing separately before closed-source
commercial distribution.
