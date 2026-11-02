"""Export only the rerun initial YOLO11l/BoT-SORT label for CVAT Append."""
import json
import collections
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import export_cvat_comparison as writer
from audit_cvat_video_annotations import expand_track

BASE=Path('/mnt/e/New_research/outputs/tracktrack_reproduction/aria_164557_first60s_v1')
RUN=BASE/'yolo11l_botsort_round0_rectified_v1'
OUT=RUN/'cvat_append'
payload=json.loads((RUN/'tracks.json').read_text())
assert payload['video']['processed_frames']==600
assert payload['method']['confidence_threshold']==.15
assert payload['method']['imgsz']==1280
assert payload['method']['nms_iou_threshold']==.7
assert payload['environment']['ultralytics']=='8.4.41'
OUT.mkdir(exist_ok=True)
video=BASE/'input/aria_rgb_rectified_first60s.mp4'
meta=writer.video_metadata(video)
label=writer.label_spec('yolo11l_botsort_initial','#33cc66',True)
root=writer.annotation_root(meta,video,[label])
tracks={}
expected={}
for t in payload['tracks']:
    tid=int(t['track_id'])
    rows=[]
    for r in t['boxes']:
        box,_=writer.clip_box(r['bbox_xyxy'],meta['width'],meta['height'])
        rows.append(dict(frame=int(r['frame_id']),box=box,score=float(r['confidence'])))
    rows.sort(key=lambda r:r['frame'])
    assert len({r['frame'] for r in rows})==len(rows)
    tracks[tid]=rows
    expected[tid+200000]={r['frame']:r['box'] for r in rows}
writer.add_real_tracks(root,tracks,meta,label['name'])
for t in root.findall('track'):
    t.set('id',str(int(t.get('id'))+200000))
xml=OUT/'yolo11l_botsort_initial_only.xml'
writer.save_xml(xml,root)
actual={}
for t in ET.parse(xml).getroot().findall('track'):
    keys=[dict(frame=int(b.get('frame')),outside=b.get('outside')=='1',
               box=[float(b.get(k)) for k in ('xtl','ytl','xbr','ybr')]) for b in t.findall('box')]
    actual[int(t.get('id'))]=expand_track(keys,599)
assert actual.keys()==expected.keys()
for tid,frames in actual.items():
    assert frames.keys()==expected[tid].keys()
    for f,box in frames.items():
        np.testing.assert_allclose(box,expected[tid][f],atol=5.01e-7,rtol=0)
by_frame=collections.Counter(r['frame'] for rows in tracks.values() for r in rows)
report=dict(label=label['name'],video_sha256=writer.digest(video),xml_sha256=writer.digest(xml),
            original_run='outputs/aria_annotations/round0/unknown_20260803_164557_first60s',
            comparison_scope='Original pipeline rerun on the identical rectified video, NOT transformed historical annotations',
            counts=dict(tracks=len(tracks),boxes=sum(map(len,tracks.values())),frames_with_boxes=len(by_frame),
                        median_visible_frames_per_id=float(np.median([len(v) for v in tracks.values()]))),
            method=payload['method'],environment=payload['environment'],
            visible_frame_id_and_coordinates_roundtrip_exact=True,
            frame_counts_70_85={str(f):by_frame[f] for f in range(70,86)},
            provenance={str(p):writer.digest(p) for p in [RUN/'tracks.json',RUN/'gt/gt.txt',
                 Path('/mnt/e/New_research/models/tracking/yolo11l.pt'),
                 Path('/mnt/e/New_research/tracking/botsort_reid.yaml'),
                 Path('/mnt/e/New_research/tracking/auto_annotate_people.py')]})
writer.write_json(OUT/'export_summary.json',report)
writer.write_json(OUT/'label.json',label)
print(json.dumps(report,indent=2))
