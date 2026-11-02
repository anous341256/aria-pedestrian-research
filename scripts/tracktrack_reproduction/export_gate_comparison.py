"""Export two gate-ablation tracks as separate CVAT labels; no server calls."""
import collections
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import export_cvat_comparison as writer
from audit_cvat_video_annotations import expand_track

BASE=Path('/mnt/e/New_research/outputs/tracktrack_reproduction/aria_164557_first60s_v1')
ROOT=BASE/'iou_gate005_v1'
OUT=ROOT/'cvat_comparison'
OUT.mkdir(exist_ok=True)
video=BASE/'input/aria_rgb_rectified_first60s.mp4'
meta=writer.video_metadata(video)
labels=[writer.label_spec('tracktrack_iou010','#00bcd4',True),
        writer.label_spec('tracktrack_iou005','#ff9800',True)]
root=writer.annotation_root(meta,video,labels)
expected={}
counts={}
for mode,label,offset in [('gate010',labels[0]['name'],0),('gate005',labels[1]['name'],100000)]:
    tracks=collections.defaultdict(list)
    for row in np.loadtxt(ROOT/mode/'tracks.txt',delimiter=',',ndmin=2):
        f,tid=map(int,row[:2]); x,y,w,h,score=row[2:7]
        box,_=writer.clip_box([x,y,x+w,y+h],meta['width'],meta['height'])
        tracks[tid].append(dict(frame=f-1,box=box,score=float(score)))
    before=len(root.findall('track'))
    writer.add_real_tracks(root,tracks,meta,label)
    # XML track identifiers must be globally unique; original IDs remain in attributes.
    for node in root.findall('track')[before:]:
        tid=int(node.get('id'))
        node.set('id',str(tid+offset))
        expected[tid+offset]=set(r['frame'] for r in tracks[tid])
    counts[label]=dict(ids=len(tracks),visible_boxes=sum(map(len,tracks.values())))
path=OUT/'gate010_vs_gate005.xml'
writer.save_xml(path,root)
parsed=ET.parse(path).getroot()
actual={}
for t in parsed.findall('track'):
    keys=[dict(frame=int(b.get('frame')),outside=b.get('outside')=='1',
               box=[float(b.get(k)) for k in ('xtl','ytl','xbr','ybr')]) for b in t.findall('box')]
    actual[int(t.get('id'))]=set(expand_track(keys,599))
assert actual==expected
assert sum(len(v) for v in actual.values())==4244
writer.write_json(OUT/'labels.json',labels)
writer.write_json(OUT/'export_summary.json',dict(counts=counts,visible_frame_id_pairs_exact=True,
                  xml_sha256=writer.digest(path),video_sha256=writer.digest(video),
                  frame_count=600,frame_convention='CVAT 0..599, 10fps',
                  cross_run_ids_are_not_person_correspondences=True))
print(json.dumps(counts))
