"""Check delivered video completeness and logged prediction/observation semantics."""
from pathlib import Path
import sys,gzip,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_v6_v8_20261001';P=OUT/'presentation/prediction_boxes'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import read_json,write_json,digest
manifest=read_json(P/'manifest.json');media=[];count_distance=0
for c in manifest['clips']:
    cid=c['clip'];n=c['frames']
    for v in ['v6','v8']:
        with gzip.open(OUT/cid/v/'decisions.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
        for row in rows:
            for d in row['branches']['motion_with_camera']['decisions'].values():
                if d.get('innovation_px') is not None:
                    a=np.asarray(d['prior']['box']);b=np.asarray(d['observation_box'])
                    delta=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
                    assert abs(delta-d['innovation_px'])<1e-7;count_distance+=1
    media.extend([(P/f'{cid}_v6_predictions.mp4',n,1760,1120),(P/f'{cid}_v6_v8_predictions.mp4',n,1920,1000)])
media.extend([(P/'all_10_v6_predictions.mp4',530,1760,1120),(P/'all_10_v6_v8_predictions.mp4',530,1920,1000)])
results=[]
for path,n,w,h in media:
    cap=cv2.VideoCapture(str(path));assert abs(cap.get(cv2.CAP_PROP_FPS)-10)<1e-6
    decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(h,w);decoded+=1
    cap.release();assert decoded==n
    results.append(dict(file=path.name,frames=n,fps=10,size=[w,h],sha256=digest(path)))
write_json(P/'verification.json',dict(status='passed',source_predictions_and_masks_unchanged=manifest['sources_unchanged'],
    prediction_observation_distance_checks=count_distance,videos=results,
    visual_review='Reviewed 01 middle frame, 02 final comparison and 04 final single-version diagnostic for boxes, masks, status labels, clipping and layout.'))
print(f'Verified {len(results)} complete videos; {count_distance} prediction-observation distance checks.')
