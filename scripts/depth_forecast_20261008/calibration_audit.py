"""Representative per-recording rectification checks independent of depth scores."""
from pathlib import Path
import json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008'
plan=json.loads((OUT/'extraction_plan.json').read_text());rows=[]
for rec in sorted({e['recording'] for e in plan['entries']}):
    ee=sorted([e for e in plan['entries'] if e['recording']==rec],key=lambda e:e['rgb_timestamp_ns'])
    for e in [ee[0],ee[len(ee)//2],ee[-1]]:
        path=OUT/'frames'/e['key']/'depth.npz'
        if not path.exists():rows.append(dict(key=e['key'],missing=True));continue
        p=np.load(path);ref=np.load(ROOT/e['input']/'reference.npz');R=(p['T_world_cv']@np.linalg.inv(p['T_rgb_cv']))[:3,:3]
        d=R.T@ref['R'][e['reference_frame']];angle=float(np.degrees(np.linalg.norm(cv2.Rodrigues(d)[0])))
        sift=cv2.SIFT_create(nfeatures=1500);k1,d1=sift.detectAndCompute(p['left'],None);k2,d2=sift.detectAndCompute(p['right'],None);dy=[]
        if d1 is not None and d2 is not None:
            match=cv2.BFMatcher();back=match.match(d2,d1)
            for a,b in match.knnMatch(d1,d2,k=2):
                if a.distance<.7*b.distance and back[a.trainIdx].trainIdx==a.queryIdx:dy.append(abs(k1[a.queryIdx].pt[1]-k2[a.trainIdx].pt[1]))
        rows.append(dict(key=e['key'],recording=rec,matches=len(dy),vertical_median_px=float(np.median(dy)) if dy else None,vertical_p90_px=float(np.percentile(dy,90)) if dy else None,rgb_pose_disagreement_deg=angle,baseline_m=float(p['baseline_m'])))
(OUT/'calibration_audit.json').write_text(json.dumps(rows,indent=2));print('Checked',len(rows),'predefined first/middle/last frames')
