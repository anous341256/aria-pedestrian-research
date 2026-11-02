"""Independent sparse background diagnostic; never fed to forecasting."""
import sys
from pathlib import Path
import cv2
import numpy as np
from analyze import ROOT,SRC,OUT,read,write,project
sys.path.insert(0,str(ROOT/'scripts'))
from aria_rectified_rotation_background import background_mask,correspondences

def main():
    cv2.setNumThreads(4);rows=[]
    for spec in read(OUT/'candidates.json'):
        sid=spec['recording'];ref=np.load(OUT/f'{sid}_reference.npz');K=ref['K'];rs=ref['R'];ki=np.linalg.inv(K)
        tracks=read(SRC/sid/'sam2_raw_v1/tracks.json')['tracks'];boxes={}
        for t in tracks:
            for r in t['frames']:
                if r.get('bbox_xyxy'):boxes.setdefault(r['frame'],[]).append(r['bbox_xyxy'])
        cap=cv2.VideoCapture(str(SRC/sid/'rgb.mp4'))
        for f in range(spec['start'],spec['end'],10):
            ims=[];masks=[]
            cap.set(cv2.CAP_PROP_POS_FRAMES,f)
            for g in [f,f+1]:
                ok,im=cap.read();assert ok
                gray=cv2.cvtColor(cv2.resize(im,(1280,960),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
                ims.append(gray);masks.append(background_mask(gray,np.array(boxes.get(g,[])).reshape(-1,4),.5))
            a,b,n=correspondences(*ims,*masks,.5)
            row={'key':spec['key'],'frame':f,'candidate_points':len(a),'initial_points':n,'status':'valid'}
            if len(a)<10:row['status']='too_few_correspondences'
            elif not np.isfinite(rs[[f,f+1]]).all():row['status']='invalid_pose'
            else:
                rot=rs[f+1].T@rs[f]
                q,v=project(a,K@rot@ki);wrong,w=project(a,K@rot.T@ki)
                good=v&w
                row['common_points']=int(good.sum())
                if good.sum()<10:row['status']='too_few_projectable_points'
                else:
                    for name,p in [('raw',a),('rotation',q),('inverse_rotation_control',wrong)]:
                        e=np.linalg.norm(p[good]-b[good],axis=1)
                        row[name]=float(np.median(e))
            rows.append(row)
        cap.release();print('BACKGROUND',spec['key'],flush=True)
    valid=[r for r in rows if r['status']=='valid']
    summary={'sampled_pairs':len(rows),'valid_pairs':len(valid),'points':sum(r['common_points'] for r in valid),
        'median_pair_median_px':{m:float(np.median([r[m] for r in valid])) for m in ['raw','rotation','inverse_rotation_control']},
        'fraction_pairs_rotation_better':float(np.mean([r['rotation']<r['raw'] for r in valid])),
        'per_track':{s['key']:{m:float(np.median([r[m] for r in valid if r['key']==s['key']])) for m in ['raw','rotation','inverse_rotation_control']} for s in read(OUT/'candidates.json')},
        'caveat':'KLT background candidates, not verified static ground truth. SAM partial person exclusions can miss other moving objects; no correspondences are selected by rotation residual.'}
    write(OUT/'background_pairs.json',rows);write(OUT/'background_summary.json',summary)
    print(summary,flush=True)

if __name__=='__main__':main()
