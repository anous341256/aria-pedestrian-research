"""Deterministic first completed frame per recording for visual registration audit."""
import sys,json
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008'
sys.path.insert(0,str(ROOT/'scripts/aria_stereo_validation_20261008'))
from analyze import project,decode,color_depth,text
plan=json.loads((OUT/'extraction_plan.json').read_text());dest=OUT/'depth_review';dest.mkdir(exist_ok=True)
selected=[]
for rec in sorted({e['recording'] for e in plan['entries']}):
    ee=sorted([e for e in plan['entries'] if e['recording']==rec],key=lambda e:e['rgb_timestamp_ns'])
    e=next((e for e in ee if (OUT/'frames'/e['key']/'complete.json').exists()),None)
    if e is None:continue
    # Repeated runs only overwrite a contact image, never measurements or selection.
    p=dict(np.load(OUT/'frames'/e['key']/'depth.npz'));cap=cv2.VideoCapture(str(ROOT/e['input']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,e['reference_frame']);ok,rgb=cap.read();cap.release();assert ok
    disp=p['disparity_px'];yy,xx=np.indices(disp.shape);xr=xx-disp;xi=np.clip(xr,0,511).astype(int);support=p['support_left']&p['support_right'][yy,xi]&(xr>=0)&(xr<=511)
    pts=project(p,p['depth_m'],support,p['valid']);small=cv2.resize(rgb,(640,480));over=small.copy();colors=color_depth(pts['z_rgb'].reshape(-1,1)).reshape(-1,3)
    for j in np.flatnonzero(pts['valid'])[::2]:cv2.circle(over,(int(pts['u'][j]/4),int(pts['v'][j]/4)),1,tuple(map(int,colors[j])),-1)
    over=cv2.addWeighted(small,.65,over,.35,0)
    for t in e['targets']:
        mask=cv2.resize(decode(t['rle']),(640,480),interpolation=cv2.INTER_NEAREST);c,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,c,-1,(255,255,255),1)
    d=color_depth(p['depth_m']);d[~p['valid']]=0
    im=np.full((570,1280,3),(32,24,16),np.uint8);im[45:525,:640]=small;im[45:525,640:]=over
    text(im,rec+' | RGB / projected depth + required masks',12,26,.62)
    values=json.loads((OUT/'frames'/e['key']/'measurements.json').read_text());labels=[r['target_key']+(': %.2fm'%r['z_median_m'] if r['z_median_m'] else ': missing')+(' [accepted]' if r['usable'] else ' [rejected]') for r in values]
    text(im,'; '.join(labels)[:170],12,553,.43)
    cv2.imwrite(str(dest/(rec+'.jpg')),im);selected.append(dict(recording=rec,key=e['key'],values=values))
(dest/'selection.json').write_text(json.dumps(selected,indent=2));print('Created',len(selected),'recording review cards')
