"""Short scientific evidence videos; examples are marked post-hoc illustrations."""
import cv2
import numpy as np
from common import OUT,read_json,encoder,finish_encoder

def frames(clip):
    cap=cv2.VideoCapture(str(OUT/'clips'/clip/'rgb.mp4'));ims=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        ims.append(im)
    cap.release();return ims

def box(im,b,col,txt):
    b=(np.asarray(b)*.25).astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),col,2)
    cv2.putText(im,txt,tuple(b[:2]-[0,4]),cv2.FONT_HERSHEY_SIMPLEX,.45,col,1)

def title(im,lines):
    cv2.rectangle(im,(0,0),(im.shape[1],len(lines)*22+6),(0,0,0),-1)
    for i,t in enumerate(lines):cv2.putText(im,t,(8,21+i*22),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)

def identity(out):
    name='test_150809_080';ims=frames(name);auto=read_json(OUT/'clips'/name/'yolo/tracks.json')['tracks'];part=read_json(OUT/'clips'/name/'annotation_v1/tracks.json')['tracks']
    track=next(t for t in auto if t['track_id']==3);old={r['frame_id']:r for r in track['boxes']}
    new={t['track_id']:{r['frame_id']:r for r in t['boxes']} for t in part if t['track_id'] in [1,2]}
    proc,log=encoder(out/'01_identity_repair.mp4',1280,480,5)
    for f in range(54):
        a=cv2.resize(ims[f],(640,480));b=a.copy()
        if f in old:box(a,old[f]['bbox_xyxy'],(0,100,255),'old ID 3')
        for tid,col in [(1,(255,200,0)),(2,(0,240,240))]:
            if f in new[tid]:box(b,new[tid][f]['bbox_xyxy'],col,f'P{tid}')
        title(a,[f'BEFORE: YOLO11l + BoT-SORT | f={f}','ID 3 switches black -> white at frame 16'])
        title(b,['AFTER: SAM identity + YOLO boxes + visual check','Partial silver; independent user review still needed'])
        joined=np.hstack([a,b]);proc.stdin.write(joined.tobytes())
        if f in [10,16,24,40]:cv2.imwrite(str(out/f'identity_f{f:03d}.jpg'),joined)
    finish_encoder(proc,log)

def association(out):
    case=read_json(OUT/'results/association_unique/rotation_helps_1.json');name=case['clip'];ims=frames(name)
    rows={r['frame']:r for r in read_json(OUT/'results/association_unique/pairs.json') if r['clip']==name and r['track']==case['track']}
    lo=max(0,case['frame']-4);hi=min(99,case['frame']+10);proc,log=encoder(out/'02_rotation_matching.mp4',1280,480,3)
    for f in range(lo,hi+1):
        panels=[]
        for m,c in [('none',(0,100,255)),('aria_rotation',(255,200,0))]:
            im=cv2.resize(ims[f],(640,480));r=rows.get(f)
            if r:
                for k,d in enumerate(r['detection_boxes']):box(im,d[:4],(30,230,50) if k==r['true_index'] else (80,80,80),'')
                b=r['detection_boxes'][r[m+'_selected']];box(im,b[:4],c,'selected')
                p=np.asarray(r['predicted_centers'][m])*.25;cv2.drawMarker(im,tuple(p.astype(int)),c,cv2.MARKER_CROSS,14,2)
            title(im,[f'{m} | {name} f={f}',f'Fixed candidate boxes; {"CORRECT" if r and r[m+"_top1"] else "WRONG" if r else "not scored: ambiguous/support"}',
                      'Offline MPS diagnostic, not RGB-only forecast'])
            panels.append(im)
        proc.stdin.write(np.hstack(panels).tobytes())
    finish_encoder(proc,log)

def forecast(out,tag):
    e=read_json(OUT/f'results/forecast/case_{tag}.json');ims=frames(e['clip']);f=e['frame'];proc,log=encoder(out/f'03_forecast_{tag}.mp4',1280,960,5)
    colors={'raw_cv':(50,90,255),'rgb_factor_cv':(255,220,20),'offline_future_rgb':(220,90,240)}
    for k in range(-9,21):
        im=cv2.resize(ims[f+k],(1280,960))
        if k<=0:
            q=np.asarray(e['history'][k+9])/2;cv2.circle(im,tuple(q.astype(int)),8,(60,255,70),-1)
            title(im,[f'POST-HOC {tag} | {e["clip"]} P{e["track"]}',f'Observing history: f={f+k}. Prediction cutoff is f={f}.'])
        else:
            q=np.asarray(e['target'][k-1])/2;cv2.circle(im,tuple(q.astype(int)),9,(60,255,70),-1)
            for j,(m,c) in enumerate(colors.items()):
                p=np.asarray(e['predictions'][m][k-1])/2
                if np.isfinite(p).all() and 0<=p[0]<1280 and 0<=p[1]<960:cv2.drawMarker(im,tuple(p.astype(int)),c,cv2.MARKER_CROSS,24,3)
                else:cv2.putText(im,f'{m}: off-screen prediction',(15,135+j*22),cv2.FONT_HERSHEY_SIMPLEX,.5,c,1)
            title(im,[f'POST-HOC {tag} | forecast +{k/10:.1f}s, no new frames supplied to past-only models',
                'GREEN: prechecked silver reference | RED: raw constant velocity | CYAN: historical RGB factor',
                'PURPLE: actual future RGB compensation, OFFLINE DIAGNOSTIC / NOT deployable',
                'All predictions were fixed at cutoff; this video only reveals the evaluation frames.'])
        proc.stdin.write(im.tobytes())
    finish_encoder(proc,log)

if __name__=='__main__':
    out=OUT/'showcase';out.mkdir(exist_ok=True)
    if not (out/'01_identity_repair.mp4').exists():identity(out)
    if not (out/'02_rotation_matching.mp4').exists():association(out)
    for tag in ['supporting','failure']:
        if not (out/f'03_forecast_{tag}.mp4').exists():forecast(out,tag)
