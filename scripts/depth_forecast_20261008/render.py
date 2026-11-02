"""Fixed six test cases; same seed, observations and future frames in both panels."""
import sys,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008';TRAIN=OUT/'forecast'
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder,finish_encoder
from PIL import Image,ImageDraw,ImageFont
def read(p):return json.loads(Path(p).read_text())
def text(im,s,x,y,scale=.6,c=(235,235,235)):
    cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,c,1,cv2.LINE_AA)
def point(q):return tuple(np.clip(np.rint(np.asarray(q)/4),-10000,10000).astype(int))

def main():
    media=OUT/'presentation';media.mkdir(exist_ok=True);cases=read(OUT/'fixed_cases.json')['cases'];meta=read(ROOT/'outputs/transformer_uncovered200_20261006/data/samples.json');data=np.load(ROOT/'outputs/transformer_uncovered200_20261006/data/dataset.npz');depth=np.load(OUT/'depth_all21.npz')['depth'];cards=[]
    enc,log=encoder(media/'depth_ablation_fixed6.mp4',1280,620,10,20)
    try:
        for case in cases:
            i=case['index'];m=meta[i];fold=case['fold'];pred={}
            for arm in ['no_depth','with_depth']:
                p=np.load(TRAIN/f'fold_{fold}/seed_17/{arm}/predictions.npz');j=int(np.where(p['indices']==i)[0][0]);pred[arm]={k:p[k][j] for k in ['image','image_error','angle_deg','residual_error']}
            cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][0])
            valid=int(depth[i,:6,4].sum());last=depth[i,5,0] if depth[i,5,4] else None
            for f in range(21):
                ok,rgb=cap.read();assert ok;rgb=cv2.resize(rgb,(640,480));im=np.full((620,1280,3),(32,24,16),np.uint8)
                text(im,f'Fixed test case {m["key"]} | seed 17 | observed 0.5s -> future 1.5s',16,27,.65)
                for j,arm in enumerate(['no_depth','with_depth']):
                    x=j*640;panel=rgb.copy();truth=point(data['xy'][i,f]);cv2.drawMarker(panel,truth,(240,230,30),cv2.MARKER_CROSS,18,2)
                    if f>=6:
                        q=point(pred[arm]['image'][f-6]);cv2.circle(panel,q,7,(65,80,255),2);cv2.line(panel,truth,q,(65,80,255),1)
                    im[65:545,x:x+640]=panel
                    text(im,'No depth' if j==0 else 'With historical depth',x+16,54,.6)
                    p=pred[arm];text(im,f"ADE {p['image_error'].mean():.1f}px | FDE {p['image_error'][-1]:.1f}px",x+16,570,.6)
                    text(im,f"Residual ADE {p['residual_error'].mean():.1f}px | camera {p['angle_deg'].mean():.2f} deg",x+16,594,.5)
                phase=f'Observed t={(f-5)/10:.1f}s' if f<6 else f'Forecast +{(f-5)/10:.1f}s'
                label=f'{valid}/6 historical depth valid';label+=f' | last Z {last:.2f}m' if last is not None else ' | last Z unavailable'
                cv2.rectangle(im,(630,65),(1280,91),(32,24,16),-1);text(im,label,650,84,.48)
                text(im,phase,16,84,.5);text(im,'Cyan: SAM reference center | Red: prediction',16,615,.46)
                enc.stdin.write(im.tobytes())
                if f==20:cv2.imwrite(str(media/f'{fold}_{m["key"]}_final.jpg'),im);cards.append(im)
            cap.release()
    finally:finish_encoder(enc,log)
    cv2.imwrite(str(media/'fixed6_contact_sheet.jpg'),np.concatenate(cards,axis=0))
    cap=cv2.VideoCapture(str(media/'depth_ablation_fixed6.mp4'));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(620,1280,3);n+=1
    cap.release();assert n==126
    (media/'checks.json').write_text(json.dumps(dict(decoded_frames=n,width=1280,height=620,fps=10,selection='predeclared six cases, same seed17')))
    print('RENDERED',n,'frames')

if __name__=='__main__':main()
