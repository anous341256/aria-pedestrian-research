"""Six synchronized panels: identical frames, labels, horizons and seed."""
from shared import *
import cv2,subprocess
from PIL import Image,ImageDraw,ImageFont
PRES=OUT/'presentation'
FONT='C:/Windows/Fonts/msyh.ttc'
FFMPEG='E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'


def label(im,lines):
    p=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(p)
    for x,y,text,size in lines:d.text((x,y),text,font=ImageFont.truetype(FONT,size),fill='white')
    return cv2.cvtColor(np.asarray(p),cv2.COLOR_RGB2BGR)


def point(im,xy,color,r):
    p=np.asarray(xy)*.25
    if not np.isfinite(p).all():return
    if (p>=[0,0]).all() and (p<[640,480]).all():cv2.circle(im,tuple(p.astype(int)),r,color,2,cv2.LINE_AA)
    else:cv2.drawMarker(im,tuple(np.clip(p,[10,10],[629,469]).astype(int)),color,cv2.MARKER_TILTED_CROSS,14,2)


def decode_mask(rle):
    x1,y1,x2,y2=map(int,rle[-4:]);counts=np.asarray(rle[:-4],np.int64)
    value=np.repeat(np.arange(len(counts))%2,counts).reshape(y2-y1+1,x2-x1+1).astype(np.uint8)
    out=np.zeros((1920,2560),np.uint8);out[y1:y2+1,x1:x2+1]=value
    return cv2.resize(out,(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    PRES.mkdir(parents=True,exist_ok=True);data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json')
    lookup={m['key']:i for i,m in enumerate(meta)};pred={};scores={}
    for fold in read(OUT/'folds.json'):
        for v in VARIANTS:
            z=np.load(OUT/f'fold_{fold["fold"]}/seed_17/{v}/predictions_reviewed.npz')
            for j,i in enumerate(z['indices']):
                pred.setdefault(int(i),{})[v]=z['pred'][j]
                scores.setdefault(meta[i]['key'],{})[v]=metrics(z['pred'][j],data['xy'][i,P:],z['depth'][j])['ADE15']
    cases=read(OUT/'fixed_cases.json');fixed=cases['old']+cases['new'];selected=[];reasons={}
    for a,b in [('D_bidirectional','A_early_fusion'),('F_interacting_geometry','C_separated')]:
        for name,func in [('gain',min),('loss',max)]:
            key=func([k for k in scores if meta[lookup[k]]['source_version']=='new82'],key=lambda k:scores[k][a]-scores[k][b]);reasons[a+'_vs_'+b+'_'+name]=key
            if key not in selected:selected.append(key)
    write(PRES/'selection.json',dict(fixed=fixed,selected=selected,selection_reason=reasons,display_seed=17,scores=scores,
                                    fixed_rule='Original nine plus nine new cases frozen before fitting; first eligible identity and median cutoff per clip.',
                                    selected_rule='Post-score seed-17 new-cohort gain/loss extrema; not representative average performance.'))
    frames={};masks={}
    for cid in sorted({meta[lookup[k]]['clip'] for k in fixed+selected}):
        video=ROOT/(f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4' if int(cid)<=10 else f'outputs/sam2_g1_expansion_20261005/{cid}/input/rgb.mp4');cap=cv2.VideoCapture(str(video));ff=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            ff.append(cv2.resize(im,(640,480)))
        cap.release();frames[cid]=ff
        labelpath=ROOT/(f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz' if int(cid)<=10 else f'outputs/sam2_g1_expansion_20261005/{cid}/final/masks_rle.jsonl.gz');masks[cid]={(r['id'],r['frame']):r for r in gz(labelpath)}
    checks=[]
    for name,keys in [('fixed_old9_six_groups',cases['old']),('fixed_new9_six_groups',cases['new']),('selected_new_gain_loss_six_groups',selected)]:
        dest=PRES/f'{name}.mp4';assert not dest.exists(),'Preserve rendered outputs'
        log=dest.with_suffix('.ffmpeg.log').open('wb')
        proc=subprocess.Popen([FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1200','-r','10','-i','pipe:0','-an',
                               '-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE,stderr=log)
        for key in keys:
            i=lookup[key];m=meta[i]
            for j,frame in enumerate(range(m['cutoff']-5,m['cutoff']+16)):
                im=frames[m['clip']][frame].copy();r=masks[m['clip']].get((m['id'],frame))
                if r:
                    mask=decode_mask(r['rle']);im[mask]=im[mask]*.84+.16*np.array([255,220,80])
                canvas=np.full((1200,1920,3),(27,23,19),np.uint8)
                texts=[(16,8,f'{key} | {"固定规则案例" if name.startswith("fixed") else "事后选择改善／退化案例"} | 同一输入 · 种子17 · 半速',27)]
                for c,v in enumerate(VARIANTS):
                    x=(c%3)*640;y=50+(c//3)*552
                    texts.append((x+10,y,NAMES[v]+f' | ADE {scores[key][v]:.1f}px',24))
                    panel=im.copy();truth=data['xy'][i,j];point(panel,truth,(255,255,255),6)
                    if j>=6:
                        xy=pred[i][v][j-6];point(panel,xy,(0,170,255),9)
                        if np.isfinite(xy).all():
                            a=np.clip(truth*.25,-5000,5000).astype(int);b=np.clip(xy*.25,-5000,5000).astype(int)
                            cv2.line(panel,tuple(a),tuple(b),(0,170,255),2,cv2.LINE_AA)
                    canvas[y+42:y+522,x:x+640]=panel
                stage='观察过去 0.5 秒' if j<6 else f'固定预测：未来 {(j-5)/10:.1f} 秒'
                texts.append((16,1160,stage+' | 白圈：SAM参考中心；橙圈：预测中心；线段：误差。预测开始后不再输入观测。',25))
                canvas=label(canvas,texts)
                proc.stdin.write(canvas.tobytes());proc.stdin.write(canvas.tobytes())
                if j==20:cv2.imwrite(str(PRES/f'{key}_six_groups.jpg'),canvas)
        proc.stdin.close();assert proc.wait(timeout=60)==0;log.close()
        cap=cv2.VideoCapture(str(dest));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(1200,1920,3);n+=1
        cap.release();assert n==reported==len(keys)*42
        checks.append(dict(file=dest.name,frames=n,seconds=n/10,decoded_all=True))
        print(dest.name,'verified',n,'frames',flush=True)
    write(PRES/'video_checks.json',dict(passed=True,checks=checks))


if __name__=='__main__':main()
