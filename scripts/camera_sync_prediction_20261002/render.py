"""Three-way synchronized playback using the same fixed examples and seed as V1."""
from shared import *
import cv2,subprocess
from PIL import Image,ImageDraw,ImageFont

PRES=OUT/'presentation'
FONT='C:/Windows/Fonts/msyh.ttc'
FFMPEG='E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
DISPLAY=['old_old','old_new','new_new']
CAPTIONS=['旧相机：网络与投影均为旧版','仅最后重投影换成新相机','网络条件与重投影同步换新']


def label(image,lines):
    p=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(p)
    for x,y,txt,size in lines:d.text((x,y),txt,font=ImageFont.truetype(FONT,size),fill='white')
    return cv2.cvtColor(np.asarray(p),cv2.COLOR_RGB2BGR)


def point(im,xy,color,r=7):
    p=np.array(xy)*.25
    if not np.isfinite(p).all():return
    if (p>=[0,0]).all() and (p<[640,480]).all():cv2.circle(im,tuple(p.astype(int)),r,color,2,cv2.LINE_AA)
    else:cv2.drawMarker(im,tuple(np.clip(p,[10,10],[629,469]).astype(int)),color,cv2.MARKER_TILTED_CROSS,14,2)


def mask(rle):
    x1,y1,x2,y2=map(int,rle[-4:]);counts=np.asarray(rle[:-4],np.int64)
    value=np.repeat(np.arange(len(counts))%2,counts).reshape(y2-y1+1,x2-x1+1).astype(np.uint8)
    out=np.zeros((1920,2560),np.uint8);out[y1:y2+1,x1:x2+1]=value
    return cv2.resize(out,(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)


def main():
    PRES.mkdir(parents=True,exist_ok=True)
    data=np.load(OLD/'dataset.npz');meta=read(OLD/'samples.json');lookup={m['key']:i for i,m in enumerate(meta)};pred={}
    for f in read(OLD/'folds.json'):
        z=np.load(OUT/f'fold_{f["fold"]}/seed_17/comp_dual/predictions.npz')
        for j,i in enumerate(z['indices']):pred[int(i)]={a:z[a][j] for a in DISPLAY}
    fixed=read(OLD/'presentation/selection.json')['fixed']
    rows=read(OUT/'metrics.json');scores={}
    for key in lookup:
        scores[key]={a:float(np.mean([r['metrics']['ADE15'] for r in rows if r['variant']=='comp_dual' and r['key']==key and r['arm']==a])) for a in DISPLAY}
    best=min(scores,key=lambda k:scores[k]['new_new']-scores[k]['old_old'])
    worst=max(scores,key=lambda k:scores[k]['new_new']-scores[k]['old_old'])
    syncbest=min(scores,key=lambda k:scores[k]['new_new']-scores[k]['old_new'])
    syncworst=max(scores,key=lambda k:scores[k]['new_new']-scores[k]['old_new'])
    selected=list(dict.fromkeys([best,worst,syncbest,syncworst]))
    selection=dict(display_seed=17,fixed=fixed,selected=selected,selected_best_vs_old=best,selected_worst_vs_old=worst,
                   selected_best_vs_projection_only=syncbest,selected_worst_vs_projection_only=syncworst,
                   fixed_rule='Exactly the nine V1 first eligible identity / median cutoff examples',
                   special_rule='Post-score extrema by three-seed mean ADE; not representative averages',scores=scores)
    write(PRES/'selection.json',selection)
    frames={};masks={}
    for cid in sorted({meta[lookup[k]]['clip'] for k in fixed+selected}):
        cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4'));ff=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            ff.append(cv2.resize(im,(640,480)))
        cap.release();frames[cid]=ff
        masks[cid]={(r['id'],r['frame']):r for r in gz(ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz')}
    expected={}
    for name,keys in [('fixed_9_examples',fixed),('selected_gain_and_loss',selected)]:
        dest=PRES/f'{name}.mp4';expected[dest.name]=len(keys)*42
        if dest.exists():continue
        log=dest.with_suffix('.ffmpeg.log').open('wb')
        proc=subprocess.Popen([FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x620','-r','10','-i','pipe:0','-an',
                               '-c:v','libx264','-threads','3','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE,stderr=log)
        for key in keys:
            i=lookup[key];m=meta[i]
            for j,f in enumerate(range(m['cutoff']-5,m['cutoff']+16)):
                im=frames[m['clip']][f].copy();r=masks[m['clip']].get((m['id'],f))
                if r:
                    mm=mask(r['rle']);im[mm]=im[mm]*.83+.17*np.array([255,220,80])
                canvas=np.zeros((620,1920,3),np.uint8)
                kind='固定规则案例' if name=='fixed_9_examples' else '事后选取的改善／退化案例'
                texts=[(12,8,f'{key} | {kind} | 固定人物权重 · 种子17 · 半速播放',25)]
                for column,arm in enumerate(DISPLAY):
                    panel=im.copy();truth=data['xy'][i,j];point(panel,truth,(255,255,255),6)
                    if j>=6:
                        xy=pred[i][arm][j-6];point(panel,xy,(0,170,255),9)
                        if np.isfinite(xy).all():
                            a=np.clip(truth*.25,-5000,5000).astype(int);b=np.clip(xy*.25,-5000,5000).astype(int)
                            cv2.line(panel,tuple(a),tuple(b),(0,170,255),2,cv2.LINE_AA)
                    x=column*640;canvas[88:568,x:x+640]=panel
                    ade=metrics(pred[i][arm],data['xy'][i,6:])['ADE15']
                    texts.append((x+9,53,f'{CAPTIONS[column]} | ADE {ade:.1f}px',21))
                stage='读取过去 0.5 秒' if j<6 else f'固定预测：未来 {(j-5)/10:.1f} 秒'
                texts.append((12,586,stage+' | 白圈：SAM参考中心；橙圈：预测中心。三组使用相同视频，预测起点后不接收新观测。',22))
                canvas=label(canvas,texts);proc.stdin.write(canvas.tobytes());proc.stdin.write(canvas.tobytes())
                if j==20:cv2.imwrite(str(PRES/f'{key}_comparison.jpg'),canvas)
        proc.stdin.close();assert proc.wait(timeout=60)==0;log.close()
    checks=[]
    for name,count in expected.items():
        cap=cv2.VideoCapture(str(PRES/name));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(620,1920);n+=1
        cap.release();assert reported==n==count
        checks.append(dict(file=name,frames=n,seconds=n/10,passed=True))
    write(PRES/'verification.json',dict(passed=True,videos=checks));print(checks);print('Special cases',selected)


if __name__=='__main__':main()
