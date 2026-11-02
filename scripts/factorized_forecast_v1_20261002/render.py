"""Fixed-case synchronized old / matched coupled / separated forecast videos."""
from shared import *
import cv2,subprocess,importlib.util
from PIL import Image,ImageDraw,ImageFont

PRES=OUT/'presentation'
FONT='C:/Windows/Fonts/msyh.ttc'
FFMPEG='E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
DISPLAY=['old_original','coupled_dual/full','separated_dual/full']
CAPTIONS=['保存的旧版','匹配耦合对照 · 新相机','独立分离双任务 · 新相机']


def label(image,lines):
    p=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(p)
    for x,y,txt,size in lines:d.text((x,y),txt,font=ImageFont.truetype(FONT,size),fill='white')
    return cv2.cvtColor(np.asarray(p),cv2.COLOR_RGB2BGR)


def point(im,xy,color,r=7):
    p=np.asarray(xy)*.25
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
    data=np.load(OLD/'dataset.npz');meta=read(OLD/'samples.json');lookup={m['key']:i for i,m in enumerate(meta)};pred={};depth={}
    for fold in read(OUT/'folds.json'):
        f=fold['fold'];original=np.load(OLD/f'fold_{f}/seed_17/comp_dual/predictions_reviewed.npz')
        sep=np.load(OUT/f'fold_{f}/seed_17/separated_dual/predictions_reviewed.npz')
        coupled=np.load(OUT/f'fold_{f}/seed_17/coupled_dual/predictions_reviewed.npz')
        assert np.array_equal(original['indices'],sep['indices']) and np.array_equal(sep['indices'],coupled['indices'])
        for j,i in enumerate(sep['indices']):
            pred[int(i)]={'old_original':original['pred'][j],'coupled_dual/full':coupled['full'][j],'separated_dual/full':sep['full'][j]}
            depth[int(i)]={'old_original':original['depth'][j],'coupled_dual/full':coupled['full_depth'][j],'separated_dual/full':sep['full_depth'][j]}
    scores={key:{arm:metrics(pred[i][arm],data['xy'][i,P:],depth[i][arm])['ADE15'] for arm in DISPLAY} for key,i in lookup.items()}
    fixed=read(OLD/'presentation/selection.json')['fixed'];selected=[];why={}
    for reference in ['old_original','coupled_dual/full']:
        for description,func in [('improvement',min),('degradation',max)]:
            key=func(scores,key=lambda k:scores[k]['separated_dual/full']-scores[k][reference])
            why[reference+'_'+description]=key
            if key not in selected:selected.append(key)
    write(PRES/'selection.json',dict(display_seed=17,fixed=fixed,selected=selected,selection_reason=why,scores=scores,
          fixed_rule='Same nine predeclared V1 cases; no replacement based on performance.',
          selected_rule='Post-score gain/loss extrema for displayed seed 17, not representative aggregate scores.'))
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
        assert not dest.exists(),'Preserve existing visualization'
        log=dest.with_suffix('.ffmpeg.log').open('wb')
        proc=subprocess.Popen([FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1920x620','-r','10','-i','pipe:0','-an',
                               '-c:v','libx264','-threads','3','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE,stderr=log)
        for key in keys:
            i=lookup[key];m=meta[i]
            for j,frame in enumerate(range(m['cutoff']-5,m['cutoff']+16)):
                im=frames[m['clip']][frame].copy();r=masks[m['clip']].get((m['id'],frame))
                if r:
                    mm=mask(r['rle']);im[mm]=im[mm]*.83+.17*np.array([255,220,80])
                canvas=np.zeros((620,1920,3),np.uint8)
                kind='固定规则案例' if name=='fixed_9_examples' else '事后选取的改善／退化案例'
                texts=[(12,8,f'{key} | {kind} | 旋转分离版本 · 种子17 · 半速播放',25)]
                for col,arm in enumerate(DISPLAY):
                    panel=im.copy();truth=data['xy'][i,j];point(panel,truth,(255,255,255),6)
                    if j>=6:
                        xy=pred[i][arm][j-6];point(panel,xy,(0,170,255),9)
                        if np.isfinite(xy).all():
                            a=np.clip(truth*.25,-5000,5000).astype(int);b=np.clip(xy*.25,-5000,5000).astype(int)
                            cv2.line(panel,tuple(a),tuple(b),(0,170,255),2,cv2.LINE_AA)
                    x=col*640;canvas[88:568,x:x+640]=panel
                    texts.append((x+9,53,f'{CAPTIONS[col]} | ADE {scores[key][arm]:.1f}px',21))
                stage='读取过去 0.5 秒' if j<6 else f'固定预测：未来 {(j-5)/10:.1f} 秒'
                texts.append((12,586,stage+' | 白圈：SAM参考中心；橙圈：预测中心。预测起点后不再接收新观测。',22))
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
    write(PRES/'verification.json',dict(passed=True,videos=checks));print(checks);print('Selected',selected)


if __name__=='__main__':main()
