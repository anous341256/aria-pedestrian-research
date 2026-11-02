"""Complete fixed-window examples and explicitly selected failure diagnostics."""
from common import *
import subprocess,collections
from PIL import Image,ImageDraw,ImageFont
PRES=OUT/'presentation';FONT='C:/Windows/Fonts/msyh.ttc';FFMPEG=Path('E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe')
NAMES={'raw_future':'原始轨迹 · 只预测未来','raw_dual':'原始轨迹 · 历史重建＋未来预测','comp_future':'旋转补偿 · 只预测未来','comp_dual':'旋转补偿 · 历史重建＋未来预测'}

def text(im,lines,size=24):
    p=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(p);f=ImageFont.truetype(FONT,size)
    for x,y,s in lines:d.text((x,y),s,font=f,fill='white')
    return cv2.cvtColor(np.array(p),cv2.COLOR_RGB2BGR)
def encoder(path,w,h):
    assert not path.exists();log=path.with_suffix('.ffmpeg.log').open('wb');p=subprocess.Popen([str(FFMPEG),'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=log);return p,log
def decode(rle):
    m=np.zeros((1920,2560),np.uint8);x1,y1,x2,y2=map(int,rle[-4:]);flat=np.repeat(np.arange(len(rle[:-4]))%2,np.array(rle[:-4],np.int64)).astype(np.uint8);m[y1:y2+1,x1:x2+1]=flat.reshape(y2-y1+1,x2-x1+1);return m
def point(im,p,color,radius=8):
    p=np.asarray(p)*.375
    if not np.isfinite(p).all():return
    if 0<=p[0]<960 and 0<=p[1]<720:cv2.circle(im,tuple(p.astype(int)),radius,color,3,cv2.LINE_AA)
    else:
        q=np.clip(p,[12,12],[947,707]).astype(int);cv2.drawMarker(im,tuple(q),color,cv2.MARKER_TILTED_CROSS,16,3)
def close(p,log):p.stdin.close();assert p.wait(timeout=60)==0;log.close()
def main():
    PRES.mkdir(parents=True,exist_ok=True);cfg=read(OUT/'protocol.json');meta=read(OUT/'samples.json');data=dict(np.load(OUT/'dataset.npz'));lookup={m['key']:i for i,m in enumerate(meta)};pred={};folds=read(OUT/'folds.json')
    for fold in folds:
        for name in cfg['variants']:
            d=np.load(OUT/f'fold_{fold["fold"]}'/'seed_17'/name/'predictions_reviewed.npz')
            for j,i in enumerate(d['indices']):pred[int(i),name]={k:d[k][j] for k in ['pred','oracle','depth','component']}
    fixed=[]
    for cid in sorted({m['clip'] for m in meta}):
        ids=sorted({m['id'] for m in meta if m['clip']==cid});rr=[m for m in meta if m['clip']==cid and m['id']==ids[0]];fixed.append(sorted(rr,key=lambda m:m['cutoff'])[len(rr)//2]['key'])
    metric=read(OUT/'metrics.json');scores={}
    for m in meta:
        scores[m['key']]={n:float(np.mean([r['metrics']['ADE15'] for r in metric if r['key']==m['key'] and r['mode']=='reviewed' and r['method']==n])) for n in ['raw_future','raw_dual','comp_dual','comp_dual_reference_camera']}
    gain=min(scores,key=lambda k:scores[k]['comp_dual']-scores[k]['raw_future']);loss=max(scores,key=lambda k:scores[k]['comp_dual']-scores[k]['raw_future']);recon=max(scores,key=lambda k:scores[k]['raw_dual']-scores[k]['raw_future']);camera=max(scores,key=lambda k:scores[k]['comp_dual']-scores[k]['comp_dual_reference_camera'])
    special=list(dict.fromkeys([gain,loss,recon,camera]));selection=dict(fixed=fixed,selected_gain=gain,selected_loss=loss,selected_reconstruction_failure=recon,selected_camera_failure=camera,display_seed=17,fixed_rule='First eligible identity in each clip, median cutoff, independent of scores',special_rule='Post-score largest mean-over-seeds gain/loss, reconstruction loss, camera-reference gap; not representative averages')
    write(PRES/'selection.json',selection);frames={};masks={}
    for cid in sorted({m['clip'] for m in meta}):
        cap=cv2.VideoCapture(str(VIDEOS/cid/'input/rgb.mp4'));ff=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            ff.append(cv2.resize(im,(960,720)))
        cap.release();frames[cid]=ff;masks[cid]={(r['id'],r['frame']):r for r in gz(LABELS/cid/'final/masks_rle.jsonl.gz')}
    def visual(m,i,f):
        im=frames[m['clip']][f].copy();row=masks[m['clip']].get((m['id'],f))
        if row:
            mm=cv2.resize(decode(row['rle']),(960,720),interpolation=cv2.INTER_NEAREST).astype(bool);im[mm]=.84*im[mm]+.16*np.array([255,220,80])
        return im
    for label,keys in [('fixed_9_examples',fixed),('selected_success_and_failures',special)]:
        dest=PRES/f'{label}.mp4'
        if dest.exists():continue
        enc,log=encoder(dest,1920,1640)
        for key in keys:
            i=lookup[key];m=meta[i];start=m['cutoff']-5
            # Six history frames and fifteen future frames, played at half speed.
            for j,f in enumerate(range(start,m['cutoff']+16)):
                canvas=np.zeros((1640,1920,3),np.uint8);lines=[(20,10,f'{m["clip"]} / {m["id"]}  预测起点 {m["cutoff"]/10:.1f}s  | 留出录制测试 · 固定种子17 · 半速播放')]
                for n,name in enumerate(cfg['variants']):
                    x=n%2*960;y=60+n//2*770;im=visual(m,i,f);truth=data['xy'][i,j]
                    point(im,truth,(255,255,255),7)
                    if j>=6:
                        p=pred[i,name]['pred'][j-6];point(im,p,(0,170,255),11)
                        if np.isfinite(p).all():
                            a=np.clip((truth*.375),[-5000,-5000],[5000,5000]).astype(int);b=np.clip(p*.375,[-5000,-5000],[5000,5000]).astype(int);cv2.line(im,tuple(a),tuple(b),(0,170,255),2,cv2.LINE_AA)
                    canvas[y+36:y+756,x:x+960]=im;err=metrics(pred[i,name]['pred'],data['xy'][i,P:],pred[i,name]['depth'])['ADE15'];lines.append((x+12,y+2,f'{NAMES[name]}  |  该窗口 ADE {err:.1f}px'))
                stage='正在读取历史' if j<6 else f'冻结预测：未来 {(j-5)/10:.1f} 秒'
                lines.append((20,1605,f'{stage}  | 白圈：SAM参考中心；橙圈：预测中心。预测起点后不使用新观测纠正。'))
                canvas=text(canvas,lines,24);enc.stdin.write(canvas.tobytes());enc.stdin.write(canvas.tobytes())
                if j==20:cv2.imwrite(str(PRES/f'{key}_comparison.jpg'),canvas)
        close(enc,log)
    dest=PRES/'camera_failure_diagnostic.mp4'
    if not dest.exists():
        enc,log=encoder(dest,1920,840);keys=list(dict.fromkeys([loss,camera]))
        for key in keys:
            i=lookup[key];m=meta[i]
            for j,f in enumerate(range(m['cutoff']-5,m['cutoff']+16)):
                canvas=np.zeros((840,1920,3),np.uint8)
                for n,kind in enumerate(['pred','oracle']):
                    im=visual(m,i,f);point(im,data['xy'][i,j],(255,255,255),7)
                    if j>=6:point(im,pred[i,'comp_dual'][kind][j-6],(0,170,255),11)
                    canvas[65:785,n*960:(n+1)*960]=im
                canvas=text(canvas,[(12,12,f'{key} | 预测未来相机'),(975,12,'相同人物预测＋参考未来旋转（诊断）'),(12,803,'只替换重投影旋转；白圈为参考位置，橙圈为预测。右侧使用未来信息。')],25)
                enc.stdin.write(canvas.tobytes());enc.stdin.write(canvas.tobytes())
                if j==20:cv2.imwrite(str(PRES/f'{key}_camera.jpg'),canvas)
        close(enc,log)
    expected={'fixed_9_examples':len(fixed)*42,'selected_success_and_failures':len(special)*42,'camera_failure_diagnostic':len(set([loss,camera]))*42};checks=[]
    for file in PRES.glob('*.mp4'):
        cap=cv2.VideoCapture(str(file));reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            n+=1
        cap.release();assert n==reported==expected[file.stem];checks.append(dict(file=file.name,frames=n,seconds=n/10,passed=True))
    write(PRES/'verification.json',dict(passed=True,videos=checks));print(selection);print(checks)
if __name__=='__main__':main()
