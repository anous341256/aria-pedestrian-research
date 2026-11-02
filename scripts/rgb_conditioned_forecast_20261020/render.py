from common import *
import cv2,subprocess
def main():
    pres=OUT/"presentation";pres.mkdir(parents=True,exist_ok=True)
    dest=pres/"fixed_9_rgb_comparison.mp4"
    if (pres/"verification.json").exists():return
    if dest.exists():raise RuntimeError("Previous incomplete render exists; preserve before rerun")
    d=np.load(SOURCE/"dataset.npz");meta=read(OUT/"samples.json")
    keys=read(ROOT/"outputs/biflow_aria_v1_20261002/presentation/selection.json")["fixed"]
    lookup={m["key"]:i for i,m in enumerate(meta)};pred={}
    for fold in read(OUT/"folds.json"):
        for arm in ARMS:
            z=np.load(OUT/arm/f'fold_{fold["fold"]}/seed_17/predictions.npz')
            for j,i in enumerate(z["indices"]):pred.setdefault(int(i),{})[arm]=(z["image"][j],z["error"][j].mean())
        z=np.load(SOURCE/f'with_translation/fold_{fold["fold"]}/seed_17/person/predictions.npz')
        for j,i in enumerate(z["indices"]):pred[int(i)]["previous"]=(z["image"][j],z["image_error"][j].mean())
    ffmpeg="/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe"
    args=[ffmpeg,"-v","error","-n","-f","rawvideo","-pix_fmt","bgr24","-s","1920x1160","-r","10","-i","pipe:0","-an","-c:v","libx264","-threads","3","-preset","veryfast","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart","E:/"+str(dest).removeprefix("/mnt/e/")]
    log=(pres/"render.log").open("wb");proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
    def text(im,s,x,y,size=.7):cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(240,240,240),2,cv2.LINE_AA)
    def pt(im,p,color,r):
        q=np.asarray(p)/4
        if np.isfinite(q).all():cv2.circle(im,tuple(np.clip(q,-5000,5000).astype(int)),r,color,2,cv2.LINE_AA)
    count=0
    for key in keys:
        i=lookup[key];m=meta[i];cap=cv2.VideoCapture(str(source_video(m)));cap.set(cv2.CAP_PROP_POS_FRAMES,m["frames"][0])
        truth=d["xy"][i,16:21];hold=np.repeat(d["xy"][i,15:16],5,axis=0)
        pred[i]["hold"]=(hold,np.linalg.norm(hold-truth,axis=-1).mean())
        layout=[("no_rgb","A: No RGB"),("local_rgb","B: Local RGB"),("scene_rgb","C: Scene RGB"),
                ("both_rgb","D: Local + Scene RGB"),("previous","Previous separated + translation"),("hold","Hold last position")]
        for t in range(21):
            ok,im=cap.read();assert ok;im=cv2.resize(im,(640,480))
            canvas=np.zeros((1160,1920,3),np.uint8);text(canvas,f"{key} | 1.5s history -> 0.5s future | Seed17 | Half speed",15,30,.85)
            for j,(arm,label) in enumerate(layout):
                row,col=divmod(j,3);x=col*640;y=60+row*530
                arr,ade=pred[i][arm];panel=im.copy()
                pt(panel,d["xy"][i,t],(255,255,255),6)
                if t<=15:
                    for h in range(t+1):pt(panel,d["xy"][i,h],(255,200,60),3)
                else:
                    pt(panel,arr[t-16],(0,160,255),9)
                    a=np.clip(truth[t-16]/4,-5000,5000).astype(int);b=np.clip(arr[t-16]/4,-5000,5000).astype(int)
                    cv2.line(panel,tuple(a),tuple(b),(0,160,255),2,cv2.LINE_AA)
                text(canvas,f"{label} | ADE {ade:.1f}px",x+8,y+15,.59)
                canvas[y+25:y+505,x:x+640]=panel
            phase=f"History: {(t-15)*.1:+.1f}s" if t<=15 else f"Forecast +{(t-15)*.1:.1f}s, no new observations"
            text(canvas,phase+" | White: SAM reference center; orange: forecast.",15,1140,.75)
            proc.stdin.write(canvas.tobytes());proc.stdin.write(canvas.tobytes());count+=2
            if t==20:cv2.imwrite(str(pres/f"{key}.jpg"),canvas)
        cap.release()
    proc.stdin.close();assert proc.wait(timeout=60)==0;log.close()
    cap=cv2.VideoCapture(str(dest));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1160,1920);decoded+=1
    cap.release();assert decoded==count==378
    write(pres/"verification.json",dict(passed=True,frames=decoded,seconds=37.8,fixed_keys=keys,
        selection="Same old nine cases, seed17; failures preserved, not selected on new results."))
    print(str(dest),flush=True)
if __name__=="__main__":main()
