"""Independent saved-prediction metrics and feature-to-frame/crop association audit."""
from common import *
import cv2
def main():
    assert read(OUT/"verification.json")["passed"]
    d=dict(np.load(SOURCE/"dataset.npz"));v=dict(np.load(OUT/"features.npz"));meta=read(OUT/"samples.json")
    files=0;maxerr=0.;metricrows=read(OUT/"metrics.json")
    for arm in ARMS:
        for fold in read(OUT/"folds.json"):
            for s in [17,29,43]:
                z=np.load(OUT/arm/f'fold_{fold["fold"]}/seed_{s}/predictions.npz')
                idx=z["indices"];a=z["image"][:,:,0]-d["xy"][idx,16:21,0];b=z["image"][:,:,1]-d["xy"][idx,16:21,1]
                err=np.hypot(a,b);maxerr=max(maxerr,float(np.max(abs(err-z["error"]))))
                for j,i in enumerate(idx):
                    r=next(r for r in metricrows if r["arm"]==arm and r["seed"]==s and r["index"]==i)
                    assert abs(err[j].sum()/5-r["ADE"])<1e-9 and abs(err[j,-1]-r["FDE"])<1e-9
                assert np.max(abs(z["image"]-(z["displacement"]*2560+d["xy"][idx,15,None])))==0
                files+=1
    refs={};checked=0
    for cache in sorted((OUT/"feature_cache").rglob("*.npz")):
        z=np.load(cache);clip=cache.parent.name;frame=int(cache.stem)
        if clip not in refs:refs[clip]=np.load(ROOT/f"outputs/sam2_g1_v6_v8_20261001/{clip}/input/reference.npz")
        for j,(ii,tt) in enumerate(z["entries"]):
            i=int(ii);t=int(tt);m=meta[i]
            assert t<16 and m["clip"]==clip and m["frames"][t]==frame
            assert refs[clip]["times_ns"][frame]==v["timestamps_ns"][i,t]==m["timestamps_ns"][t]
            assert v["timestamps_ns"][i,t]<=m["timestamp_ns"]
            box=d["boxes"][i,t].astype(float);c=(box[:2]+box[2:])/2;s=(box[2:]-box[:2])*1.5
            crop=np.r_[np.maximum(np.floor(c-s/2),0),np.minimum(np.ceil(c+s/2),[2560,1920])].astype(int)
            assert np.array_equal(crop,z["crops"][j]) and np.array_equal(crop,v["crops"][i,t])
            assert np.array_equal(z["scene"],v["scene"][i,t]) and np.array_equal(z["local"][j],v["local"][i,t])
            assert np.max(abs(c-d["xy"][i,t]))<1e-6
            checked+=1
    assert checked==117*16
    # Diverse review page: one last historical frame from each clip.
    panels=[];seen=set()
    for i,m in enumerate(meta):
        if m["clip"] in seen:continue
        seen.add(m["clip"]);cap=cv2.VideoCapture(str(source_video(m)));cap.set(cv2.CAP_PROP_POS_FRAMES,m["frames"][15]);ok,im=cap.read();cap.release();assert ok
        im=cv2.resize(im,(640,480));box=(v["crops"][i,15]/4).astype(int);a,b,c,e=box
        cv2.rectangle(im,(a,b),(c,e),(0,180,255),2);xy=(d["xy"][i,15]/4).astype(int)
        cv2.circle(im,tuple(xy),5,(255,255,255),-1)
        cv2.putText(im,m["key"]+" | last historical frame",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        panels.append(im)
    while len(panels)%2:panels.append(np.zeros_like(panels[0]))
    cv2.imwrite(str(OUT/"diverse_crop_audit.jpg"),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))
    write(OUT/"independent_audit.json",dict(passed=True,prediction_files=files,person_frame_associations=checked,unique_scene_frames=len(list((OUT/"feature_cache").rglob("*.npz"))),
        max_metric_discrepancy_px=maxerr,history_box_centers_match=True,all_feature_timestamps_at_or_before_cutoff=True))
    print(read(OUT/"independent_audit.json"),flush=True)
if __name__=="__main__":main()
