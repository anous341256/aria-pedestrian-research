"""Frozen historical-only RGB features. No forecasting labels enter the encoder."""
from common import *
import cv2,torch,time
import torch.nn.functional as F
torch.set_num_threads(2)
HUB=Path("/root/.cache/torch/hub/facebookresearch_dinov2_main")
WEIGHT=Path("/root/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
def cropbox(box):
    center=(box[:2]+box[2:])/2;size=(box[2:]-box[:2])*1.5
    a=np.floor(center-size/2).astype(int);b=np.ceil(center+size/2).astype(int)
    a=np.maximum(a,0);b=np.minimum(b,[2560,1920])
    assert (b>a).all()
    return np.r_[a,b]
def positions(box):
    yy,xx=np.meshgrid((np.arange(4)+.5)/4,(np.arange(4)+.5)/4,indexing="ij")
    uv=np.stack([xx.ravel(),yy.ravel()],-1)
    return (box[:2]+uv*(box[2:]-box[:2]))/2560.
@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=dict(version=OUT.name,arms=ARMS,seeds=[17,29,43],history_points=16,future_points=5,
      cohort="Exact 117 windows and six recording-held-out folds of translation_forecast_20261020.",
      encoder="Frozen DINOv2 ViT-S/14; FP32 extraction; RGB ImageNet normalization; no fine-tuning.",
      visual=dict(scene_size=[448,336],local_size=[224,224],crop_expansion=1.5,pooled_grid=[4,4],patch_dim=384),
      model="64-dimensional numeric streams, two temporal self blocks, same-time visual cross-attention with zero residual map, two fused temporal blocks, deterministic five-point image displacement head.",
      inputs="Raw historical centers/width/height/deltas, MPS historical relative rotation/translation; historical-only RGB and boxes. No rotation compensation or future geometry.",
      training=dict(updates=240,batch=48,lr=.001,weight_decay=.01,clip=1.,validation_every=20,step0_eligible=True,
        loss="MSE of five future image displacement vectors, both axes divided by2560.",
        augmentation="None in all four arms, preserving frozen crop/trajectory spatial alignment.",
        sampling="Recording-balanced, exactly paired batch indices and initial weights across arms."),
      control="Same registered structure; disabled visual tokens zeroed and masked, constant null token remains. All arms receive box size/position and identical camera inputs.",
      normalization="Image coordinates divided by2560 on BOTH axes. Time /1.5; rotations radians; translations meters; no fitted global statistics.",
      evaluation="ADE/FDE, seeds, recording-equal, sparse subset, fixed old nine cases. Only new four arms isolate visual information.",
      reference="Previous separated with-translation model is a system reference; architectures, losses, extra camera-only pretraining and augmentation differ.",
      limitations="Reviewed offline SAM/MPS; frozen images still contain coupled motion. No independent future camera/residual outputs. No online-system claim.")
    if (OUT/"protocol.json").exists():assert read(OUT/"protocol.json")==cfg
    else:write(OUT/"protocol.json",cfg)
    if (OUT/"extraction_complete.json").exists():return
    start=time.monotonic()
    d=dict(np.load(SOURCE/"dataset.npz"));meta=read(SOURCE/"samples.json");folds=read(SOURCE/"folds.json")
    protected={str(p.relative_to(ROOT)):sha(p) for p in SOURCE.rglob("*") if p.is_file() and p.suffix in [".json",".npz",".pt"]}
    for p in (ROOT/"scripts/translation_forecast_20261020").glob("*.py"):protected[str(p.relative_to(ROOT))]=sha(p)
    for m in meta:
        p=source_video(m)
        if str(p.relative_to(ROOT)) not in protected:protected[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/"source_hashes.json",protected)
    for name in ["samples.json","folds.json"]:write(OUT/name,read(SOURCE/name))
    for fold in folds:
        g=[{meta[i]["recording"] for i in fold[k]} for k in ["train_indices","val_indices","test_indices"]]
        assert not(g[0]&g[1] or g[0]&g[2] or g[1]&g[2])
    model=torch.hub.load(str(HUB),"dinov2_vits14",source="local",pretrained=True).eval().cuda()
    for p in model.parameters():p.requires_grad_(False)
    write(OUT/"encoder_manifest.json",dict(weight_sha256=sha(WEIGHT),weight_path=str(WEIGHT),
       code_hashes={str(p.relative_to(HUB)):sha(p) for p in HUB.rglob("*.py")},
       parameters=sum(p.numel() for p in model.parameters())))
    mean=torch.tensor([.485,.456,.406],device="cuda")[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device="cuda")[None,:,None,None]
    def encode(im,size):
        rgb=cv2.cvtColor(cv2.resize(im,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
        x=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda().float()/255
        z=model.forward_features((x-mean)/std)["x_norm_patchtokens"]
        z=z.transpose(1,2).reshape(1,384,size[1]//14,size[0]//14)
        return F.adaptive_avg_pool2d(z,(4,4)).flatten(2).transpose(1,2)[0].cpu().numpy().astype(np.float16)
    scene=np.empty((len(meta),16,16,384),np.float16);local=np.empty_like(scene)
    pos=np.empty((len(meta),16,32,2),np.float32);crops=np.empty((len(meta),16,4),np.int32)
    lookup={};audit=[];scene_count=0;local_count=0;checks=[]
    for i,m in enumerate(meta):
        for t,frame in enumerate(m["frames"][:16]):lookup.setdefault((m["clip"],int(frame)),[]).append((i,t))
    refs={}
    for n,((clip,frame),entries) in enumerate(sorted(lookup.items())):
        m=meta[entries[0][0]]
        if clip not in refs:
            p=source_video(m).parent/"reference.npz";refs[clip]=dict(np.load(p));protected[str(p.relative_to(ROOT))]=sha(p)
        for i,t in entries:
            assert int(refs[clip]["times_ns"][frame])==meta[i]["timestamps_ns"][t]
            assert meta[i]["timestamps_ns"][t]<=meta[i]["timestamp_ns"]
        cached=OUT/f"feature_cache/{clip}/{frame:05d}.npz"
        if cached.exists():
            z=dict(np.load(cached));sg=z["scene"];lf=z["local"];cb=z["crops"]
            expected=np.stack([cropbox(d["boxes"][i,t]) for i,t in entries])
            assert np.array_equal(cb,expected) and np.array_equal(z["entries"],entries)
        else:
            cap=cv2.VideoCapture(str(source_video(m)));assert cap.isOpened();cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,im=cap.read();cap.release();assert ok and im.shape==(1920,2560,3)
            sg=encode(im,(448,336));lf=[];cb=[]
            for i,t in entries:
                box=cropbox(d["boxes"][i,t]);a,b,c,e=box
                lf.append(encode(im[b:e,a:c],(224,224)));cb.append(box)
            lf=np.stack(lf);cb=np.stack(cb)
            cached.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cached,scene=sg,local=lf,crops=cb,entries=np.array(entries))
            if len(checks)<8:
                view=cv2.resize(im,(640,480));a,b,c,e=(cb[0]/4).astype(int)
                cv2.rectangle(view,(a,b),(c,e),(0,200,255),2)
                cv2.putText(view,f"{meta[entries[0][0]]['key']} | history {entries[0][1]}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                checks.append(view)
        for j,(i,t) in enumerate(entries):
            scene[i,t]=sg;local[i,t]=lf[j];crops[i,t]=cb[j]
            pos[i,t,:16]=positions(cb[j]);pos[i,t,16:]=positions(np.array([0,0,2560,1920]))
            audit.append(dict(index=i,time_index=t,clip=clip,frame=frame,timestamp_ns=meta[i]["timestamps_ns"][t],crop=cb[j].tolist()))
        scene_count+=1;local_count+=len(entries)
        if n%20==0:
            write(OUT/"progress.json",dict(stage="RGB feature extraction",frames=n+1,total=len(lookup)))
            print("features",n+1,"/",len(lookup),flush=True)
    assert np.isfinite(scene).all() and np.isfinite(local).all()
    np.savez_compressed(OUT/"features.npz",scene=scene,local=local,positions=pos,crops=crops,
      timestamps_ns=np.array([m["timestamps_ns"][:16] for m in meta],dtype=np.int64))
    if checks:
        while len(checks)<8:checks.append(np.zeros_like(checks[0]))
        cv2.imwrite(str(OUT/"crop_alignment.jpg"),np.vstack([np.hstack(checks[j:j+2]) for j in range(0,8,2)]))
    write(OUT/"source_hashes.json",protected);write(OUT/"frame_manifest.json",audit)
    write(OUT/"extraction_complete.json",dict(passed=True,unique_scene_frames=scene_count,person_frames=local_count,
        historical_timestamps_checked=True,seconds=time.monotonic()-start,features_sha256=sha(OUT/"features.npz")))
    print(read(OUT/"extraction_complete.json"),flush=True)
if __name__=="__main__":main()
