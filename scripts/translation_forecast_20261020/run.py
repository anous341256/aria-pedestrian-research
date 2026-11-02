"""Matched fits with/without historical translation; validation-only selection."""
import time, copy, hashlib, random
from pathlib import Path
import numpy as np
from prepare import ROOT,OUT,PREV,OLD,CAM,ARMS,read,write,sha,rm,rv,warp
from network import Camera,Person,torch

torch.set_num_threads(2)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SCALE=np.array([2560.,1920.])
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def codehash():return {n:sha(Path(__file__).with_name(n)) for n in ["prepare.py","network.py","run.py"]}
def statehash(model):return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
def balanced(indices,meta,rng):
    groups=sorted({meta[i]["recording"] for i in indices})
    return np.array([rng.choice([i for i in indices if meta[i]["recording"]==g]) for g in rng.choice(groups,48)])
def translation(data,idx,arm):
    a=data["translation"][idx,:16]
    return a if arm=="with_translation" else np.zeros_like(a)
def camera_inputs(data,idx,arm,key="cam"):
    return np.concatenate([data[key][idx,:16],translation(data,idx,arm)],axis=-1)
def features(data,idx,arm,train=False,rng=None):
    obs=data["xy"][idx,:16].copy();valid=np.ones(obs.shape[:2])
    if train:
        choice=rng.random(len(idx));old=(choice>=.4)&(choice<.6);noisy=choice>=.6
        obs[old]=data["raw"][idx,:16][old];valid[old]=data["raw_valid"][idx,:16][old]
        jitter=rng.normal(0,.005,obs.shape)*SCALE;obs[noisy]+=jitter[noisy]
        keep=(rng.random(valid.shape)>=.2).astype(float);keep[:,-1]=1;valid[noisy]=keep[noisy]
    origin=obs[:,-1]/SCALE
    stable=warp(obs,data["H"][idx,:16])[0]/SCALE
    rel=stable-origin[:,None];delta=np.zeros_like(rel)
    delta[:,1:]=(stable[:,1:]-stable[:,:-1])*(valid[:,1:]*valid[:,:-1])[...,None];rel[valid==0]=0
    seq=np.concatenate([rel,delta,valid[...,None],data["seconds"][idx,:16,None]/1.5,translation(data,idx,arm)],-1)
    return seq.astype(np.float32),origin.astype(np.float32)
def noise(meta,idx):
    return np.stack([np.random.default_rng(100000+int(hashlib.sha256((meta[i]["key"]+"future").encode()).hexdigest()[:8],16)).normal(0,.15,(5,10)) for i in idx]).astype(np.float32)
@torch.no_grad()
def infer_camera(model,x):
    return np.concatenate([model(tensor(y)).cpu().numpy() for y in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))])
@torch.no_grad()
def predict(person,camera,data,meta,idx,arm):
    seq,origin=features(data,idx,arm)
    pc=person.predict(tensor(seq),tensor(origin),tensor(noise(meta,idx))).cpu().numpy()
    return {"component":(pc+origin[:,None])*SCALE,"rotation":infer_camera(camera,camera_inputs(data,idx,arm))}
def project(x,r,K):
    H=np.einsum("nij,ntkj,nkl->ntil",K,rm(r),np.linalg.inv(K))
    return warp(x,H)
def diagnostics(z,data,idx):
    z=dict(z);target=data["xy"][idx,16:21];truth=warp(target,data["H"][idx,16:21])[0]
    z["residual_error"]=np.linalg.norm(z["component"]-truth,axis=-1)
    z["angle_deg"]=np.rad2deg(np.linalg.norm(rv(np.einsum("ntji,ntjk->ntik",rm(z["rotation"]),rm(data["cam"][idx,16:21]))),axis=-1))
    for name,x,r in [("image",z["component"],z["rotation"]),("gt_camera",z["component"],data["cam"][idx,16:21]),("gt_person",truth,z["rotation"]),("frozen_camera",z["component"],np.zeros_like(z["rotation"]))]:
        p,depth=project(x,r,data["K"][idx]);e=np.linalg.norm(p-target,axis=-1)
        e[(depth<=1e-6)|~np.isfinite(e)]=32000.
        z[name]=p;z[name+"_depth"]=depth;z[name+"_error"]=e
    return z
def camera_fit(cdata,fold,seed,arm):
    dest=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}/camera';dest.mkdir(parents=True,exist_ok=True)
    seed_all(seed);model=Camera().to(DEVICE)
    if (dest/"complete.json").exists():
        assert read(dest/"complete.json")["code_hashes"]==codehash()
        model.load_state_dict(torch.load(dest/"checkpoint.pt",map_location=DEVICE,weights_only=False)["model"])
        return model.eval()
    begin=time.monotonic();init=statehash(model)
    idx=np.asarray(fold["train_indices"]);vi=np.asarray(fold["val_indices"])
    x=tensor(camera_inputs(cdata,idx,arm,"rotation"));y=tensor(cdata["rotation"][idx,16:21])
    vx=camera_inputs(cdata,vi,arm,"rotation");vy=cdata["rotation"][vi,16:21]
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed);left=np.array([],int);logs=[];best=float("inf");drawhash=hashlib.sha256()
    for step in range(1,2001):
        while len(left)<256:left=np.concatenate([left,rng.permutation(len(x))])
        ii,left=left[:256],left[256:];drawhash.update(ii.tobytes());ii=torch.as_tensor(ii,device=DEVICE)
        model.train();loss=(model(x[ii])-y[ii]).square().mean();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:
            model.eval();v=float(np.mean((infer_camera(model,vx)-vy)**2))
            logs.append(dict(step=step,validation_mse=v))
            if v<best:best=v;beststep=step;state=copy.deepcopy(model.state_dict())
            write(OUT/"progress.json",dict(arm=arm,fold=fold["fold"],seed=seed,stage="camera",step=step,total=2000))
    model.load_state_dict(state)
    torch.save(dict(model=state,step=beststep,seed=seed,arm=arm,fold=fold,protocol_sha256=sha(OUT/"protocol.json")),dest/"checkpoint.pt")
    write(dest/"training.json",logs)
    write(dest/"complete.json",dict(seconds=time.monotonic()-begin,step=beststep,initial_hash=init,draws_hash=drawhash.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),code_hashes=codehash()))
    print(arm,fold["fold"],seed,"camera",round(time.monotonic()-begin,1),flush=True)
    return model.eval()
def person_fit(data,meta,fold,seed,arm,camera):
    dest=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}/person';dest.mkdir(parents=True,exist_ok=True)
    seed_all(seed);model=Person().to(DEVICE)
    if (dest/"complete.json").exists():
        assert read(dest/"complete.json")["code_hashes"]==codehash()
        return
    begin=time.monotonic();init=statehash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    logs=[];best=float("inf");drawhash=hashlib.sha256();aughash=hashlib.sha256();stable=warp(data["xy"],data["H"])[0]/SCALE
    for step in range(240):
        epoch,update=divmod(step,2)
        rng=np.random.default_rng(seed*100000+epoch*10+update)
        idx=balanced(fold["train_indices"],meta,rng);drawhash.update(idx.tobytes())
        seq,origin=features(data,idx,arm,True,rng);y=stable[idx]-origin[:,None]
        t1=tensor(rng.random(48));t2=tensor(rng.random(48))
        n1=tensor(rng.normal(0,.15,(48,32)));n2=tensor(rng.normal(0,.15,(48,10)))
        for val in [seq[:,:,:6],origin,t1.cpu().numpy(),t2.cpu().numpy(),n1.cpu().numpy(),n2.cpu().numpy()]:
            aughash.update(np.ascontiguousarray(val).tobytes())
        model.train()
        loss,lf,lh=model.loss(tensor(seq),tensor(origin),tensor(y[:,:16].reshape(48,-1)),tensor(y[:,16:21].reshape(48,-1)),t1,t2,n1,n2)
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (step+1)%20==0:
            model.eval();idx=np.asarray(fold["val_indices"])
            z=predict(model,camera,data,meta,idx,arm)
            v=float(np.linalg.norm(z["component"]-stable[idx,16:21]*SCALE,axis=-1).mean())
            logs.append(dict(step=step+1,validation_residual_ADE=v))
            if v<best:best=v;beststep=step+1;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    torch.save(dict(model=state,step=beststep,seed=seed,arm=arm,fold=fold,protocol_sha256=sha(OUT/"protocol.json")),dest/"checkpoint.pt")
    write(dest/"training.json",logs)
    idx=np.asarray(fold["test_indices"]);z=diagnostics(predict(model,camera,data,meta,idx,arm),data,idx)
    np.savez_compressed(dest/"predictions.npz",indices=idx,**z)
    write(dest/"complete.json",dict(seconds=time.monotonic()-begin,step=beststep,initial_hash=init,draws_hash=drawhash.hexdigest(),augmentation_hash=aughash.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),code_hashes=codehash()))
    print(arm,fold["fold"],seed,"person",round(time.monotonic()-begin,1),flush=True)
def summary(rows):
    keys=[k for k in ["ADE","FDE","residual_ADE","residual_FDE","camera_mean_deg","camera_final_deg","gt_camera_ADE","gt_person_ADE"] if k in rows[0]]
    def avg(rr):return {k:float(np.mean([r[k] for r in rr])) for k in keys}
    rec={g:avg([r for r in rows if r["recording"]==g]) for g in sorted({r["recording"] for r in rows})}
    return dict(window_mean=avg(rows),recording_equal={k:float(np.mean([v[k] for v in rec.values()])) for k in keys},
        per_recording=rec,per_seed={str(s):avg([r for r in rows if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})},
        sparse=avg([r for r in rows if r["nonoverlap"]]))
def row(m,i,seed,arm,z,j):
    r=dict(arm=arm,index=int(i),key=m["key"],recording=m["recording"],seed=seed,nonoverlap=m["nonoverlap"],
        ADE=float(z["image_error"][j].mean()),FDE=float(z["image_error"][j,-1]))
    if "residual_error" in z:
        r.update(residual_ADE=float(z["residual_error"][j].mean()),residual_FDE=float(z["residual_error"][j,-1]),
            camera_mean_deg=float(z["angle_deg"][j].mean()),camera_final_deg=float(z["angle_deg"][j,-1]),
            gt_camera_ADE=float(z["gt_camera_error"][j].mean()),gt_person_ADE=float(z["gt_person_error"][j].mean()))
    return r
def evaluate(data,meta,folds,cfg):
    rows=[];maxshared=0.;seconds=0.;invalid=0;paired=0
    temporal=["xy","raw","raw_valid","R","H","cam","seconds","boxes"]
    short={k:v[:,:16].copy() if k in temporal else v.copy() for k,v in data.items()}
    changed={k:v.copy() for k,v in data.items()};changed["translation"]+=123.45
    for fold in folds:
        idx=np.asarray(fold["test_indices"])
        for seed in cfg["seeds"]:
            outputs={}
            for arm in ARMS:
                folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}'
                cm=Camera().to(DEVICE);pm=Person().to(DEVICE)
                cm.load_state_dict(torch.load(folder/"camera/checkpoint.pt",map_location=DEVICE,weights_only=False)["model"])
                pm.load_state_dict(torch.load(folder/"person/checkpoint.pt",map_location=DEVICE,weights_only=False)["model"])
                cm.eval();pm.eval()
                stored=dict(np.load(folder/"person/predictions.npz"))
                z=predict(pm,cm,data,meta,idx,arm);trunc=predict(pm,cm,short,meta,idx,arm)
                for k in ["component","rotation"]:
                    assert np.array_equal(z[k],stored[k])
                    assert np.array_equal(z[k],trunc[k])
                if arm=="no_translation":
                    ignored=predict(pm,cm,changed,meta,idx,arm)
                    assert all(np.array_equal(z[k],ignored[k]) for k in z)
                groups={}
                for j,i in enumerate(idx):groups.setdefault((meta[i]["recording"],meta[i]["timestamp_ns"]),[]).append(j)
                for jj in groups.values():maxshared=max(maxshared,float(np.max(np.abs(z["rotation"][jj]-z["rotation"][jj[0]]))))
                d=diagnostics(z,data,idx);outputs[arm]=d
                assert all(np.array_equal(d[k],stored[k]) for k in d)
                invalid+=int((d["image_depth"]<=1e-6).sum())
                for part in ["camera","person"]:
                    seconds+=read(folder/f"{part}/complete.json")["seconds"]
                    if arm==ARMS[1]:
                        a=read(OUT/ARMS[0]/f'fold_{fold["fold"]}/seed_{seed}/{part}/complete.json')
                        b=read(folder/f"{part}/complete.json")
                        assert a["draws_hash"]==b["draws_hash"] and a["initial_hash"]==b["initial_hash"]
                        assert a["parameters"]==b["parameters"]
                        if part=="person":assert a["augmentation_hash"]==b["augmentation_hash"]
                        paired+=1
                rows += [row(meta[i],i,seed,arm,d,j) for j,i in enumerate(idx)]
            for name,pa,ca in [("new_person_control_camera",ARMS[1],ARMS[0]),("control_person_new_camera",ARMS[0],ARMS[1])]:
                swapped=diagnostics(dict(component=outputs[pa]["component"],rotation=outputs[ca]["rotation"]),data,idx)
                rows += [row(meta[i],i,seed,name,swapped,j) for j,i in enumerate(idx)]
            old=np.load(PREV/f'long_history/fold_{fold["fold"]}/seed_{seed}/person/predictions.npz')
            assert np.array_equal(old["indices"],idx)
            rows += [row(meta[i],i,seed,"previous_saved",old,j) for j,i in enumerate(idx)]
    for i,m in enumerate(meta):
        e=np.linalg.norm(data["xy"][i,15]-data["xy"][i,16:21],axis=-1)
        rows.append(row(m,i,0,"hold_last",{"image_error":e[None]},0))
    result={a:summary([r for r in rows if r["arm"]==a]) for a in sorted({r["arm"] for r in rows})}
    contrast={}
    for ref in ["no_translation","previous_saved","hold_last"]:
        a=np.array([np.mean([r["ADE"] for r in rows if r["arm"]=="with_translation" and r["index"]==i]) for i in range(len(meta))])
        b=np.array([np.mean([r["ADE"] for r in rows if r["arm"]==ref and r["index"]==i]) for i in range(len(meta))])
        diff=a-b
        rec=np.array([diff[[i for i,m in enumerate(meta) if m["recording"]==g]].mean() for g in sorted({m["recording"] for m in meta})])
        bootstrap=np.random.default_rng(20261020).choice(rec,(5000,len(rec)),replace=True).mean(1)
        contrast[ref]=dict(ADE_difference=float(diff.mean()),relative_ADE_percent=float(100*(a.mean()/b.mean()-1)),
            improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),
            recording_bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist())
    assert maxshared==0
    protected=read(OUT/"source_hashes.json")
    assert all(sha(ROOT/k)==v for k,v in protected.items())
    write(OUT/"metrics.json",rows)
    write(OUT/"comparison.json",dict(results=result,contrasts=contrast,training_seconds=seconds))
    write(OUT/"verification.json",dict(passed=True,checkpoint_replays=72,exact_forecast_replay=True,physically_future_truncated_replay=True,
        no_translation_ignores_changed_input=True,paired_camera_person_fits=paired,same_time_camera_max=maxshared,
        invalid_prediction_rays=invalid,protected_files=len(protected),code_hashes=codehash()))
    print("RESULTS",{k:v["window_mean"] for k,v in result.items()},flush=True)
def smoke(data,meta):
    seed_all(17);cm=Camera().to(DEVICE);pm=Person().to(DEVICE);idx=np.arange(4)
    a=predict(pm,cm,data,meta,idx,ARMS[0]);b=predict(pm,cm,data,meta,idx,ARMS[1])
    assert all(np.array_equal(a[k],b[k]) for k in a)
    x=tensor(camera_inputs(data,idx,ARMS[1]));cm(x).square().mean().backward()
    seq,o=features(data,idx,ARMS[1]);pm.encode(tensor(seq),tensor(o)).square().sum().backward()
    cg=float(cm.encoder.weight_ih_l0.grad[:,3:].abs().max())
    pg=float(pm.encoder.weight_ih_l0.grad[:,6:].abs().max())
    assert cg>0 and pg>0
    write(OUT/"initialization_check.json",dict(passed=True,initial_predictions_identical=True,translation_gradients_camera=cg,translation_gradients_person=pg,
        camera_parameters=sum(p.numel() for p in cm.parameters()),person_parameters=sum(p.numel() for p in pm.parameters())))
def main():
    cfg=read(OUT/"protocol.json");data=dict(np.load(OUT/"dataset.npz"));meta=read(OUT/"samples.json")
    cdata=dict(np.load(OUT/"camera_dataset.npz"));folds=read(OUT/"folds.json");cfolds=read(OUT/"camera_folds.json")
    assert len(data["xy"])==117 and len(cdata["rotation"])==27185
    write(OUT/"execution_code_hashes.json",codehash());smoke(data,meta)
    print("DEVICE",DEVICE,flush=True)
    for fold,cf in zip(folds,cfolds):
        assert fold["test"]==cf["test"] and fold["validation"]==cf["validation"]
        for seed in cfg["seeds"]:
            for arm in ARMS:
                cm=camera_fit(cdata,cf,seed,arm)
                person_fit(data,meta,fold,seed,arm,cm)
    evaluate(data,meta,folds,cfg)
    write(OUT/"training_complete.json",dict(passed=True,camera_fits=36,person_fits=36))

if __name__=="__main__":main()
