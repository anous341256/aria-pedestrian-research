"""Four matched visual-information arms; fixed validation rule, no test tuning."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from common import *
from network import Predictor,torch
import time,copy,random
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
DEVICE="cuda"
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def statehash(model):return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
def hashes():return {n:sha(Path(__file__).with_name(n)) for n in ["common.py","network.py","run.py","extract.py"]}
def load():
    assert read(OUT/"extraction_complete.json")["passed"]
    assert sha(OUT/"features.npz")==read(OUT/"extraction_complete.json")["features_sha256"]
    d=dict(np.load(SOURCE/"dataset.npz"));v=dict(np.load(OUT/"features.npz"))
    p,c=numeric(d)
    inputs=[tensor(a) for a in [p,c,v["local"],v["scene"],v["positions"]]]
    return d,inputs
def forward(model,inputs,idx,arm):
    return model(*[v[idx] for v in inputs],arm)
@torch.no_grad()
def predict(model,inputs,idx,arm):
    # Fixed one-sample inference shape makes physical-suffix and stored replay exact.
    return np.concatenate([forward(model,inputs,np.array([i]),arm).cpu().numpy() for i in idx])
def draws(fold,meta,s,step):
    rng=np.random.default_rng(s*100000+step)
    groups=sorted({meta[i]["recording"] for i in fold["train_indices"]})
    return np.array([rng.choice([i for i in fold["train_indices"] if meta[i]["recording"]==g]) for g in rng.choice(groups,48)])
def fit(d,inputs,meta,fold,s,arm):
    dest=OUT/arm/f'fold_{fold["fold"]}/seed_{s}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/"complete.json").exists():
        assert read(dest/"complete.json")["code_hashes"]==hashes();return
    seed(s);model=Predictor().cuda();initial=statehash(model);begin=time.monotonic()
    target=tensor((d["xy"][:,16:21]-d["xy"][:,15,None])/2560.)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=float("inf");history=[];drawhash=hashlib.sha256();vi=np.array(fold["val_indices"])
    for step in range(241):
        if step:
            idx=draws(fold,meta,s,step);drawhash.update(idx.tobytes())
            model.train();output=forward(model,inputs,idx,arm)
            loss=(output-target[idx]).square().mean();assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%20==0:
            model.eval();pred=predict(model,inputs,vi,arm)*2560+d["xy"][vi,15,None]
            score=float(np.linalg.norm(pred-d["xy"][vi,16:21],axis=-1).mean())
            history.append(dict(step=step,validation_ADE=score))
            if score<best:best=score;beststep=step;state=copy.deepcopy(model.state_dict())
            write(OUT/"progress.json",dict(stage="training",arm=arm,fold=fold["fold"],seed=s,step=step,total=240))
    model.load_state_dict(state);model.eval();idx=np.array(fold["test_indices"])
    disp=predict(model,inputs,idx,arm);image=disp*2560+d["xy"][idx,15,None]
    error=np.linalg.norm(image-d["xy"][idx,16:21],axis=-1)
    assert np.isfinite(image).all()
    torch.save(dict(model=state,arm=arm,seed=s,fold=fold,step=beststep,protocol_sha256=sha(OUT/"protocol.json")),dest/"checkpoint.pt")
    np.savez_compressed(dest/"predictions.npz",indices=idx,displacement=disp,image=image,error=error)
    write(dest/"training.json",history)
    seconds=time.monotonic()-begin
    write(dest/"complete.json",dict(passed=True,seconds=seconds,step=beststep,initial_hash=initial,draw_hash=drawhash.hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()),code_hashes=hashes()))
    print("fit",arm,fold["fold"],s,round(seconds,1),"s",flush=True)
def summary(rows):
    def avg(rr):return {k:float(np.mean([r[k] for r in rr])) for k in ["ADE","FDE"]}
    rec={g:avg([r for r in rows if r["recording"]==g]) for g in sorted({r["recording"] for r in rows})}
    return dict(window_mean=avg(rows),recording_equal={k:float(np.mean([v[k] for v in rec.values()])) for k in ["ADE","FDE"]},
      per_recording=rec,per_seed={str(s):avg([r for r in rows if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})},
      sparse=avg([r for r in rows if r["nonoverlap"]]))
def evaluate(d,inputs,meta,folds):
    rows=[];seconds=0.;replays=0;maskchecks=0;step0={}
    # Independently rebuild only historical numeric arrays after physically removing future suffix.
    truncated={k:v[:,:16].copy() if v.ndim>1 and v.shape[1]==21 else v.copy() for k,v in d.items()}
    pp,cc=numeric(truncated);short=[tensor(pp),tensor(cc)]+inputs[2:]
    assert all(torch.equal(a,b) for a,b in zip(inputs,short))
    for fold in folds:
        for s in [17,29,43]:
            initials=[];drawhash=[];counts=[]
            for arm in ARMS:
                dest=OUT/arm/f'fold_{fold["fold"]}/seed_{s}'
                ck=torch.load(dest/"checkpoint.pt",map_location="cuda",weights_only=False)
                model=Predictor().cuda();model.load_state_dict(ck["model"]);model.eval()
                z=np.load(dest/"predictions.npz");idx=z["indices"]
                pred=predict(model,inputs,idx,arm);replay=predict(model,short,idx,arm)
                assert np.array_equal(pred,z["displacement"]) and np.array_equal(pred,replay)
                error=np.sqrt(np.sum((z["image"]-d["xy"][idx,16:21])**2,axis=-1))
                assert np.array_equal(error,z["error"])
                # Perturb omitted modalities; paired single-example inference avoids batch roundoff.
                if arm!="both_rgb":
                    changed=list(inputs)
                    if arm in ["no_rgb","scene_rgb"]:changed[2]=inputs[2]+100
                    if arm in ["no_rgb","local_rgb"]:changed[3]=inputs[3]-100
                    assert np.array_equal(pred,predict(model,changed,idx,arm));maskchecks+=1
                info=read(dest/"complete.json");seconds+=info["seconds"];initials.append(info["initial_hash"]);drawhash.append(info["draw_hash"]);counts.append(info["parameters"])
                step0[arm]=step0.get(arm,0)+int(info["step"]==0)
                for j,i in enumerate(idx):
                    rows.append(dict(arm=arm,index=int(i),key=meta[i]["key"],recording=meta[i]["recording"],seed=s,
                      nonoverlap=meta[i]["nonoverlap"],ADE=float(error[j].mean()),FDE=float(error[j,-1])))
                replays+=1
            assert len(set(initials))==len(set(drawhash))==len(set(counts))==1
    for i,m in enumerate(meta):
        e=np.linalg.norm(d["xy"][i,15]-d["xy"][i,16:21],axis=-1)
        rows.append(dict(arm="hold_last",index=i,key=m["key"],recording=m["recording"],seed=0,nonoverlap=m["nonoverlap"],ADE=float(e.mean()),FDE=float(e[-1])))
    oldrows=read(SOURCE/"metrics.json")
    for r in oldrows:
        if r["arm"]=="with_translation":rows.append(dict(r,arm="previous_separated"))
    result={a:summary([r for r in rows if r["arm"]==a]) for a in ARMS+["hold_last","previous_separated"]}
    contrasts={}
    for arm in ARMS[1:]:
        a=np.array([np.mean([r["ADE"] for r in rows if r["arm"]==arm and r["index"]==i]) for i in range(len(meta))])
        b=np.array([np.mean([r["ADE"] for r in rows if r["arm"]=="no_rgb" and r["index"]==i]) for i in range(len(meta))])
        diff=a-b;rec=np.array([diff[[i for i,m in enumerate(meta) if m["recording"]==g]].mean() for g in sorted({m["recording"] for m in meta})])
        boot=np.random.default_rng(20261020).choice(rec,(5000,len(rec)),replace=True).mean(1)
        contrasts[arm]=dict(ADE_difference=float(diff.mean()),relative_ADE_percent=float(100*(a.mean()/b.mean()-1)),
          improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
    protected=read(OUT/"source_hashes.json");assert all(sha(ROOT/k)==v for k,v in protected.items())
    write(OUT/"metrics.json",rows)
    write(OUT/"comparison.json",dict(results=result,contrasts=contrasts,training_seconds=seconds,step0_selected=step0))
    write(OUT/"verification.json",dict(passed=True,checkpoint_replays=replays,future_numeric_suffix_removed_exact=True,
      disabled_visual_modality_checks=maskchecks,paired_initialization_and_batches=True,protected_files=len(protected),code_hashes=hashes()))
    print("RESULTS",{a:r["window_mean"] for a,r in result.items()},flush=True)
def smoke(d,inputs):
    seed(17);m=Predictor().cuda();m.eval();idx=np.arange(2)
    outputs=[predict(m,inputs,idx,a) for a in ARMS]
    assert all(np.array_equal(outputs[0],v) for v in outputs) and np.max(abs(outputs[0]))==0
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    for _ in range(3):
        opt.zero_grad();z=forward(m,inputs,idx,"both_rgb")
        (z-tensor((d["xy"][idx,16:21]-d["xy"][idx,15,None])/2560.)).square().mean().backward()
        opt.step()
    grad=float(m.visual[-1].weight.grad.abs().sum());assert grad>0
    write(OUT/"smoke.json",dict(passed=True,paired_initial_outputs_exact=True,initial_hold_last=True,
      visual_gradient_after_three_updates=grad,trainable_parameters=sum(p.numel() for p in m.parameters())))
def main():
    d,inputs=load();meta=read(OUT/"samples.json");folds=read(OUT/"folds.json")
    smoke(d,inputs)
    write(OUT/"training_code_hashes.json",hashes())
    begin=time.monotonic()
    for fold in folds:
        for s in [17,29,43]:
            for arm in ARMS:fit(d,inputs,meta,fold,s,arm)
    evaluate(d,inputs,meta,folds)
    write(OUT/"training_complete.json",dict(passed=True,fits=72,orchestration_seconds=time.monotonic()-begin))
if __name__=="__main__":main()
