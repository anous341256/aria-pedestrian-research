from pathlib import Path
import sys,time,copy,json,hashlib
import numpy as np
from d_model import DModel,torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/transformer_d_trimmed_20261029';OLD=ROOT/'outputs/translation_forecast_20261020';TRIM=ROOT/'outputs/trimmed_forecast_finetune_20261028';sys.path.insert(0,str(ROOT/'scripts/translation_forecast_20261020'))
import run as base
from network import Camera
from prepare import read,write,sha
ARMS=['original_data','trimmed_data'];torch.set_num_threads(2)

def history(data,idx):
 obs=data['xy'][idx,:16]/base.SCALE;origin=obs[:,-1];delta=np.zeros_like(obs);delta[:,1:]=np.diff(obs,axis=1);seq=np.concatenate([obs-origin[:,None],delta,np.ones((*obs.shape[:2],1)),data['seconds'][idx,:16,None]/1.5],-1);k=data['K'][idx];cal=np.stack([k[:,0,0]/2560,k[:,1,1]/1920,k[:,0,2]/2560,k[:,1,2]/1920],-1);return seq,np.concatenate([origin,cal],-1),origin

@torch.no_grad()
def camfeat(cm,data):
 x=base.camera_inputs(data,np.arange(len(data['xy'])),'with_translation');seq,_=cm.encoder(base.tensor(x));return np.concatenate([x,seq.cpu().numpy()],-1)

@torch.no_grad()
def predict(model,data,idx,cf):
 # Fixed one-window replay batches avoid tiny batch-dependent attention numerics.
 model.eval();out=[]
 for i in idx:
  seq,cond,origin=history(data,[i]);offset=model(base.tensor(seq),base.tensor(cond),base.tensor(cf[[i]])).cpu().numpy();out.append((offset[0]+origin[0])*base.SCALE)
 return np.stack(out)

def fit(old,meta,new,nmeta,fold,plan,seed,arm,oc,nc):
 folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 start=time.monotonic();base.seed_all(seed);model=DModel().to(base.DEVICE);initial=base.statehash(model);data,tm,ti,cf=(old,meta,fold['train_indices'],oc) if arm==ARMS[0] else (new,nmeta,plan['trimmed_train_indices'],nc)
 assert all(tm[i]['recording'] not in [fold['test'],fold['validation']] for i in ti)
 def val():return float(np.linalg.norm(predict(model,old,fold['val_indices'],oc)-old['xy'][fold['val_indices'],16:21],axis=-1).mean())
 best=val();state=copy.deepcopy(model.state_dict());beststep=0;logs=[dict(step=0,val_ADE=best)];opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);draws=hashlib.sha256()
 for step in range(240):
  epoch,update=divmod(step,2);rng=np.random.default_rng(seed*100000+epoch*10+update);idx=base.balanced(ti,tm,rng);draws.update(idx.tobytes());seq,cond,origin=history(data,idx);y=data['xy'][idx,16:21]/base.SCALE-origin[:,None];model.train();pred=model(base.tensor(seq),base.tensor(cond),base.tensor(cf[idx]));loss=(pred-base.tensor(y)).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if (step+1)%20==0:
   v=val();logs.append(dict(step=step+1,val_ADE=v,loss=float(loss.detach().cpu())))
   if v<best:best=v;state=copy.deepcopy(model.state_dict());beststep=step+1
   write(OUT/'progress.json',dict(arm=arm,fold=fold['fold'],seed=seed,step=step+1))
 model.load_state_dict(state);idx=np.array(fold['test_indices']);pred=predict(model,old,idx,oc);torch.save(dict(model=state,arm=arm,fold=fold,seed=seed,step=beststep),folder/'checkpoint.pt');np.savez_compressed(folder/'predictions.npz',indices=idx,image=pred,image_error=np.linalg.norm(pred-old['xy'][idx,16:21],axis=-1));write(folder/'training.json',logs);write(folder/'complete.json',dict(seconds=time.monotonic()-start,step=beststep,initial_hash=initial,draws_hash=draws.hexdigest(),training_indices=ti,parameters=sum(p.numel() for p in model.parameters())));print(arm,fold['fold'],seed,beststep,round(time.monotonic()-start,1),flush=True)

def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=dict(model='User confirmed D:2self layers per stream, simultaneous bidirectional cross-attention,concat,MLP direct image forecast. width64,heads4,FF55,dropout0. Historical camera rotation3+translation3+frozen matching translation-camera GRU32 =>38. Person6;origin/calibration6. No geometry output or flow loss.',horizon='16history1.5s->5future.5s',arms=ARMS,training='Train each D from scratch, identical initial weights per fold/seed; old 6history->15future D checkpoint dimensions incompatible.240updates,batch48,lr.001,wd.01,clip1,noaugmentation;future normalized image MSE; original validation imageADE every20,step0 eligible. Recording-balanced draws, actual data differ.',data='Old117 vs locked412trimmed-only; same original six folds/seeds17,29,43; both test and validation recordings excluded from each training arm. No expansion-data inclusion. Frozen matching saved camera encoder trained excluding heldout recordings.',comparison='D original vs D trimmed isolates data recipe within D. Saved GRU and its fine-tunes are system references, not attention-only ablation. Previous .5->1.5 D scores not directly comparable.',preserve='All old weights/data unchanged; no test tuning or default-baseline replacement.')
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==cfg
 else:
  write(OUT/'protocol.json',cfg);write(ROOT/'plans/TRANSFORMER_D_TRIMMED_20261029.txt',cfg);protected={str(p.relative_to(ROOT)):sha(p) for folder in [OLD,TRIM] for p in folder.rglob('*') if p.is_file() and p.suffix in ['.pt','.npz','.json']};protected.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts/transformer_fusion_20261005').glob('*.py')});write(OUT/'protected_sources.json',protected)
 old=dict(np.load(OLD/'dataset.npz'));new=dict(np.load(TRIM/'trimmed_dataset.npz'));meta=read(OLD/'samples.json');nm=read(TRIM/'trimmed_samples.json');folds=read(OLD/'folds.json');plans=read(TRIM/'fold_plan.json');rows=[];replay=0;steps={a:0 for a in ARMS};seconds=0
 for f,p in zip(folds,plans):
  for seed in [17,29,43]:
   cm=Camera().to(base.DEVICE);source=OLD/'with_translation'/f'fold_{f["fold"]}/seed_{seed}/camera/checkpoint.pt';cm.load_state_dict(torch.load(source,map_location=base.DEVICE,weights_only=False)['model']);cm.eval()
   for param in cm.parameters():param.requires_grad_(False)
   oc,nc=camfeat(cm,old),camfeat(cm,new)
   for arm in ARMS:fit(old,meta,new,nm,f,p,seed,arm,oc,nc)
   checks=[]
   for arm in ARMS:
    folder=OUT/arm/f'fold_{f["fold"]}/seed_{seed}';m=DModel().to(base.DEVICE);m.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);z=dict(np.load(folder/'predictions.npz'));idx=z['indices'];pred=predict(m,old,idx,oc);short={k:v[:,:16] if k in ['xy','raw','raw_valid','R','H','cam','seconds','boxes'] else v for k,v in old.items()};sc=camfeat(cm,short);assert np.array_equal(oc,sc);assert np.array_equal(pred,z['image']) and np.array_equal(pred,predict(m,short,idx,sc));assert np.isfinite(pred).all();replay+=1
    rows += [base.row(meta[i],i,seed,arm,z,j) for j,i in enumerate(idx)];c=read(folder/'complete.json');checks.append(c);steps[arm]+=c['step']==0;seconds+=c['seconds']
   assert checks[0]['initial_hash']==checks[1]['initial_hash']
 refs=read(TRIM/'metrics.json');rows += [r for r in refs if r['arm'] in ['saved_translation','original_data_finetune','trimmed_data_finetune','hold_last']];results={a:base.summary([r for r in rows if r['arm']==a]) for a in ARMS+['saved_translation','original_data_finetune','trimmed_data_finetune','hold_last']};a=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ARMS[1] and r['index']==i]) for i in range(117)]);b=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ARMS[0] and r['index']==i]) for i in range(117)]);diff=a-b;rec=np.array([diff[[i for i,m in enumerate(meta) if m['recording']==g]].mean() for g in sorted({m['recording'] for m in meta})]);boot=np.random.default_rng(20261029).choice(rec,(5000,6),replace=True).mean(1);protected=read(OUT/'protected_sources.json');assert all(sha(ROOT/k)==v for k,v in protected.items())
 write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=results,trimmed_vs_original=dict(ADE_difference=float(diff.mean()),relative_percent=float((a.mean()/b.mean()-1)*100),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist()),step0=steps,training_seconds=seconds));write(OUT/'verification.json',dict(passed=True,checkpoint_replays=replay,future_truncated_exact=True,paired_initialization=True,recording_disjoint=True,protected_files=len(protected)));write(OUT/'training_complete.json',dict(passed=True,models=36));print({a:r['window_mean'] for a,r in results.items()},flush=True)
if __name__=='__main__':main()
