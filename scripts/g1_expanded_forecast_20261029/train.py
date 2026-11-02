from expanded_common import *

def main():
 assert read(OUT/'preflight.json')['passed'];g,d=trainers();base=g.base;torch=g.torch;old=dict(np.load(OLD/'dataset.npz'));new=dict(np.load(OUT/'expanded_dataset.npz'));om=read(OLD/'samples.json');nm=read(OUT/'expanded_samples.json');folds=read(OUT/'folds.json');plans=read(OUT/'fold_plan.json');rows=[];replays=0;seconds=0;step0={a:0 for a in g.ARMS+d.ARMS};invalid=0
 write(OUT/'execution_code_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')})
 short={k:v[:,:16] if k in ['xy','raw','raw_valid','R','H','cam','seconds','boxes'] else v for k,v in old.items()}
 for f,p in zip(folds,plans):
  for seed in [17,29,43]:
   cm,state,source=g.load_base(f,seed);oc,nc=d.camfeat(cm,old),d.camfeat(cm,new)
   for arm in g.ARMS:g.fit(old,om,new,nm,f,p,seed,arm)
   for arm in d.ARMS:d.fit(old,om,new,nm,f,p,seed,arm,oc,nc)
   idx=np.array(f['test_indices']);ref=dict(np.load(source/'person/predictions.npz'));rows += [base.row(om[i],i,seed,'saved_translation',ref,j) for j,i in enumerate(idx)]
   hashes={}
   for arm in g.ARMS+d.ARMS:
    folder=OUT/arm/f'fold_{f["fold"]}/seed_{seed}';stored=dict(np.load(folder/'predictions.npz'));c=read(folder/'complete.json');seconds+=c['seconds'];step0[arm]+=c['step']==0;hashes[arm]=c['initial_hash']
    if arm.startswith('gru'):
     m=g.Person().to(base.DEVICE);m.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);m.eval();z=base.predict(m,cm,old,om,idx,'with_translation');trunc=base.predict(m,cm,short,om,idx,'with_translation');assert all(np.array_equal(z[k],trunc[k]) for k in z);z=base.diagnostics(z,old,idx);assert np.array_equal(z['rotation'],ref['rotation']);assert all(np.array_equal(z[k],stored[k]) for k in z);invalid+=int((z['image_depth']<=1e-6).sum())
    else:
     m=d.DModel().to(base.DEVICE);m.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=base.DEVICE,weights_only=False)['model']);pred=d.predict(m,old,idx,oc);sc=d.camfeat(cm,short);assert np.array_equal(oc,sc);assert np.array_equal(pred,d.predict(m,short,idx,sc));assert np.array_equal(pred,stored['image'])
    # Both original-only controls must exactly replay their previous same-recipe experiment.
    if arm=='gru_original':prior=ROOT/'outputs/trimmed_forecast_finetune_20261028/original_data_finetune'/f'fold_{f["fold"]}/seed_{seed}/predictions.npz'
    elif arm=='d_original':prior=ROOT/'outputs/transformer_d_trimmed_20261029/original_data'/f'fold_{f["fold"]}/seed_{seed}/predictions.npz'
    else:prior=None
    if prior:assert np.array_equal(stored['image'],np.load(prior)['image'])
    rows += [base.row(om[i],i,seed,arm,stored,j) for j,i in enumerate(idx)];replays+=1
   assert hashes['gru_original']==hashes['gru_expanded'] and hashes['d_original']==hashes['d_expanded']
 for i,m in enumerate(om):
  e=np.linalg.norm(old['xy'][i,15]-old['xy'][i,16:21],axis=-1);rows.append(base.row(m,i,0,'hold_last',{'image_error':e[None]},0))
 results={a:base.summary([r for r in rows if r['arm']==a]) for a in g.ARMS+d.ARMS+['saved_translation','hold_last']};contrasts={}
 for arm,ref in [('gru_expanded','gru_original'),('gru_expanded','saved_translation'),('d_expanded','d_original'),('d_expanded','saved_translation')]:
  a=np.array([np.mean([r['ADE'] for r in rows if r['arm']==arm and r['index']==i]) for i in range(117)]);b=np.array([np.mean([r['ADE'] for r in rows if r['arm']==ref and r['index']==i]) for i in range(117)]);diff=a-b;rec=np.array([diff[[i for i,m in enumerate(om) if m['recording']==group]].mean() for group in sorted({m['recording'] for m in om})]);boot=np.random.default_rng(20261029).choice(rec,(5000,6),replace=True).mean(1);contrasts[arm+'_vs_'+ref]=dict(ADE_difference=float(diff.mean()),relative_percent=float((a.mean()/b.mean()-1)*100),improved_windows=int((diff<0).sum()),improved_recordings=int((rec<0).sum()),recording_bootstrap_95=np.quantile(boot,[.025,.975]).tolist())
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/p)==s for p,s in protected.items());write(OUT/'metrics.json',rows);write(OUT/'comparison.json',dict(results=results,contrasts=contrasts,step0=step0,training_seconds=seconds));write(OUT/'verification.json',dict(passed=True,checkpoint_replays=replays,future_physically_truncated_exact=True,original_control_replays_exact=36,camera_unchanged=True,paired_initializations=True,invalid_rays=invalid,protected_files=len(protected)));write(OUT/'training_complete.json',dict(passed=True,fits=72,camera_fits=0));print({a:r['window_mean'] for a,r in results.items()},flush=True)

if __name__=='__main__':main()
