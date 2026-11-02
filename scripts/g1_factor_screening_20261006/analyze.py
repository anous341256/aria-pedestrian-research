"""Frozen-weight scenario associations, clustered uncertainty and diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from setup import *
import time
FACTORS=list('ABCDEFGH');METRICS=['ADE','FDE','nADE','gain_ADE']

def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x

def fit(y,x,rc,w,mask=None,columns=None,interaction=None):
 n=len(x);mask=np.ones(n,bool) if mask is None else mask
 cols=list(range(8)) if columns is None else columns
 z=rc[mask];z=z[:,np.any(z,axis=0)];a=np.c_[z,x[mask][:,cols]]
 if interaction is not None:a=np.c_[a,np.prod(x[mask][:,interaction],axis=1)]
 ww=w[mask];ww=ww/ww.sum();g=a.T@(ww[:,None]*a)
 rank=int(np.linalg.matrix_rank(np.sqrt(ww[:,None])*a));status=dict(n=int(mask.sum()),recordings=z.shape[1],rank=rank,columns=a.shape[1],condition=float(np.linalg.cond(g)))
 effect=np.full((8,y.shape[1]),np.nan)
 if rank==a.shape[1]:
  beta=np.linalg.solve(g,a.T@(ww[:,None]*y[mask]));effect[cols]=2*beta[z.shape[1]:z.shape[1]+len(cols)]
 status['estimable']=rank==a.shape[1]
 return effect,status

def main():
 begin=time.perf_counter();assert read(OUT/'inference_audit.json')['passed']
 data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');n=len(meta)
 x=np.array([m['levels'] for m in meta],float);w=np.array([m['analysis_weight'] for m in meta]);recs=sorted({m['recording'] for m in meta});rc=np.array([[m['recording']==rec for rec in recs] for m in meta],float)
 target=data['xy'][:,6:];height=data['boxes'][:,5,3]-data['boxes'][:,5,1]
 hold=np.load(OUT/'predictions_hold_last.npz')['pred'];hold_error=np.linalg.norm(hold-target,axis=-1);holdade=hold_error.mean(-1)
 response=[];invalids={};perwindow=[];overall={};replay=read(OUT/'inference_audit.json')
 for v in VARIANTS:
  z=np.load(OUT/f'predictions_{v}.npz');e=np.linalg.norm(z['pred']-target[None],axis=-1);bad=(~np.isfinite(e))|(z['depth']<=1e-6);e[bad]=32000
  ade=e.mean(-1);fde=e[:,:,-1];res=np.stack([ade,fde,ade/height[None],holdade[None]-ade],-1);response.append(res)
  invalids[v]=dict(steps=int(bad.sum()),windows=int(bad.any(-1).sum()))
  avg=res.mean(0);overall[v]=dict(unweighted=dict(zip(METRICS,avg.mean(0).tolist())),weighted=dict(zip(METRICS,(w@avg).tolist())),recording_equal=dict(zip(METRICS,np.array([avg[rc[:,r]>0].mean(0) for r in range(9)]).mean(0).tolist())),seed_weighted={str(s):dict(zip(METRICS,(w@res[j]).tolist())) for j,s in enumerate(SEEDS)})
  for i,m in enumerate(meta):perwindow.append(dict(key=m['key'],variant=v,recording=m['recording'],weight=w[i],seed_mean=dict(zip(METRICS,avg[i])),seeds=[dict(seed=s,**dict(zip(METRICS,res[j,i]))) for j,s in enumerate(SEEDS)]))
 # response dimensions: model,seed,window,metric; fit all outcomes jointly.
 response=np.stack(response);y=response.mean(1).transpose(1,0,2).reshape(n,-1)
 effect,main_status=fit(y,x,rc,w);assert main_status['estimable']
 seed_effect=[]
 for s in range(3):seed_effect.append(fit(response[:,s].transpose(1,0,2).reshape(n,-1),x,rc,w)[0])
 seed_effect=np.stack(seed_effect)
 loo=[]
 for r in range(9):
  e,info=fit(y,x,rc,w,mask=rc[:,r]==0);assert info['estimable'];loo.append(e)
 loo=np.stack(loo)
 # Recording-cluster pairs bootstrap, all 24 outcomes share each draw.
 rng=np.random.default_rng(20261006);boot=[];rejected=0
 for _ in range(2000):
  mult=rng.multinomial(9,np.full(9,1/9));bw=w*(rc@mult);mask=bw>0
  e,info=fit(y,x,rc,bw,mask=mask)
  if info['estimable']:boot.append(e)
  else:rejected+=1
 boot=np.stack(boot);assert len(boot)>=1800
 ci=np.quantile(boot,[.025,.975],axis=0)
 subset=read(DESIGN/'sensitivity_subsets.json');diagnostics={}
 def add(name,mask=None,weights=w,columns=None,interaction=None):
  e,info=fit(y,x,rc,weights,mask,columns,interaction)
  mm=np.ones(n,bool) if mask is None else mask;ww=weights[mm]/weights[mm].sum();xx=x[mm];mu=ww@xx;cent=xx-mu
  cov=cent.T@(ww[:,None]*cent);cor=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
  diagnostics[name]=dict(effect=e.reshape(8,6,4),fit=info,factor_mean=mu,max_abs_pair_correlation=float(abs(cor-np.eye(8)).max()))
 add('unweighted',weights=np.full(n,1/n))
 for k,keys in subset.items():add(k,mask=np.array([m['key'] in set(keys) for m in meta]))
 add('without_H',columns=list(range(7)))
 for term in ['AF','AG','CH','EG']:add('interaction_'+term,interaction=[FACTORS.index(k) for k in term])
 # Hold-last uses the same design: a reference for intrinsic image displacement.
 hy=np.c_[holdade,hold_error[:,-1],holdade/height,np.zeros(n)];he,hi=fit(hy,x,rc,w)
 overall['hold_last']=dict(unweighted=dict(zip(METRICS,hy.mean(0).tolist())),weighted=dict(zip(METRICS,(w@hy).tolist())),recording_equal=dict(zip(METRICS,np.array([hy[rc[:,r]>0].mean(0) for r in range(9)]).mean(0).tolist())))
 effect=effect.reshape(8,6,4);ci=ci.reshape(2,8,6,4);seed_effect=seed_effect.reshape(3,8,6,4);loo=loo.reshape(9,8,6,4)
 results=[]
 for f in range(8):
  for v in range(6):
   rr=dict(factor=FACTORS[f],variant=VARIANTS[v],metrics={})
   for k,m in enumerate(METRICS):
    val=effect[f,v,k];sign=np.sign(val)
    rr['metrics'][m]=dict(effect=val,bootstrap95=[ci[0,f,v,k],ci[1,f,v,k]],loo_range=[loo[:,f,v,k].min(),loo[:,f,v,k].max()],loo_same_sign=int((np.sign(loo[:,f,v,k])==sign).sum()),seed_effects=seed_effect[:,f,v,k],seed_same_sign=int((np.sign(seed_effect[:,f,v,k])==sign).sum()),sensitivity={name:d['effect'][f,v,k] for name,d in diagnostics.items()})
   results.append(rr)
 # Descriptive weighted contrasts and distributions alongside adjusted effects.
 groups={}
 for f in range(8):
  groups[FACTORS[f]]={}
  for lv in [-1,1]:
   mask=x[:,f]==lv;ww=w[mask]/w[mask].sum()
   groups[FACTORS[f]][str(lv)]=dict(n=int(mask.sum()),weight=float(w[mask].sum()),outcomes=(ww@y[mask]).reshape(6,4))
 np.savez_compressed(OUT/'analysis_arrays.npz',responses=response,effects=effect,bootstrap_effects=boot.reshape(-1,8,6,4),seed_effects=seed_effect,loo_effects=loo,hold_effects=he,weights=w,levels=x)
 for name,obj in [('overall',overall),('factor_effects',results),('sensitivity',diagnostics),('group_descriptives',groups),('per_window_metrics',perwindow)]:write(OUT/f'{name}.json',clean(obj))
 write(OUT/'analysis_audit.json',clean(dict(passed=True,windows=n,recordings=9,primary_fit=main_status,bootstrap_replicates=2000,bootstrap_valid=len(boot),bootstrap_rank_rejected=rejected,interval_warning='Exploratory percentile cluster bootstrap with only nine recordings; no multiple-comparison significance claim.',invalid_predictions=invalids,seconds=time.perf_counter()-begin,metrics=METRICS,variants=VARIANTS,factors=FACTORS,hold_reference_fit=hi)))
 print('OVERALL',json.dumps({v:{'ADE':r['weighted']['ADE'],'gain':r['weighted']['gain_ADE']} for v,r in overall.items()}),flush=True)
 print('ADE effects rows A-H, columns model A-F\n',np.round(effect[:,:,0],1),flush=True)
 print('Gain effects\n',np.round(effect[:,:,3],1),flush=True)
 print('Bootstrap',len(boot),'valid;',time.perf_counter()-begin,'sec',flush=True)
if __name__=='__main__':main()
