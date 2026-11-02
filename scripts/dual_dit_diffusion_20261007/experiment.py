"""Versioned six-fold, one-seed dual DiT experiment. No source mutations."""
from pathlib import Path
import sys,json,hashlib,time,copy,argparse
import numpy as np
import torch
from network import ConditionalDiT
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/factorized_forecast_v1_20261002'))
from shared import input_history,targets,project,warp,SCALE,P,F
OUT=ROOT/'outputs/dual_dit_diffusion_20261007'
OLD=ROOT/'outputs/biflow_aria_v1_20261002'
BASE=ROOT/'outputs/factorized_forecast_v1_20261002'
CAM=ROOT/'outputs/camera_full_data_20261002'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def seed(s):np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def fixed_noise(keys,channels,count=5,length=15):
 return np.stack([np.random.default_rng(int(hashlib.sha256(str(k).encode()).hexdigest()[:16],16)).normal(size=(count,length,channels)) for k in keys]).astype(np.float32)
def camera_keys(x):return ['camera:'+hashlib.sha256(np.asarray(a,dtype=np.float32).tobytes()).hexdigest() for a in x]
def balanced(indices,meta,n,rng):
 groups=sorted({meta[i]['recording'] for i in indices})
 return np.array([rng.choice([i for i in indices if meta[i]['recording']==g]) for g in rng.choice(groups,n)])

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=dict(name='Dual conditional DiT diffusion V1',date='2026-10-07',seed=17,folds=6,
  scope='Both camera and pedestrian branches replaced. Original 117 windows, same six recording folds.',
  camera_updates=2000,camera_batch=256,camera_validation_every=100,
  person_updates=240,person_batch=48,person_validation_every=20,learning_rate=.001,weight_decay=.01,
  width=64,diffusion_steps=1000,sampling_steps=20,samples=5,reconstruction_weight=.5,
  parameterization='cosine discrete VP schedule, v-prediction MSE, training-only coordinate normalization',
  primary_sampler='DDIM eta=0; deterministic conditioned on initial noise',secondary_sampler='stochastic respaced DDIM eta=1, same weights and 20 model evaluations',
  continuous_sde_ode_implemented=False,
  primary_point_estimate='Mean of five residual samples and mean of five relative rotation-vector samples, followed by fixed projection; not best-of-K. Paired joint samples saved separately.',
  camera_selection='Full validation recording rotation-vector MSE with primary sampler',
  person_selection='Validation reference-orientation residual pixel ADE with primary sampler',
  comparison='Saved seed17 GRU+flow system; equal data and optimizer update counts, not capacity/FLOP/architecture-matched diffusion-only ablation',
  no_test_tuning=True,labels='Reviewed offline SAM references and historical offline MPS, rotation-only compensation, not pure physical pedestrian motion',
  note='One model seed does not establish training-seed robustness. Stochastic DDIM has fixed recorded RNG seed, not a calibrated uncertainty claim.')
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==cfg
 else:write(OUT/'protocol.json',cfg)
 personfold=read(OLD/'folds.json');camfold=read(CAM/'folds.json')
 for p,c in zip(personfold,camfold):
  assert p['fold']==c['fold'] and p['test']==c['test'] and p['validation']==c['validation']
  assert p['test'] not in c['train'] and p['validation'] not in c['train']
 paths=[OLD/'dataset.npz',OLD/'samples.json',OLD/'folds.json',CAM/'dataset.npz',CAM/'folds.json',BASE/'protocol.json']
 for f in range(6):
  paths += list((BASE/f'fold_{f}/seed_17/separated_dual').glob('*'))
  paths += list((CAM/f'fold_{f}/seed_17/full').glob('*'))
 paths += list((ROOT/'scripts/factorized_forecast_v1_20261002').glob('*.py'))
 if not (OUT/'source_hashes.json').exists():write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()})
 write(OUT/'code_hashes.json',{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
 write(OUT/'folds.json',personfold)
 return cfg

@torch.no_grad()
def infer(model,history,origin=None,noise=None,eta=0.,steps=20,rngseed=20261007):
 model.eval();pieces=[]
 generator=torch.Generator(device=DEVICE).manual_seed(rngseed)
 if noise is None:noise=fixed_noise(camera_keys(history),model.coordinates)
 for start in range(0,len(history),128):
  sl=slice(start,start+128)
  p=model.sample(tensor(history[sl]),tensor(noise[sl]),tensor(origin[sl]) if origin is not None else None,
                 steps=steps,eta=eta,generator=generator)
  pieces.append(p.cpu().numpy())
 return np.concatenate(pieces)

def camera_fit(full,fold,cfg):
 dest=OUT/f'fold_{fold["fold"]}/camera';dest.mkdir(parents=True,exist_ok=True)
 seed(cfg['seed']);m=ConditionalDiT(3,3,width=cfg['width']).to(DEVICE)
 train=full[fold['train_indices']];val=full[fold['val_indices']]
 m.fit_normalization(tensor(train[:,:P]),tensor(train[:,P:]))
 if (dest/'checkpoint.pt').exists():
  saved=torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False);m.load_state_dict(saved['model']);return m
 optimizer=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 rng=np.random.default_rng(cfg['seed']);left=np.array([],int);tx=tensor(train[:,:P]);ty=tensor(train[:,P:]);best=float('inf');logs=[];start=time.monotonic()
 noise=fixed_noise(camera_keys(val[:,:P]),3)
 for step in range(1,cfg['camera_updates']+1):
  while len(left)<cfg['camera_batch']:left=np.concatenate([left,rng.permutation(len(train))])
  idx,left=left[:cfg['camera_batch']],left[cfg['camera_batch']:]
  m.train();loss=m.loss(tx[idx],ty[idx]);assert torch.isfinite(loss)
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
  if step%cfg['camera_validation_every']==0:
   pred=infer(m,val[:,:P],noise=noise,steps=cfg['sampling_steps']).mean(1)
   score=float(np.mean((pred-val[:,P:])**2));assert np.isfinite(score)
   logs.append(dict(step=step,loss=loss.item(),validation_rotation_mse=score))
   if score<best:best=score;state=copy.deepcopy(m.state_dict());beststep=step
   write(dest/'progress.json',dict(step=step,total=cfg['camera_updates'],best_step=beststep,seconds=time.monotonic()-start))
   print(f'camera fold {fold["fold"]} step {step}: val MSE {score:.6f}',flush=True)
 m.load_state_dict(state);m.eval()
 torch.save(dict(model=state,fold=fold,selected_step=beststep,validation=best,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt')
 write(dest/'training.json',logs);write(dest/'complete.json',dict(seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),best_step=beststep))
 return m

def person_fit(data,meta,fold,cfg):
 dest=OUT/f'fold_{fold["fold"]}/person';dest.mkdir(parents=True,exist_ok=True)
 seed(cfg['seed']);m=ConditionalDiT(6,2,origin_features=2,width=cfg['width']).to(DEVICE)
 idx=fold['train_indices'];h=input_history(data,idx);y=targets(data,idx,h['origin'])
 m.fit_normalization(tensor(h['seq']),tensor(y['future'].reshape(-1,F,2)),tensor(h['origin']),tensor(y['past'].reshape(-1,P,2)))
 if (dest/'checkpoint.pt').exists():
  saved=torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False);m.load_state_dict(saved['model']);return m
 optimizer=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
 idx=np.array(fold['val_indices']);vh=input_history(data,idx);vy=targets(data,idx,vh['origin'])['stable'][:,P:]
 noise=fixed_noise(['person:'+meta[i]['key'] for i in idx],2)
 best=float('inf');logs=[];start=time.monotonic()
 for step in range(1,cfg['person_updates']+1):
  epoch=(step-1)//2;update=(step-1)%2;rng=np.random.default_rng(cfg['seed']*100000+epoch*10+update)
  idx=balanced(fold['train_indices'],meta,cfg['person_batch'],rng);h=input_history(data,idx,'train',rng);y=targets(data,idx,h['origin'])
  m.train();loss=m.loss(tensor(h['seq']),tensor(y['future'].reshape(-1,F,2)),tensor(h['origin']),tensor(y['past'].reshape(-1,P,2)),cfg['reconstruction_weight']);assert torch.isfinite(loss)
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
  if step%cfg['person_validation_every']==0:
   pred=(infer(m,vh['seq'],vh['origin'],noise,steps=cfg['sampling_steps']).mean(1)+vh['origin'][:,None])*SCALE
   score=float(np.linalg.norm(pred-vy,axis=-1).mean());assert np.isfinite(score)
   logs.append(dict(step=step,loss=loss.item(),validation_residual_ADE=score))
   if score<best:best=score;state=copy.deepcopy(m.state_dict());beststep=step
 m.load_state_dict(state);m.eval()
 torch.save(dict(model=state,fold=fold,selected_step=beststep,validation=best,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt')
 write(dest/'training.json',logs);write(dest/'complete.json',dict(seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),best_step=beststep))
 print(f'person fold {fold["fold"]}: selected step {beststep}',flush=True)
 return m

def predict(data,meta,fold,camera,person,cfg):
 idx=np.array(fold['test_indices']);h=input_history(data,idx)
 noise=fixed_noise(['person:'+meta[i]['key'] for i in idx],2)
 saved=np.load(BASE/f'fold_{fold["fold"]}/seed_17/separated_dual/predictions_reviewed.npz')
 assert np.array_equal(saved['indices'],idx)
 oldcam=np.load(CAM/f'fold_{fold["fold"]}/seed_17/full/predictions.npz')['person_rotation']
 result=dict(indices=idx,truth=data['xy'][idx,P:],old_prediction=saved['full'],old_component=saved['component'],old_camera=oldcam[idx])
 for name,eta in [('ddim0',0.),('ddim1',1.)]:
  cs=infer(camera,data['cam'][idx,:P],eta=eta,steps=cfg['sampling_steps'])
  ps=infer(person,h['seq'],h['origin'],noise,eta=eta,steps=cfg['sampling_steps'])
  cm=np.zeros_like(data['cam'][:,P:]);cm[idx]=cs.mean(1)
  pm=(ps.mean(1)+h['origin'][:,None])*SCALE
  result[name+'_camera_samples']=cs;result[name+'_person_samples']=ps
  result[name+'_component']=pm
  result[name],result[name+'_depth']=project(pm,cm,data,idx)
  paired=[]
  for k in range(cfg['samples']):
   cm[idx]=cs[:,k];pp,dd=project((ps[:,k]+h['origin'][:,None])*SCALE,cm,data,idx);paired.append(pp)
  result[name+'_joint_samples']=np.stack(paired,1)
  if eta==0:
   cm[idx]=cs.mean(1)
   for label,p,c in [('new_person_old_camera',pm,oldcam),('old_person_new_camera',saved['component'],cm),('new_person_GT_camera',pm,data['cam'][:,P:])]:
    result[label],result[label+'_depth']=project(p,c,data,idx)
 np.savez_compressed(OUT/f'fold_{fold["fold"]}/predictions.npz',**result)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--fold',type=int);args=parser.parse_args()
 cfg=prepare()
 if args.prepare_only:return
 data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');full=np.load(CAM/'dataset.npz')['rotation']
 pf=read(OLD/'folds.json');cf=read(CAM/'folds.json');start=time.monotonic()
 print('Device',DEVICE,torch.__version__,flush=True)
 for p,c in zip(pf,cf):
  if args.fold is not None and p['fold']!=args.fold:continue
  camera=camera_fit(full,c,cfg);person=person_fit(data,meta,p,cfg)
  predict(data,meta,p,camera,person,cfg)
 write(OUT/('training_complete.json' if args.fold is None else f'fold_{args.fold}/finished.json'),dict(seconds=time.monotonic()-start,seed=17,folds=6 if args.fold is None else 1))

if __name__=='__main__':main()
