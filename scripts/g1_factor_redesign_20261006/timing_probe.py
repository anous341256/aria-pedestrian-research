"""Small runtime probe with synthetic inputs/responses; no research scores."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/transformer_matched97_20261006'))
from network import Predictor
torch.set_num_threads(1)

def main():
 out=ROOT/'outputs/g1_factor_redesign_20261006';bank=ROOT/'outputs/transformer_matched97_20261006/new'
 cfg=json.loads((bank/'protocol.json').read_text(encoding='utf8'))
 device='cuda' if torch.cuda.is_available() else 'cpu'
 seq=torch.zeros(32,6,6,device=device);cond=torch.zeros(32,6,device=device);cam=torch.zeros(32,6,35,device=device)
 def sync():
  if device=='cuda':torch.cuda.synchronize()
 timings=[]
 with torch.inference_mode():
  for variant in cfg['variants']:
   sync();start=time.perf_counter()
   ck=torch.load(bank/f'fold_0/seed_17/{variant}/checkpoint.pt',map_location=device,weights_only=False)
   model=Predictor(variant,ck['ff_width']).to(device);model.load_state_dict(ck['model']);model.eval();sync();loaded=time.perf_counter()-start
   args=(seq,cond,cam if model.uses_camera_features else None)
   for _ in range(5):model(*args)
   sync();start=time.perf_counter()
   for _ in range(50):model(*args)
   sync();elapsed=time.perf_counter()-start
   timings.append(dict(variant=variant,load_seconds=loaded,batch32_forward_ms=elapsed/50*1000))
 z=np.load(out/'design_matrix.npz');x=z['levels'];rc=z['recording_dummies'];w=z['weights'];rng=np.random.default_rng(20261006)
 # Simultaneous 24 artificial responses: 6 architectures x 4 metrics.
 y=rng.normal(size=(len(x),24));start=time.perf_counter();valid=0
 for _ in range(2000):
  mult=rng.multinomial(rc.shape[1],np.full(rc.shape[1],1/rc.shape[1]));ww=w*(rc@mult)
  a=np.c_[rc[:,mult>0],x];gram=a.T@(ww[:,None]*a)
  if np.linalg.matrix_rank(gram)<len(gram):continue
  np.linalg.solve(gram,a.T@(ww[:,None]*y));valid+=1
 seconds=time.perf_counter()-start
 result=dict(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,synthetic_inputs_only=True,research_predictions_or_metrics_computed=False,network_timings=timings,synthetic_cluster_bootstrap=dict(replicates=2000,simultaneous_responses=24,valid_fits=valid,seconds=seconds),estimate_scope='Microbenchmark excludes data assembly, camera/geometry, full validation, reporting, and tool/edit latency.')
 (out/'timing_probe.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
