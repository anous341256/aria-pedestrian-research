"""Independent least-squares and metric verification of the executed analysis."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from setup import *

def main():
 meta=read(OUT/'samples.json');z=np.load(OUT/'analysis_arrays.npz');d=np.load(OUT/'dataset.npz')
 weights=np.array([r['analysis_weight'] for r in meta]);levels=np.array([r['levels'] for r in meta]);recs=sorted({r['recording'] for r in meta})
 design=np.c_[np.array([[r['recording']==rec for rec in recs] for r in meta],float),levels]
 y=z['responses'].mean(1).transpose(1,0,2).reshape(214,24)
 # SVD-based lstsq, separate from the normal-equation implementation.
 beta,_,rank,_=np.linalg.lstsq(np.sqrt(weights[:,None])*design,np.sqrt(weights[:,None])*y,rcond=None)
 effect=(2*beta[-8:]).reshape(8,6,4)
 difference=float(abs(effect-z['effects']).max());assert difference<1e-8
 linear=float(abs(z['effects'][:,:,3]-(z['hold_effects'][:,0,None]-z['effects'][:,:,0])).max());assert linear<1e-8
 seed=float(abs(z['seed_effects'].mean(0)-z['effects']).max());assert seed<1e-8
 metricdiff=0.
 for j,v in enumerate(VARIANTS):
  p=np.load(OUT/f'predictions_{v}.npz');error=np.sqrt(((p['pred']-d['xy'][None,:,6:])**2).sum(-1));error[p['depth']<=1e-6]=32000
  metricdiff=max(metricdiff,float(abs(error.mean(-1)-z['responses'][j,:,:,0]).max()))
 assert metricdiff<1e-8
 assert read(OUT/'analysis_audit.json')['bootstrap_valid']==2000
 designmanifest=read(DESIGN/'manifest.json')
 for p,h in designmanifest['files'].items():assert sha(DESIGN/p)==h,p
 protected=read(OUT/'protected_sources.json');changed=[p for p,h in protected.items() if sha(ROOT/p)!=h];assert not changed,changed
 hist={};geom_errors=[]
 # Verify historical compensation independently using only the past R and K.
 for i in range(214):
  R=d['R'][i,:6].astype(float);K=d['K'][i].astype(float)
  H=np.array([K@R[5].T@r@np.linalg.inv(K) for r in R]);points=np.c_[d['xy'][i,:6],np.ones(6)]
  a=np.einsum('nij,nj->ni',H,points);b=np.einsum('nij,nj->ni',d['H'][i,:6],points)
  geom_errors.append(float(abs(a[:,:2]/a[:,2:]-b[:,:2]/b[:,2:]).max()))
 assert max(geom_errors)<.002,max(geom_errors)
 write(OUT/'verification.json',dict(passed=True,weighted_design_rank=int(rank),svd_effect_max_difference=difference,gain_linear_identity_max_difference=linear,seed_mean_commutes_with_fit_max_difference=seed,independent_ADE_max_difference=metricdiff,history_geometry_max_px=max(geom_errors),frozen_design_files_verified=len(designmanifest['files']),protected_files_verified=len(protected),changed_files=changed,original_replay_max_px=max(r['original_replay_max_px'] for r in read(OUT/'inference_audit.json')['checks']),future_removed_max_px=max(r['future_removed_effect_px'] for r in read(OUT/'inference_audit.json')['checks'])))
 print(read(OUT/'verification.json'),flush=True)
if __name__=='__main__':main()
