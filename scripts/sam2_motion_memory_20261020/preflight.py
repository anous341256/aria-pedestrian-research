"""Geometry and zero-intervention checks before full inference."""
import time,copy,inspect,hashlib,json
import run as task
from run import np,torch,near,ROOT,OUT,read,write,MotionMemory,metadata
def main():
 torch.manual_seed(17)
 result={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  tracker=near.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
  model=tracker._predictor;model.eval()
  result['vendor_source_hashes']={}
  for cls in [type(model),type(model.memory_attention),type(model.memory_attention.layers[0].cross_attn_image)]:
   path=inspect.getfile(cls)
   result['vendor_source_hashes'][path]=task.digest(task.Path(path))
  c=read(OUT/'selection.json')[0];cal,cached=task.load_frames(c,tracker)
  entry=c['entries'][0];anchor=entry['anchor_frame']
  f=anchor-1 if anchor>entry['start'] else anchor+1
  assert entry['start']<=f<=entry['end']
  R,K,times=cal['R'],cal['K'],cal['times_ns']/1e9
  pp,feat,hsv=cached[anchor]
  seed,score,_=task.image_candidate(model,pp,entry['anchor_box'])
  state=near.init_state(tracker,pp,seed,feat,hsv,0,0.,np.eye(3))
  box=near.describe_mask(seed)['bbox_xyxy']
  task.add_meta(state,0,seed,box,R[anchor],times[anchor],1.)
  ppf,ff,hf=cached[f]
  def candidate():
   return near.inc.base.make_candidate(model,ppf,state,1,entry['id'],'memory',None,ff,hf)
  torch.cuda.synchronize();t=time.perf_counter();old=candidate();torch.cuda.synchronize()
  result['original_candidate_seconds']=time.perf_counter()-t
  control=MotionMemory(model)
  new=candidate()
  result['inactive_exact']=bool(torch.equal(old['model_output']['pred_masks'],new['model_output']['pred_masks']))
  state['motion'].predict(abs(times[f]-times[anchor]))
  prior=state['motion'].image_prior(K@R[f].T@R[anchor]@np.linalg.inv(K),2560,1920)
  control.set('camera_person',K,R[f],f,times[f],prior,1.,0);control.strength=0
  zero=candidate();result['zero_strength_exact']=bool(torch.equal(old['model_output']['pred_masks'],zero['model_output']['pred_masks']))
  control.strength=1.5
  torch.cuda.synchronize();t=time.perf_counter();biased=candidate();torch.cuda.synchronize()
  result['biased_candidate_seconds']=time.perf_counter()-t
  result['biased_logit_max_change']=float((old['model_output']['pred_masks']-biased['model_output']['pred_masks']).abs().max())
  bias=control.bias
  result['bias_shape']=list(bias.shape);result['bias_max']=float(bias.max())
  result['query_key_variation']=bool((bias.max(-1).values-bias.min(-1).values).max()>0)
  result['finite_bias']=bool(torch.isfinite(bias).all())
  result['nonnegative_bounded']=bool(bias.min()>=0 and bias.max()<=1.5)
  nspatial=sum(int(np.prod(o['maskmem_features'].shape[-2:])) for o in control.spatial)
  result['pointer_bias_zero']=bool((bias[...,nspatial:]==0).all())
  # Same-pose projection is identity; arbitrary rotation round trip is inverse.
  from attention import project
  points=np.array([[100.,200.],[1280.,960.],[2200.,1600.]])
  H=K@R[f].T@R[anchor]@np.linalg.inv(K)
  p,ok=project(points,H);back,ok2=project(p,np.linalg.inv(H))
  result['geometry_roundtrip_px']=float(np.max(abs(back-points)))
  result['geometry_valid']=bool(ok.all() and ok2.all())
  result['peak_cuda_gb']=torch.cuda.max_memory_allocated()/1024**3
  result['parameters_trainable_but_no_optimizer']=sum(p.numel() for p in model.parameters())
  assert all(result[k] for k in ['inactive_exact','zero_strength_exact','query_key_variation','finite_bias','nonnegative_bounded','pointer_bias_zero','geometry_valid'])
  assert result['geometry_roundtrip_px']<1e-7
 write(OUT/'preflight.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
