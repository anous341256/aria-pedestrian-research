"""Measure zero dense-bias backend differences separately from motion effects."""
import copy,hashlib,json,types
import run as task
from run import torch,np,near,ROOT,OUT,read,write,MotionMemory
def weight_hash(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  h.update(name.encode());h.update(p.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 result={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  tracker=near.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
  model=tracker._predictor;model.eval();before=weight_hash(model)
  c=read(OUT/'selection.json')[0];cal,cached=task.load_frames(c,tracker)
  e=c['entries'][0];anchor=e['anchor_frame'];f=anchor-1 if anchor>e['start'] else anchor+1
  pp,feat,hsv=cached[anchor];seed,_,_=task.image_candidate(model,pp,e['anchor_box'])
  state=near.init_state(tracker,pp,seed,feat,hsv,0,0.,np.eye(3))
  task.add_meta(state,0,seed,near.describe_mask(seed)['bbox_xyxy'],cal['R'][anchor],cal['times_ns'][anchor]/1e9,1.)
  base=copy.deepcopy(state['memory'])
  ppf,ff,hf=cached[f];control=MotionMemory(model)
  original_bias=control.make_bias
  def zero_bias(side,nkeys,nspatial,device,dtype):
   return torch.zeros((1,1,side*side,nkeys),device=device,dtype=dtype)
  for history in [1,7]:
   state['memory']=copy.deepcopy(base)
   for t in range(1,history):state['memory']['non_cond_frame_outputs'][t]=copy.deepcopy(base['cond_frame_outputs'][0])
   def predict():
    return near.inc.base.make_candidate(model,ppf,state,history,e['id'],'memory',None,ff,hf)
   control.active=None
   original=predict()
   control.set('camera',cal['K'],cal['R'][f],f,cal['times_ns'][f]/1e9,{},1.,0)
   control.make_bias=zero_bias
   zero=predict()
   control.make_bias=original_bias
   biased=predict()
   pixel_count=1920*2560
   result[str(history)+'_memory_frames']=dict(
     zero_dense_logit_max=float((original['model_output']['pred_masks']-zero['model_output']['pred_masks']).abs().max()),
     zero_dense_changed_mask_pixels=int(np.count_nonzero(original['mask']!=zero['mask'])),
     zero_dense_changed_mask_fraction=float(np.count_nonzero(original['mask']!=zero['mask'])/pixel_count),
     motion_vs_zero_dense_logit_max=float((biased['model_output']['pred_masks']-zero['model_output']['pred_masks']).abs().max()),
     motion_vs_zero_dense_changed_mask_pixels=int(np.count_nonzero(biased['mask']!=zero['mask'])))
  result['weights_unchanged']=weight_hash(model)==before
  result['parameter_sha256']=before
  assert result['weights_unchanged']
 write(OUT/'numeric_backend_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
