"""Instance-local bounded geometric memory-attention bias; no vendor edits."""
import math, types, inspect
import numpy as np
import torch
from torch.nn import functional as F
from sam2.modeling.sam.transformer import apply_rotary_enc
from sam2.modeling.sam2_utils import select_closest_cond_frames

def project(p,H):
 r=np.c_[p,np.ones(len(p))]@H.T
 ok=np.isfinite(r).all(1)&(r[:,2]>1e-7)
 xy=np.zeros((len(p),2))
 xy[ok]=r[ok,:2]/r[ok,2:]
 return xy,ok

class MotionMemory:
 def __init__(self,model,strength=1.5):
  self.model=model;self.strength=strength;self.active=None;self.bias=None
  self.calls=0;self.biased_calls=0;self.max_bias=0.;self.max_elements=0
  original=model._prepare_memory_conditioned_features
  sig=inspect.signature(original)
  control=self
  def prepare(inst,*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();a=bound.arguments
   control.bias=None;control.spatial=[]
   if control.active is not None and not a['is_init_cond_frame']:
    memory=a['output_dict'];idx=a['frame_idx']
    assert not a.get('track_in_reverse',False),'Traversal is reindexed in the runner.'
    selected,unselected=select_closest_cond_frames(idx,memory['cond_frame_outputs'],inst.max_cond_frames_in_attn)
    outputs=list(selected.values())
    stride=inst.memory_temporal_stride_for_eval
    for pos in range(1,inst.num_maskmem):
     rel=inst.num_maskmem-pos
     f=idx-1 if rel==1 else ((idx-2)//stride)*stride-(rel-2)*stride
     out=memory['non_cond_frame_outputs'].get(f,unselected.get(f))
     if out is not None:outputs.append(out)
    control.spatial=outputs
   return original(*args,**kwargs)
  model._prepare_memory_conditioned_features=types.MethodType(prepare,model)
  for layer in model.memory_attention.layers:
   module=layer.cross_attn_image;old=module.forward
   def forward(inst,q,k,v,num_k_exclude_rope=0,_old=old):
    control.calls+=1
    if control.active is None or control.strength==0:
     return _old(q,k,v,num_k_exclude_rope=num_k_exclude_rope)
    q=inst._separate_heads(inst.q_proj(q),inst.num_heads)
    k=inst._separate_heads(inst.k_proj(k),inst.num_heads)
    v=inst._separate_heads(inst.v_proj(v),inst.num_heads)
    side=math.isqrt(q.shape[-2]);assert side*side==q.shape[-2]
    inst.freqs_cis=inst.freqs_cis.to(q.device)
    if inst.freqs_cis.shape[0]!=q.shape[-2]:
     inst.freqs_cis=inst.compute_cis(end_x=side,end_y=side).to(q.device)
    n=k.size(-2)-num_k_exclude_rope
    q,k[:,:,:n]=apply_rotary_enc(q,k[:,:,:n],freqs_cis=inst.freqs_cis,repeat_freqs_k=inst.rope_k_repeat)
    if control.bias is None:
     control.bias=control.make_bias(side,k.shape[-2],n,q.device,q.dtype)
    assert control.bias.shape[-2:]==(q.shape[-2],k.shape[-2])
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=control.bias,
       dropout_p=inst.dropout_p if inst.training else 0.)
    control.biased_calls+=1
    return inst.out_proj(inst._recombine_heads(out))
   module.forward=types.MethodType(forward,module)

 def set(self,mode,K,R,frame,time,prior,identity_reliability,missed):
  self.active=None;self.bias=None
  if mode not in ('camera','camera_person'):return
  self.active=dict(mode=mode,K=K,R=R,frame=frame,time=time,prior=prior,
       identity=float(np.clip(identity_reliability,0,1)),missed=missed)

 def make_bias(self,side,nkeys,nspatial,device,dtype):
  a=self.active;K=a['K'];R=a['R'];P=a['prior'];width,height=2560,1920
  axis=(np.arange(side)+.5)/side
  xx,yy=np.meshgrid(axis*width,axis*height);xy=np.stack([xx.ravel(),yy.ravel()],1)
  columns=[];coords=[];weights=[];scales=[];offset=0
  for out in self.spatial:
   count=int(np.prod(out['maskmem_features'].shape[-2:]));assert count==side*side
   meta=out.get('motion_meta')
   if meta:
    support=np.asarray(meta['support']).reshape(-1)
    inds=np.flatnonzero(support>=.15)
    H=K@R.T@meta['R']@np.linalg.inv(K)
    predicted,ok=project(xy[inds],H)
    dt=abs(a['time']-meta['time'])
    w,h=np.maximum(np.asarray(meta['box'])[2:]-np.asarray(meta['box'])[:2],2)
    uncertainty=np.array([max(24,.18*w),max(24,.10*h)])
    reliability=a['identity']*meta['reliability']*math.exp(-dt/1.5)*math.exp(-a['missed']/3)
    if a['mode']=='camera_person' and P['projection_valid'] and P['box'] is not None:
     centre,cv=project(np.asarray([(np.asarray(meta['box'])[:2]+np.asarray(meta['box'])[2:])/2]),H)
     sig=np.asarray(P['sigma_xy'])
     uncertainty=np.sqrt(uncertainty**2+sig**2)
     reliability*=1/(1+(float(np.max(sig))/max(32,.25*h))**2)
     if cv[0]:
      predicted+=(np.asarray(P['box'])[:2]+np.asarray(P['box'])[2:])/2-centre[0]
     else:reliability=0.
    else:
     # Rotation alone is uncertain about scene-relative target displacement.
     uncertainty=np.sqrt(uncertainty**2+(dt*.3*h)**2)
     reliability*=1/(1+(dt/.75)**2)
    reliability*=min(1.,(max(32,.35*h)/max(uncertainty))**2)
    valid=ok&np.isfinite(predicted).all(1)
    if reliability>1e-4 and valid.any():
     columns.extend((offset+inds[valid]).tolist());coords.extend(predicted[valid].tolist())
     weights.extend((self.strength*reliability*support[inds[valid]]).tolist())
     scales.extend([uncertainty.tolist()]*int(valid.sum()))
   offset+=count
  assert offset==nspatial,(offset,nspatial)
  bias=torch.zeros((side*side,nkeys),device=device,dtype=dtype)
  if columns:
   query=torch.as_tensor(xy,device=device,dtype=torch.float32)
   target=torch.as_tensor(coords,device=device,dtype=torch.float32)
   sigma=torch.as_tensor(scales,device=device,dtype=torch.float32)
   wt=torch.as_tensor(weights,device=device,dtype=torch.float32)
   cols=torch.as_tensor(columns,device=device,dtype=torch.long)
   for first in range(0,len(columns),512):
    last=first+512
    d=(query[:,None,:]-target[None,first:last,:])/sigma[None,first:last,:]
    vals=torch.exp(-.5*(d*d).sum(-1))*wt[None,first:last]
    bias[:,cols[first:last]]=vals.to(dtype)
   self.max_bias=max(self.max_bias,float(max(weights)))
  self.max_elements=max(self.max_elements,bias.numel())
  return bias[None,None]

def metadata(mask,box,R,time,reliability,side=64):
 support=F.adaptive_avg_pool2d(torch.as_tensor(mask.astype(np.float32))[None,None],(side,side))[0,0].numpy()
 return dict(support=support,box=np.asarray(box),R=np.asarray(R),time=float(time),reliability=float(reliability))
