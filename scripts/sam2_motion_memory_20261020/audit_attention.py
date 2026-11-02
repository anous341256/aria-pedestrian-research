"""Independent small-grid NumPy attention-bias check and checkpoint integrity."""
import math,types,json
import numpy as np,torch
from attention import MotionMemory,metadata,project
from selection import OUT,write
def main():
 side=8;W,H=2560,1920
 K=np.array([[1500,0,1280],[0,1500,960],[0,0,1.]],float)
 theta=.04
 Rs=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
 mask=np.zeros((192,256),bool);mask[60:150,80:140]=True
 box=np.array([800.,600.,1400.,1500.])
 meta=metadata(mask,box,Rs,1.,.8,side)
 out=dict(maskmem_features=torch.zeros(1,64,side,side),motion_meta=meta)
 ctl=object.__new__(MotionMemory);ctl.strength=1.5;ctl.spatial=[out];ctl.max_bias=0;ctl.max_elements=0
 prior=dict(projection_valid=True,box=box+np.array([40.,-20.,40.,-20.]),sigma_xy=np.array([20.,30.]))
 checks={}
 for mode in ['camera','camera_person']:
  ctl.active=dict(mode=mode,K=K,R=np.eye(3),frame=2,time=1.1,prior=prior,identity=.9,missed=1)
  got=ctl.make_bias(side,68,64,'cpu',torch.float32)[0,0].numpy()
  expected=np.zeros((64,68));h=900.;w=600.;dt=.1
  sigma=np.array([max(24,.18*w),max(24,.10*h)])
  reliability=.9*.8*math.exp(-dt/1.5)*math.exp(-1/3)
  hmat=K@Rs@np.linalg.inv(K)
  centre_h=hmat@np.r_[(box[:2]+box[2:])/2,1.]
  centre=centre_h[:2]/centre_h[2]
  delta=np.zeros(2)
  if mode=='camera_person':
   sigma=np.sqrt(sigma**2+prior['sigma_xy']**2)
   reliability/=1+(30/max(32,.25*h))**2
   delta=(prior['box'][:2]+prior['box'][2:])/2-centre
  else:
   sigma=np.sqrt(sigma**2+(dt*.3*h)**2)
   reliability/=1+(dt/.75)**2
  reliability*=min(1.,(max(32,.35*h)/max(sigma))**2)
  for j in range(64):
   support=meta['support'].ravel()[j]
   if support<.15:continue
   v=np.array([(j%8+.5)*W/8,(j//8+.5)*H/8,1.])
   p=hmat@v
   if p[2]<=1e-7:continue
   target=p[:2]/p[2]+delta
   for i in range(64):
    query=np.array([(i%8+.5)*W/8,(i//8+.5)*H/8])
    distance=((query-target)/sigma)**2
    expected[i,j]=1.5*reliability*support*math.exp(-.5*sum(distance))
  error=float(np.abs(got-expected).max())
  assert error<1e-6,error
  checks[mode+'_max_error']=error
  assert np.all(got[:,64:]==0)
  assert np.all(got[:,np.flatnonzero(meta['support'].ravel()<.15)]==0)
 checks['pointer_and_background_bias_zero']=True
 # A query-only constant indeed cancels; the implemented pairwise bias does not.
 rng=np.random.default_rng(17);logits=rng.normal(size=(64,68))
 def sm(x):
  e=np.exp(x-x.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
 checks['query_constant_cancellation_error']=float(abs(sm(logits)-sm(logits+rng.normal(size=(64,1)))).max())
 checks['pairwise_attention_change']=float(abs(sm(logits+got)-sm(logits)).max())
 assert checks['pairwise_attention_change']>0
 write(OUT/'independent_attention_audit.json',checks);print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
