import sys
from geometry_common import ROOT
sys.path.insert(0,str(ROOT/'scripts/biflow_aria_v1_20261002'))
from model import FlowHead
import torch
from torch import nn
def camera_velocity(x,seconds):
 t=seconds-seconds.mean(1,keepdim=True)
 return (t[:,:,None]*x[:,:,3:]).sum(1)/(t.square().sum(1,keepdim=True).clamp_min(1e-8))
class Camera(nn.Module):
 def __init__(self,old_state):
  super().__init__();self.encoder=nn.GRU(6,32,batch_first=True);self.head=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,15));self.load_state_dict(old_state)
  self.position=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,15));nn.init.zeros_(self.position[-1].weight);nn.init.zeros_(self.position[-1].bias)
 def forward(self,x,seconds):
  h=self.encoder(x)[1][-1];r=self.head(h).reshape(-1,5,3);v=camera_velocity(x,seconds);times=torch.arange(1,6,device=x.device,dtype=x.dtype)*.1
  b=v[:,None]*times[None,:,None]+self.position(h).reshape(-1,5,3)
  return r,b
class Person(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=nn.GRU(10,64,batch_first=True);self.context=nn.Sequential(nn.Linear(68,95),nn.SiLU(),nn.Linear(95,64),nn.LayerNorm(64));self.future=FlowHead(15,64);self.history=FlowHead(48,64)
  for head in [self.future,self.history]:nn.init.zeros_(head.out.weight);nn.init.zeros_(head.out.bias)
 def encode(self,x,anchor):return self.context(torch.cat([self.encoder(x)[1][-1],anchor],-1))
 def loss(self,x,anchor,past,future,pmask,fmask,t1,t2,n1,n2):
  h=self.encode(x,anchor);ph=self.history(t1[:,None]*past+(1-t1[:,None])*n1,t1,h);pf=self.future(t2[:,None]*future+(1-t2[:,None])*n2,t2,h)
  lh=((ph-past).square()*pmask).sum()/pmask.sum().clamp_min(1);lf=((pf-future).square()*fmask).sum()/fmask.sum().clamp_min(1)
  return lf+.5*lh
 @torch.no_grad()
 def predict(self,x,anchor,noise):
  b,k=noise.shape[:2];h=self.encode(x,anchor)[:,None].expand(-1,k,-1).reshape(b*k,-1);state=noise.reshape(b*k,-1).clone()
  for step in range(10):
   t=torch.full((b*k,),step/10,device=state.device);estimate=self.future(state,t,h);state+=(estimate-state)/(1-step/10)/10
  return state.reshape(b,k,5,3).mean(1)
