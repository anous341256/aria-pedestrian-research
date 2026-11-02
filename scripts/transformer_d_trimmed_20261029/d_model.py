import math,torch
from torch import nn

class Block(nn.Module):
 def __init__(self,ff=55):
  super().__init__();self.query_norm=nn.LayerNorm(64);self.context_norm=nn.LayerNorm(64);self.attn=nn.MultiheadAttention(64,4,dropout=0,batch_first=True);self.ff_norm=nn.LayerNorm(64);self.ff=nn.Sequential(nn.Linear(64,ff),nn.GELU(),nn.Linear(ff,64))
 def forward(self,x,context=None):
  q=self.query_norm(x);kv=self.context_norm(x if context is None else context);x=x+self.attn(q,kv,kv,need_weights=False)[0];return x+self.ff(self.ff_norm(x))

class DModel(nn.Module):
 def __init__(self):
  super().__init__();self.condition=nn.Linear(6,64);pos=torch.arange(16)[:,None];freq=torch.exp(torch.arange(0,64,2)*(-math.log(10000)/64));pe=torch.zeros(16,64);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq);self.register_buffer('position',pe)
  self.input=nn.Linear(6,64);self.person=nn.ModuleList([Block() for _ in range(2)]);self.camera_input=nn.Linear(38,64);self.camera=nn.ModuleList([Block() for _ in range(2)]);self.person_reads_camera=Block();self.camera_reads_person=Block();self.fusion=nn.Linear(128,64);self.output_norm=nn.LayerNorm(64);self.decoder=nn.Sequential(nn.Linear(16*64,128),nn.GELU(),nn.Linear(128,10))
 def forward(self,seq,condition,camera):
  x=self.input(seq)+self.position[None]+self.condition(condition)[:,None];c=self.camera_input(camera)+self.position[None]
  for block in self.person:x=block(x)
  for block in self.camera:c=block(c)
  px=self.person_reads_camera(x,c);cx=self.camera_reads_person(c,x);x=self.fusion(torch.cat([px,cx],-1));return self.decoder(self.output_norm(x).flatten(1)).reshape(-1,5,2)
