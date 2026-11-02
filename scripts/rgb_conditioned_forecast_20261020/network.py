import torch
from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1=nn.LayerNorm(64);self.n2=nn.LayerNorm(64)
        self.att=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
    def forward(self,x):
        z=self.n1(x);x=x+self.att(z,z,z,need_weights=False)[0]
        return x+self.ff(self.n2(x))
class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.person=nn.Linear(9,64);self.camera=nn.Linear(6,64)
        self.time=nn.Parameter(torch.zeros(1,16,64));nn.init.normal_(self.time,std=.02)
        self.pre=nn.Sequential(Block(),Block())
        self.visual=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,64))
        self.space=nn.Linear(4,64)
        self.modality=nn.Parameter(torch.zeros(2,64));nn.init.normal_(self.modality,std=.02)
        self.querynorm=nn.LayerNorm(64);self.keynorm=nn.LayerNorm(64)
        self.cross=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.visual_out=nn.Linear(64,64,bias=False);nn.init.zeros_(self.visual_out.weight)
        self.post=nn.Sequential(Block(),Block())
        self.head=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,64),nn.GELU(),nn.Linear(64,10))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,person,camera,local,scene,positions,arm):
        b,t=person.shape[:2]
        x=self.pre(self.person(person)+self.camera(camera)+self.time[:,:t])
        valid=torch.tensor([arm in ["local_rgb","both_rgb"]]*16+[arm in ["scene_rgb","both_rgb"]]*16,device=x.device)
        v=torch.cat([local,scene],dim=2)
        # Zero before projection AND mask after projection: disabled pixels/positions cannot leak.
        v=torch.where(valid[None,None,:,None],v,torch.zeros_like(v))
        p=torch.cat([positions,positions-person[:,:,:2,None].transpose(-1,-2)],dim=-1)
        p=torch.where(valid[None,None,:,None],p,torch.zeros_like(p))
        v=self.visual(v)+self.space(p)+self.modality.repeat_interleave(16,dim=0)[None,None]
        v=torch.cat([torch.zeros((b,t,1,64),device=x.device),v],dim=2).reshape(b*t,33,64)
        pad=torch.cat([torch.zeros(1,device=x.device,dtype=torch.bool),~valid])[None].expand(b*t,-1)
        z=self.cross(self.querynorm(x.reshape(b*t,1,64)),self.keynorm(v),self.keynorm(v),key_padding_mask=pad,need_weights=False)[0]
        x=x+self.visual_out(z.reshape(b,t,64))
        return self.head(self.post(x)[:,-1]).reshape(b,5,2)
