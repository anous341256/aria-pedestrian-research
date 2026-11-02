"""Attention adapters inserted into the saved GRU/flow factorized model."""
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnorm=nn.LayerNorm(64);self.knorm=nn.LayerNorm(64)
        self.attn=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(64)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
    def forward(self,x,context):
        y=x+self.attn(self.qnorm(x),self.knorm(context),self.knorm(context),need_weights=False)[0]
        return y+self.ff(self.norm(y))

def grouped_mean(x,group):
    count=int(group.max().item())+1
    # Dense reduction avoids CUDA index_add atomic accumulation ordering noise.
    assignment=torch.nn.functional.one_hot(group,count).to(x.dtype).T
    total=assignment@x.flatten(1)
    mean=total/assignment.sum(1,keepdim=True)
    return mean.reshape((count,)+x.shape[1:])[group]

class DualAttention(nn.Module):
    def __init__(self,camera,person,cross):
        super().__init__();self.camera=camera;self.person=person;self.cross=cross
        for m in [camera,person]:
            for p in m.parameters():p.requires_grad_(False)
        self.camera_embed=nn.Linear(32,64)
        self.person_attention=Attention();self.camera_attention=Attention()
        self.person_update=nn.Linear(64,64);self.camera_update=nn.Linear(64,32)
        # Exactly reproduce the original network before the first update.
        for m in [self.person_update,self.camera_update]:
            nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def encode(self,seq,origin,cam,group):
        with torch.no_grad():
            ps=self.person.encoder(seq)[0];cs=self.camera.encoder(cam)[0]
        ce=self.camera_embed(cs)
        pcontext=ce if self.cross else ps
        # Same-time target histories provide one common camera context.
        ccontext=grouped_mean(ps,group) if self.cross else ce
        ph=ps[:,-1]+self.person_update(self.person_attention(ps,pcontext)[:,-1])
        ch=cs[:,-1]+self.camera_update(self.camera_attention(ce,ccontext)[:,-1])
        # Frozen decoder weights still propagate gradients into new adapters.
        pc=self.person.context(torch.cat([ph,origin],-1))
        rotation=self.camera.head(ch).reshape(-1,15,3)
        return pc,rotation
    def loss(self,seq,origin,cam,group,past,future,rotation,t1,t2,n1,n2,scales):
        h,r=self.encode(seq,origin,cam,group)
        lp=(self.person.future(t2[:,None]*future+(1-t2[:,None])*n2,t2,h)-future).square().mean()
        lh=(self.person.history(t1[:,None]*past+(1-t1[:,None])*n1,t1,h)-past).square().mean()
        # Give each recording/time camera state one vote, regardless of person count.
        cr=(r-rotation).square().mean(dim=(1,2))
        counts=torch.bincount(group)[group]
        lc=(cr/counts).sum()/len(torch.unique(group))
        return lp/scales['person_mse']+.5*lh/scales['person_mse']+lc/scales['camera_mse'],(lp,lh,lc)
    @torch.no_grad()
    def predict(self,seq,origin,cam,group,noise):
        h,r=self.encode(seq,origin,cam,group)
        b,k=noise.shape[:2];h=h[:,None].expand(-1,k,-1).reshape(b*k,-1)
        x=noise.reshape(b*k,-1).clone()
        for step in range(10):
            t=torch.full((b*k,),step/10,device=x.device)
            estimate=self.person.future(x,t,h)
            x=x+(estimate-x)/(1-step/10)/10
        return x.reshape(b,k,15,2).mean(1),r
