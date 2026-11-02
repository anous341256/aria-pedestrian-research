"""Historical three-stream self/cross attention, one image trajectory output."""
import torch
from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnorm=nn.LayerNorm(64);self.knorm=nn.LayerNorm(64)
        self.attention=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.norm=nn.LayerNorm(64)
        self.ff=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
    def forward(self,q,context,padding):
        x=q+self.attention(self.qnorm(q),self.knorm(context),self.knorm(context),key_padding_mask=padding,need_weights=False)[0]
        return x+self.ff(self.norm(x))

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs=nn.ModuleList([nn.Linear(n,64) for n in [6,4,3]])
        self.summary=nn.Parameter(torch.zeros(3,1,64))
        nn.init.normal_(self.summary,std=.02)
        self.self_blocks=nn.ModuleList([nn.ModuleList([AttentionBlock(),AttentionBlock()]) for _ in range(3)])
        self.cross_blocks=nn.ModuleList([AttentionBlock() for _ in range(3)])
        self.condition=nn.Linear(6,64)
        self.decoder=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,128),nn.GELU(),nn.Linear(128,30))
    def forward(self,person,camera,gaze,gaze_valid,condition,use_gaze=True):
        if not use_gaze:
            # Remove values AND timestamps/validity. Learned content-free summary remains.
            gaze=torch.zeros_like(gaze);gaze_valid=torch.zeros_like(gaze_valid)
        streams=[];pads=[]
        for i,x in enumerate([person,camera,gaze]):
            b=x.shape[0]
            valid=gaze_valid if i==2 else torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
            pad=torch.cat([torch.zeros((b,1),dtype=torch.bool,device=x.device),~valid],dim=1)
            x=torch.cat([self.summary[i:i+1].expand(b,-1,-1),self.inputs[i](x)],dim=1)
            for block in self.self_blocks[i]:x=block(x,x,pad)
            streams.append(x);pads.append(pad)
        # All three cross updates read the same pre-cross state (no order dependence).
        fused=[]
        for i in range(3):
            context=torch.cat([streams[j] for j in range(3) if j!=i],dim=1)
            padding=torch.cat([pads[j] for j in range(3) if j!=i],dim=1)
            fused.append(self.cross_blocks[i](streams[i],context,padding)[:,0])
        return self.decoder(torch.cat(fused+[self.condition(condition)],dim=-1))
