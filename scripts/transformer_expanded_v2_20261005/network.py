"""Matched-width self/cross-attention, historical tokens only, common decoder."""
import math
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, ff):
        super().__init__()
        self.query_norm=nn.LayerNorm(64); self.context_norm=nn.LayerNorm(64)
        self.attn=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.ff_norm=nn.LayerNorm(64)
        self.ff=nn.Sequential(nn.Linear(64,ff),nn.GELU(),nn.Linear(ff,64))

    def forward(self,x,context=None):
        # No causal mask needed: all six tokens are observed before the cutoff.
        q=self.query_norm(x); kv=self.context_norm(x if context is None else context)
        x=x+self.attn(q,kv,kv,need_weights=False)[0]
        return x+self.ff(self.ff_norm(x))


class Predictor(nn.Module):
    def __init__(self,variant,ff):
        super().__init__(); self.variant=variant
        self.geometry=variant[0] in 'CF'; self.uses_camera_features=variant[0] in 'ADEF'
        self.condition=nn.Linear(6,64)
        position=torch.arange(6)[:,None]; freq=torch.exp(torch.arange(0,64,2)*(-math.log(10000)/64))
        pe=torch.zeros(6,64); pe[:,0::2]=torch.sin(position*freq); pe[:,1::2]=torch.cos(position*freq)
        self.register_buffer('position',pe)
        if variant[0]=='A':
            self.input=nn.Linear(6+35,64); self.joint=nn.ModuleList([Block(ff) for _ in range(2)])
        else:
            self.input=nn.Linear(6,64); self.person=nn.ModuleList([Block(ff) for _ in range(2)])
            if self.uses_camera_features:
                self.camera_input=nn.Linear(35,64); self.camera=nn.ModuleList([Block(ff) for _ in range(2)])
                self.person_reads_camera=Block(ff)
                if variant[0]=='D':
                    self.camera_reads_person=Block(ff); self.fusion=nn.Linear(128,64)
        self.output_norm=nn.LayerNorm(64)
        self.decoder=nn.Sequential(nn.Linear(6*64,128),nn.GELU(),nn.Linear(128,30))

    def forward(self,seq,condition,camera=None):
        base=self.position[None]+self.condition(condition)[:,None]
        if self.variant[0]=='A':
            assert camera is not None
            x=self.input(torch.cat([seq,camera],-1))+base
            for block in self.joint:x=block(x)
        else:
            x=self.input(seq)+base
            for block in self.person:x=block(x)
            if self.uses_camera_features:
                assert camera is not None
                # Camera tokens receive calibration but NOT pedestrian origin.
                # F's actual camera forecast is a separate frozen GRU.
                c=self.camera_input(camera)+self.position[None]
                for block in self.camera:c=block(c)
                px=self.person_reads_camera(x,c)
                if self.variant[0]=='D':
                    cx=self.camera_reads_person(c,x)
                    x=self.fusion(torch.cat([px,cx],-1))
                else:x=px
            else:assert camera is None,'B/C must not receive camera feature tokens'
        return self.decoder(self.output_norm(x).flatten(1))


def parameter_count(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)


def choose_width(variant,budget=210000):
    # Parameter count is affine in FF width. This never consults any scores/data.
    low=parameter_count(Predictor(variant,1)); slope=parameter_count(Predictor(variant,2))-low
    width=max(16,round((budget-low)/slope+1))
    model=Predictor(variant,width)
    return dict(ff_width=width,trainable_parameters=parameter_count(model))
