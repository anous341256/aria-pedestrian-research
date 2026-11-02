"""Same separated heads; append historical translation features."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/biflow_aria_v1_20261002"))
from model import FlowHead
import torch
from torch import nn

class Camera(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(6,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,15))
        with torch.no_grad():self.encoder.weight_ih_l0[:,3:]=0
    def forward(self,x):return self.head(self.encoder(x)[1][-1]).reshape(-1,5,3)

class Person(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(9,64,batch_first=True)
        self.context=nn.Sequential(nn.Linear(66,95),nn.SiLU(),nn.Linear(95,64),nn.LayerNorm(64))
        self.future=FlowHead(10,64)
        self.history=FlowHead(32,64)
        with torch.no_grad():self.encoder.weight_ih_l0[:,6:]=0
    def encode(self,seq,origin):return self.context(torch.cat([self.encoder(seq)[1][-1],origin],-1))
    def loss(self,seq,origin,past,future,t1,t2,n1,n2):
        h=self.encode(seq,origin)
        lp=(self.history(t1[:,None]*past+(1-t1[:,None])*n1,t1,h)-past).square().mean()
        lf=(self.future(t2[:,None]*future+(1-t2[:,None])*n2,t2,h)-future).square().mean()
        return lf+.5*lp,lf,lp
    @torch.no_grad()
    def predict(self,seq,origin,noise):
        b,k=noise.shape[:2]
        h=self.encode(seq,origin)[:,None].expand(-1,k,-1).reshape(b*k,-1)
        x=noise.reshape(b*k,-1).clone()
        for step in range(10):
            t=torch.full((b*k,),step/10,device=x.device)
            estimate=self.future(x,t,h)
            x=x+(estimate-x)/(1-step/10)/10
        return x.reshape(b,k,5,2).mean(1)
