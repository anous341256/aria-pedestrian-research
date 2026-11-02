"""Historical depth enriches person tokens; two separately supervised heads remain."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/dual_cross_attention_20261008'))
from cross_model import DualAttention,grouped_mean,torch,nn

class DepthAttention(DualAttention):
    def __init__(self,camera,person,with_depth):
        super().__init__(camera,person,True);self.with_depth=with_depth
        self.depth_embed=nn.Linear(5,64,bias=False);nn.init.zeros_(self.depth_embed.weight)
    def encode(self,seq,origin,cam,group,depth):
        with torch.no_grad():ps=self.person.encoder(seq)[0];cs=self.camera.encoder(cam)[0]
        if not self.with_depth:depth=torch.zeros_like(depth)
        enriched=ps+self.depth_embed(depth)
        ce=self.camera_embed(cs)
        ph=ps[:,-1]+self.person_update(self.person_attention(enriched,ce)[:,-1])
        ch=cs[:,-1]+self.camera_update(self.camera_attention(ce,grouped_mean(enriched,group))[:,-1])
        pc=self.person.context(torch.cat([ph,origin],-1));r=self.camera.head(ch).reshape(-1,15,3)
        return pc,r
    def loss(self,seq,origin,cam,group,depth,past,future,rotation,t1,t2,n1,n2,scales):
        h,r=self.encode(seq,origin,cam,group,depth)
        lp=(self.person.future(t2[:,None]*future+(1-t2[:,None])*n2,t2,h)-future).square().mean()
        lh=(self.person.history(t1[:,None]*past+(1-t1[:,None])*n1,t1,h)-past).square().mean()
        cr=(r-rotation).square().mean(dim=(1,2));counts=torch.bincount(group)[group]
        lc=(cr/counts).sum()/len(torch.unique(group))
        return (lp+.5*lh)/scales['person_mse']+lc/scales['camera_mse'],(lp,lh,lc)
    @torch.no_grad()
    def predict(self,seq,origin,cam,group,depth,noise):
        h,r=self.encode(seq,origin,cam,group,depth);b,k=noise.shape[:2]
        h=h[:,None].expand(-1,k,-1).reshape(b*k,-1);x=noise.reshape(b*k,-1).clone()
        for step in range(10):
            t=torch.full((b*k,),step/10,device=x.device);estimate=self.person.future(x,t,h)
            x=x+(estimate-x)/(1-step/10)/10
        return x.reshape(b,k,15,2).mean(1),r
