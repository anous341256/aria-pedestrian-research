"""Analytical sampler checks and input contracts, not forecast performance tests."""
import types
import torch
from network import ConditionalDiT
torch.set_num_threads(2);torch.manual_seed(17)
for coords,features,origin in [(3,3,0),(2,6,2)]:
 m=ConditionalDiT(features,coords,origin_features=origin)
 x=torch.randn(4,6,features);o=torch.randn(4,origin) if origin else None
 y=torch.randn(4,15,coords);past=torch.randn(4,6,coords) if origin else None
 m.fit_normalization(x,y,o,past)
 loss=m.loss(x,y,o,past);loss.backward()
 assert torch.isfinite(loss) and m.point_out.weight.grad.abs().sum()>0
 m.eval();noise=torch.randn(4,5,15,coords)
 a=m.sample(x,noise,o);b=m.sample(x,noise,o)
 assert torch.equal(a,b) and torch.isfinite(a).all()
 s=m.sample(x,noise,o,eta=1,generator=torch.Generator().manual_seed(7))
 s2=m.sample(x,noise,o,eta=1,generator=torch.Generator().manual_seed(7))
 assert torch.equal(s,s2) and not torch.equal(a,s)
 try:m.sample(torch.randn(4,21,features),noise,o);raise AssertionError('Future input accepted')
 except ValueError:pass
 # With an oracle v field, either sampler must recover the same clean target.
 clean=torch.randn(1,15,coords)
 def oracle(self,z,k,context,task=0):
  a=self.abar[k].sqrt()[:,None,None];sig=(1-self.abar[k]).sqrt()[:,None,None]
  return (a*z-clean)/sig
 m.denoise=types.MethodType(oracle,m)
 for eta in [0.,1.]:
  p=m.sample(x,noise,o,eta=eta,generator=torch.Generator().manual_seed(9))
  expected=clean*m.target_std[0]+m.target_mean[0]
  assert torch.allclose(p,expected,atol=2e-5),float((p-expected).abs().max())
 print('Passed coordinates',coords,'parameters',sum(p.numel() for p in m.parameters()))
print('All sampler, gradient, replay and history-only interface checks passed.')
