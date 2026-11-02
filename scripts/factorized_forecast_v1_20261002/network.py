"""Independent pedestrian encoder and dual conditional-flow heads.

The separated network has no camera input. Its only camera dependency is the
historical geometric normalization applied before this module.
"""
from shared import *
import torch
from torch import nn
from model import FlowHead


class PedestrianPredictor(nn.Module):
    def __init__(self, coupled=False):
        super().__init__()
        self.coupled = coupled
        self.encoder = nn.GRU(6, 64, batch_first=True)
        # 95 versus 64 matches total trainable capacity to within 29 parameters.
        middle = 64 if coupled else 95
        input_dim = 64 + 2 + (P * 3 + F * 3 if coupled else 0)
        self.context = nn.Sequential(nn.Linear(input_dim, middle), nn.SiLU(), nn.Linear(middle, 64), nn.LayerNorm(64))
        self.future = FlowHead(F * 2, 64)
        self.history = FlowHead(P * 2, 64)

    def encode(self, seq, origin, camera=None):
        h = self.encoder(seq)[1][-1]
        if self.coupled:
            if camera is None: raise ValueError('Coupled control requires camera condition')
            values = [h, origin, camera]
        else:
            if camera is not None: raise ValueError('Separated pedestrian branch must not receive camera predictions')
            values = [h, origin]
        return self.context(torch.cat(values, -1))

    def loss(self, seq, origin, past, future, t1, t2, n1, n2, weight, camera=None):
        h = self.encode(seq, origin, camera)
        past_mix = t1[:, None] * past + (1 - t1[:, None]) * n1
        future_mix = t2[:, None] * future + (1 - t2[:, None]) * n2
        lh = (self.history(past_mix, t1, h) - past).square().mean()
        lf = (self.future(future_mix, t2, h) - future).square().mean()
        return lf + weight * lh, lf, lh

    @torch.no_grad()
    def sample(self, seq, origin, noise, camera=None, kind='future', steps=10):
        h = self.encode(seq, origin, camera)
        batch, count = noise.shape[:2]
        h = h[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        x = noise.reshape(batch * count, -1).clone()
        head = getattr(self, kind)
        for j in range(steps):
            t = torch.full((batch * count,), j / steps, device=x.device)
            estimate = head(x, t, h)
            x = x + (estimate - x) / (1 - j / steps) / steps
        return x.reshape(batch, count, -1)
