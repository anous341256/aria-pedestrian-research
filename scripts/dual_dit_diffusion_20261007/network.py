"""Small conditional DiT. Discrete VP diffusion with v-prediction and DDIM.

eta=0 is deterministic given initial noise. eta=1 is stochastic DDIM on the
same respaced grid; neither is labelled an implemented continuous SDE solver.
Normalization statistics must be fitted on training indices only.
"""
import math
import torch
from torch import nn


def time_embedding(t, width):
    freq = torch.exp(-math.log(10000) * torch.arange(width//2, device=t.device) / (width//2-1))
    angle = t[:, None] * 1000 * freq[None]
    return torch.cat([angle.cos(), angle.sin()], -1)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, width*6))
        nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, c):
        shift, scale, gate, shift2, scale2, gate2 = self.mod(c)[:, None].chunk(6, -1)
        q = self.norm1(x)*(1+scale)+shift
        x = x + gate*self.attn(q, q, q, need_weights=False)[0]
        return x + gate2*self.ff(self.norm2(x)*(1+scale2)+shift2)


class ConditionalDiT(nn.Module):
    def __init__(self, history_features, coordinates, origin_features=0, width=64, diffusion_steps=1000):
        super().__init__()
        self.coordinates = coordinates
        self.origin_features = origin_features
        self.width = width
        self.history_in = nn.Linear(history_features, width)
        self.history_pos = nn.Parameter(torch.randn(1, 6, width)*.02)
        layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout=0., batch_first=True, norm_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.origin_in = nn.Linear(origin_features, width) if origin_features else None
        self.condition = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.task = nn.Embedding(2, width)  # future / auxiliary observed-history reconstruction
        self.point_in = nn.Linear(coordinates, width)
        self.future_pos = nn.Parameter(torch.randn(1, 15, width)*.02)
        self.past_pos = nn.Parameter(torch.randn(1, 6, width)*.02)
        self.blocks = nn.ModuleList([Block(width), Block(width)])
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(width, width*2))
        self.point_out = nn.Linear(width, coordinates)
        nn.init.zeros_(self.point_out.weight); nn.init.zeros_(self.point_out.bias)
        self.register_buffer('history_mean', torch.zeros(history_features))
        self.register_buffer('history_std', torch.ones(history_features))
        self.register_buffer('origin_mean', torch.zeros(origin_features))
        self.register_buffer('origin_std', torch.ones(origin_features))
        self.register_buffer('target_mean', torch.zeros(2, coordinates))
        self.register_buffer('target_std', torch.ones(2, coordinates))
        grid = torch.linspace(0, 1, diffusion_steps+1, dtype=torch.float64)
        abar = torch.cos((grid+.008)/1.008*math.pi/2).square()
        abar = abar/abar[0]
        beta = (1-abar[1:]/abar[:-1]).clamp(max=.999)
        self.register_buffer('abar', (1-beta).cumprod(0).float())

    @torch.no_grad()
    def fit_normalization(self, history, future, origin=None, past=None):
        self.history_mean.copy_(history.mean((0, 1)))
        self.history_std.copy_(history.std((0, 1), unbiased=False).clamp_min(.01))
        if origin is not None:
            self.origin_mean.copy_(origin.mean(0))
            self.origin_std.copy_(origin.std(0, unbiased=False).clamp_min(.01))
        for task, y in enumerate([future, past]):
            if y is not None:
                self.target_mean[task].copy_(y.mean((0, 1)))
                self.target_std[task].copy_(y.std((0, 1), unbiased=False).clamp_min(.01))

    def encode(self, history, origin=None):
        if history.shape[1] != 6:
            raise ValueError('Only the six observed steps are accepted')
        h = self.history_encoder(self.history_in((history-self.history_mean)/self.history_std)+self.history_pos).mean(1)
        if self.origin_in is not None:
            h = h+self.origin_in((origin-self.origin_mean)/self.origin_std)
        return self.condition(h)

    def denoise(self, x, k, context, task=0):
        c = context+self.time(time_embedding(k.float()/len(self.abar), self.width))+self.task.weight[task]
        h = self.point_in(x)+(self.future_pos if task == 0 else self.past_pos)
        for block in self.blocks:h = block(h, c)
        shift, scale = self.final_mod(c)[:, None].chunk(2, -1)
        return self.point_out(self.norm(h)*(1+scale)+shift)

    def loss(self, history, future, origin=None, past=None, history_weight=.5):
        context = self.encode(history, origin)
        losses = []
        for task, y in enumerate([future, past]):
            if y is None:continue
            clean = (y-self.target_mean[task])/self.target_std[task]
            k = torch.randint(len(self.abar), (len(y),), device=y.device)
            a = self.abar[k].sqrt()[:, None, None]
            s = (1-self.abar[k]).sqrt()[:, None, None]
            eps = torch.randn_like(clean)
            noisy = a*clean+s*eps
            v = a*eps-s*clean
            losses.append((self.denoise(noisy, k, context, task)-v).square().mean())
        return losses[0]+(history_weight*losses[1] if len(losses)>1 else 0)

    @torch.no_grad()
    def sample(self, history, noise, origin=None, steps=20, eta=0., generator=None, task=0):
        if self.training:raise RuntimeError('Call eval() before sampling')
        if not (2 <= steps <= len(self.abar)) or not (0 <= eta <= 1):raise ValueError('Invalid sampler settings')
        batch, count, length, coord = noise.shape
        assert coord == self.coordinates and length == (15 if task == 0 else 6)
        h = self.encode(history, origin)[:, None].expand(-1, count, -1).reshape(batch*count, -1)
        x = noise.reshape(batch*count, length, coord).clone()
        grid = torch.linspace(len(self.abar)-1, 0, steps, device=x.device).round().long()
        for j, k in enumerate(grid):
            at = self.abar[k]
            ap = self.abar[grid[j+1]] if j+1 < len(grid) else x.new_tensor(1.)
            v = self.denoise(x, k.expand(len(x)), h, task)
            x0 = at.sqrt()*x-(1-at).sqrt()*v
            eps = (1-at).sqrt()*x+at.sqrt()*v
            variance = eta**2*((1-ap)/(1-at))*(1-at/ap)
            x = ap.sqrt()*x0+(1-ap-variance).clamp_min(0).sqrt()*eps
            if eta and j+1 < len(grid):
                x = x+variance.clamp_min(0).sqrt()*torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        y = x.reshape(batch, count, length, coord)
        return y*self.target_std[task]+self.target_mean[task]
