# 旧系统模型可读副本
# 来源：同目录 model.py。仅调整排版并补充中文说明，未修改实际训练或推理源文件。
# Python 抽象语法树已核对一致；训练和检查点继续引用原 model.py。

"""Compact independent implementation inspired by BiFlow's dual-task learning.

Shared temporal encoder + two conditional data-prediction flow decoders.
The published multi-agent/K-to-K attention network is not reproduced here.
"""
import torch
from torch import nn
import numpy as np
from common import P, F, SCALE, warp

# 相机模型：输入 [B,6,3] 历史相对旋转向量，输出 [B,15,3] 未来相对旋转。
# 单独训练；人物模型训练期间，使用它冻结后的预测作为条件。
class CameraPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(3, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, F * 3))

    def forward(self, x):
        return self.head(self.encoder(x)[1][-1]).reshape(-1, F, 3)

# 两个头都用此结构。n=12 为历史 6 个二维点，n=30 为未来 15 个二维点。
# t 是 flow 的生成时间，不是视频的真实时间；网络输出目标状态 x1。
class FlowHead(nn.Module):

    def __init__(self, n, hidden=64):
        super().__init__()
        self.n = n
        self.first = nn.Linear(n + hidden + 5, 128)
        self.middle = nn.Linear(128, 128)
        self.out = nn.Linear(128, n)
        self.modulate = nn.Linear(hidden, 256)

    # h 是共享的 64 维条件。它既与 x 拼接，也通过 gamma/beta 调制隐藏层。
    def forward(self, x, t, h):
        time = torch.stack([t, torch.sin(t * torch.pi), torch.cos(t * torch.pi), torch.sin(2 * t * torch.pi), torch.cos(2 * t * torch.pi)], -1)
        z = torch.nn.functional.silu(self.first(torch.cat([x, tensor_expand(h, x), time], -1)))
        (gamma, beta) = self.modulate(h).chunk(2, -1)
        z = z * (1 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
        return self.out(torch.nn.functional.silu(self.middle(z)))

def tensor_expand(h, x):
    return h

# 人物模型：共享编码器 + 两个并行解码头。历史重建结果不会先输入未来头。
class DualFlow(nn.Module):

    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = nn.GRU(6, hidden, batch_first=True)
        self.context = nn.Sequential(nn.Linear(hidden + 2 + P * 3 + F * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.future = FlowHead(F * 2, hidden)
        self.history = FlowHead(P * 2, hidden)

    # seq: [B,6,6]，各点为补偿后相对中心2 + 原图框宽高2 + 有效标记1 + 历史时间1。
    # origin: [B,2] 最后观测中心。camera: [B,63] 历史旋转18 + 预测未来旋转45。
    # GRU 的64维输出与其拼接成129维，再经 MLP 生成共享64维条件。
    # 因而人物预测在结构上依赖相机预测，两个模块并非独立的物理运动估计器。
    def encode(self, seq, origin, camera):
        h = self.encoder(seq)[1][-1]
        return self.context(torch.cat([h, origin, camera], -1))

    # 本轮双分支 weight=0.5。监督是归一化旋转补偿坐标中的目标状态 MSE。
    # 这里没有最终图像重投影损失，也没有通过此损失反向训练相机预测器。
    def loss(self, seq, origin, camera, past, future, t1, t2, n1, n2, weight):
        h = self.encode(seq, origin, camera)
        xt = t1[:, None] * past + (1 - t1[:, None]) * n1
        yt = t2[:, None] * future + (1 - t2[:, None]) * n2
        ph = self.history(xt, t1, h)
        pf = self.future(yt, t2, h)
        lh = (ph - past).square().mean()
        lf = (pf - future).square().mean()
        return (lf + weight * lh, lf, lh)

    # 正式未来推理 kind=future，仅运行 future 头。5组噪声、10步采样后外部取均值。
    @torch.no_grad()
    def sample(self, seq, origin, camera, noise, kind='future', steps=10):
        h = self.encode(seq, origin, camera)
        B = seq.shape[0]
        K = noise.shape[1]
        h = h[:, None].expand(-1, K, -1).reshape(B * K, -1)
        x = noise.reshape(B * K, -1).clone()
        head = getattr(self, kind)
        for j in range(steps):
            t = torch.full((B * K,), j / steps, device=x.device)
            prediction = head(x, t, h)
            x = x + (prediction - x) / (1 - j / steps) / steps
        return x.reshape(B, K, -1)

# 数据准备同时返回历史输入和监督标签；未来标签不被送入 encode/sample。
# 原图框大小仍作为输入；几何旋转只作用于中心点。
def make_features(data, indices, comp, campred, mode='reviewed', rng=None):
    idx = np.asarray(indices)
    xy = data['xy'][idx]
    obs = xy[:, :P].copy()
    valid = np.ones(obs.shape[:2], np.float32)
    if mode == 'pre_review':
        obs = data['raw'][idx].copy()
        valid = data['raw_valid'][idx].copy()
    elif mode == 'controlled_noise':
        assert rng is not None
        obs += rng.normal(0, 0.005, obs.shape) * SCALE
        valid = (rng.random(valid.shape) >= 0.2).astype(np.float32)
        valid[:, -1] = 1.0
    elif mode == 'train':
        assert rng is not None
        choice = rng.random(len(idx))
        old = (choice >= 0.4) & (choice < 0.6)
        noisy = choice >= 0.6
        obs[old] = data['raw'][idx][old]
        valid[old] = data['raw_valid'][idx][old]
        jitter = rng.normal(0, 0.005, obs.shape) * SCALE
        obs[noisy] += jitter[noisy]
        keep = (rng.random(valid.shape) >= 0.2).astype(np.float32)
        keep[:, -1] = 1.0
        valid[noisy] = keep[noisy]
    origin = obs[:, -1] / SCALE
    transformed = warp(obs, data['H'][idx, :P])[0] if comp else obs
    truth = warp(xy, data['H'][idx])[0] if comp else xy
    rel = transformed / SCALE - origin[:, None]
    rel[valid == 0] = 0.0
    wh = (data['boxes'][idx, :P, 2:] - data['boxes'][idx, :P, :2]) / SCALE
    wh = wh * valid[..., None]
    times = data['seconds'][idx, :P, None] / 0.5
    seq = np.concatenate([rel, wh, valid[..., None], times], -1)
    camera = np.concatenate([data['cam'][idx, :P].reshape(len(idx), -1), campred[idx].reshape(len(idx), -1)], -1)
    target = truth / SCALE - origin[:, None]
    return dict(seq=seq.astype(np.float32), origin=origin.astype(np.float32), camera=camera.astype(np.float32), past=target[:, :P].reshape(len(idx), -1).astype(np.float32), future=target[:, P:].reshape(len(idx), -1).astype(np.float32), input_xy=obs, valid=valid)
