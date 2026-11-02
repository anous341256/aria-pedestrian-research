"""Small paired-mode adapters over frozen, previously fitted GRU/flow models."""
import torch
from torch import nn


def mlp(n, out):
    return nn.Sequential(nn.Linear(n, 32), nn.SiLU(), nn.Linear(32, out))


class JointModes(nn.Module):
    def __init__(self, modes, anchors, camera_std, person_std):
        super().__init__()
        self.modes = modes
        self.mode = nn.Parameter(torch.randn(modes, 8) * .1)
        self.camera = mlp(40, 45)
        self.camera_score = mlp(32, modes)
        self.person = mlp(64 + 32 + 8 + 45, 30)
        self.score = mlp(64 + 32 + 8 + 45 + 30, 1)
        self.register_buffer('anchors', torch.as_tensor(anchors, dtype=torch.float32))
        self.register_buffer('camera_std', torch.as_tensor(camera_std, dtype=torch.float32))
        self.register_buffer('person_std', torch.as_tensor(person_std, dtype=torch.float32))
        for head in [self.camera, self.person, self.score, self.camera_score]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def camera_candidates(self, feature, base):
        n = len(feature)
        c = feature[:, None].expand(-1, self.modes, -1)
        m = self.mode[None].expand(n, -1, -1)
        offset = self.anchors[None] + self.camera(torch.cat([c, m], -1)) * self.camera_std
        return base[:, None] + offset.reshape(n, self.modes, 15, 3), self.camera_score(feature)

    def forward(self, camera_feature, base_camera, person_feature, base_person):
        rotations, prior = self.camera_candidates(camera_feature, base_camera)
        n = len(camera_feature)
        common = torch.cat([person_feature[:, None].expand(-1, self.modes, -1),
                            camera_feature[:, None].expand(-1, self.modes, -1),
                            self.mode[None].expand(n, -1, -1),
                            rotations.flatten(2) / self.camera_std], -1)
        residual = base_person[:, None] + (self.person(common) * self.person_std).reshape(n, self.modes, 15, 2)
        logits = prior + self.score(torch.cat([common, residual.flatten(2) / self.person_std], -1)).squeeze(-1)
        return rotations, residual, logits


def rotation_matrix(v):
    # Stable differentiable Rodrigues formula, including exactly zero rotations.
    theta = torch.linalg.vector_norm(v, dim=-1)
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    skew = torch.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(*v.shape[:-1], 3, 3)
    a = torch.sinc(theta / torch.pi)[..., None, None]
    b = (.5 * torch.sinc(theta / (2 * torch.pi)).square())[..., None, None]
    return torch.eye(3, device=v.device, dtype=v.dtype) + a * skew + b * (skew @ skew)


def angle_error(pred, true):
    a, b = rotation_matrix(pred), rotation_matrix(true)
    # Frobenius-distance form of the SO(3) geodesic angle (radians).
    chord = torch.linalg.matrix_norm(a - b, ord='fro', dim=(-2, -1)) / (2 ** 1.5)
    return 2 * torch.asin(chord.clamp(max=1 - 1e-7))


def project(points, rotations, K):
    R = rotation_matrix(rotations).transpose(-1, -2)
    H = K[:, None, None] @ R @ torch.linalg.inv(K)[:, None, None]
    p = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    q = (H @ p[..., None]).squeeze(-1)
    depth = q[..., 2]
    # Invalid rays are additionally penalized; this only makes the loss finite.
    xy = q[..., :2] / depth.clamp(min=.02)[..., None]
    return xy, depth


def group_average(values, groups):
    n = int(groups.max().item()) + 1
    sums = torch.zeros(n, values.shape[1], device=values.device, dtype=values.dtype)
    sums.index_add_(0, groups, values)
    count = torch.bincount(groups, minlength=n).to(values.dtype)[:, None]
    return sums / count


def joint_losses(rotations, residual, logits, origin, K, true_rotation, true_residual,
                 true_image, groups, scales, temperature=.5):
    scale = residual.new_tensor([2560., 1920.])
    component = (residual + origin[:, None, None]) * scale
    image, depth = project(component, rotations, K)
    er = angle_error(rotations, true_rotation[:, None]).mean(-1)
    ep = torch.linalg.vector_norm((residual - true_residual[:, None]) * scale, dim=-1).mean(-1)
    eu = torch.linalg.vector_norm(image - true_image[:, None], dim=-1).mean(-1)
    eu = eu + 32000 * torch.relu(.02 - depth).mean(-1) / .02
    error = (er / scales[0] + ep / scales[1] + eu / scales[2]) / 3
    ge = group_average(error, groups)
    gl = group_average(logits, groups)
    winner = ge.detach().argmin(-1)
    target = torch.softmax(-ge.detach() / temperature, -1)
    regression = ge.gather(1, winner[:, None]).mean()
    classification = -(target * torch.log_softmax(gl, -1)).sum(-1).mean()
    return regression + .1 * classification, dict(error=error, er=er, ep=ep, eu=eu,
             image=image, depth=depth, component=component, group_error=ge,
             group_logits=gl, winner=winner, target=target)
