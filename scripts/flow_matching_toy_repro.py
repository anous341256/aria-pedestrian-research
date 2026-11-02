import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances


class VectorField(nn.Module):
    def __init__(self, hidden=128):
        super(VectorField, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, t], dim=1))


def sample_data(n, noise=0.06):
    x, _ = make_moons(n_samples=n, noise=noise)
    x = x.astype("float32")
    x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
    return x


def cfm_batch(data, batch_size, device):
    ids = np.random.randint(0, len(data), size=batch_size)
    x1 = torch.from_numpy(data[ids]).to(device)
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, 1, device=device)
    xt = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0
    return xt, t, target_v


def train(model, data, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    losses = []
    t0 = time.perf_counter()
    model.train()
    for step in range(1, args.train_steps + 1):
        xt, t, target_v = cfm_batch(data, args.batch_size, device)
        pred_v = model(xt, t)
        loss = ((pred_v - target_v) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % args.log_every == 0 or step == 1:
            print("step %5d/%d  loss %.6f" % (step, args.train_steps, losses[-1]))
    return losses, time.perf_counter() - t0


@torch.no_grad()
def euler_sample(model, n, steps, device):
    model.eval()
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / float(steps)
    t0 = time.perf_counter()
    for i in range(steps):
        t = torch.full((n, 1), float(i) / float(steps), device=device)
        x = x + dt * model(x, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return x.cpu().numpy(), elapsed


def chamfer_like(a, b, max_points=2000):
    if len(a) > max_points:
        a = a[np.random.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[np.random.choice(len(b), max_points, replace=False)]
    d = pairwise_distances(a, b)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def save_scatter_grid(real, samples_by_steps, out_path):
    cols = 1 + len(samples_by_steps)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2), constrained_layout=True)
    axes[0].scatter(real[:, 0], real[:, 1], s=3, alpha=0.45)
    axes[0].set_title("real two-moons")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    for ax, (steps, samples) in zip(axes[1:], samples_by_steps):
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.45)
        ax.set_title("ODE NFE=%d" % steps)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_loss(losses, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(losses, linewidth=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("MSE")
    ax.set_title("Conditional Flow Matching loss")
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sample-n", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--out-dir", default="runs/flow_matching_toy")
    parser.add_argument("--steps", default="1,2,4,8,16,32,64")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    data = sample_data(20000)
    model = VectorField(hidden=args.hidden).to(device)
    print("device:", device)
    print("training CFM: xt=(1-t)x0+t*x1, target velocity=x1-x0")
    losses, train_seconds = train(model, data, args, device)

    step_list = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    rows = []
    samples_by_steps = []
    for steps in step_list:
        samples, sample_seconds = euler_sample(model, args.sample_n, steps, device)
        metric = chamfer_like(samples, data)
        rows.append(
            {
                "nfe": steps,
                "sample_seconds": sample_seconds,
                "samples_per_second": args.sample_n / sample_seconds,
                "chamfer_like_lower_is_better": metric,
            }
        )
        samples_by_steps.append((steps, samples))
        print(
            "NFE=%3d  sample %.4fs  %.1f samples/s  chamfer-like %.4f"
            % (steps, sample_seconds, args.sample_n / sample_seconds, metric)
        )

    csv_path = os.path.join(args.out_dir, "nfe_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_loss(losses, os.path.join(args.out_dir, "loss.png"))
    save_scatter_grid(data[: args.sample_n], samples_by_steps, os.path.join(args.out_dir, "samples_by_nfe.png"))
    torch.save(model.state_dict(), os.path.join(args.out_dir, "vector_field.pt"))

    print("train_seconds: %.2f" % train_seconds)
    print("wrote:", args.out_dir)


if __name__ == "__main__":
    main()
