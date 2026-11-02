import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


DEGRADED_TRAIN_SPLITS = [
    "train_fog_medium",
    "train_rain_heavy",
    "train_lowlight_medium",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def collect_images(image_dir: Path):
    image_files = []

    for ext in IMAGE_EXTS:
        image_files.extend(image_dir.glob(f"*{ext}"))

    return sorted(image_files)


class KITTIDeg2CleanDataset(Dataset):
    """
    Dataset for degraded-to-clean feature prediction.

    For each clean training image:
        clean:    images/train/xxxx.png
        degraded: images/train_fog_medium/xxxx.png
                  images/train_rain_heavy/xxxx.png
                  images/train_lowlight_medium/xxxx.png

    During training, one degraded version is randomly selected for each clean image.
    """

    def __init__(
        self,
        dataset_root: Path,
        degraded_splits,
        transform=None,
    ):
        self.dataset_root = dataset_root
        self.clean_dir = dataset_root / "images" / "train"
        self.degraded_splits = degraded_splits
        self.transform = transform

        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean train image dir not found: {self.clean_dir}")

        self.clean_images = collect_images(self.clean_dir)

        if len(self.clean_images) == 0:
            raise RuntimeError(f"No clean training images found in: {self.clean_dir}")

        self.samples = []

        for clean_path in self.clean_images:
            degraded_paths = []

            for split in degraded_splits:
                degraded_path = dataset_root / "images" / split / clean_path.name

                if degraded_path.exists():
                    degraded_paths.append(degraded_path)

            if degraded_paths:
                self.samples.append({
                    "clean": clean_path,
                    "degraded_candidates": degraded_paths,
                })

        if len(self.samples) == 0:
            raise RuntimeError(
                "No clean-degraded image pairs found. "
                "Please check train_fog_medium / train_rain_heavy / train_lowlight_medium."
            )

        print(f"Found {len(self.samples)} clean images with degraded counterparts.")
        print(f"Using degraded splits: {degraded_splits}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        clean_path = item["clean"]
        degraded_path = random.choice(item["degraded_candidates"])

        clean_img = Image.open(clean_path).convert("RGB")
        degraded_img = Image.open(degraded_path).convert("RGB")

        if self.transform is not None:
            clean_img = self.transform(clean_img)
            degraded_img = self.transform(degraded_img)

        return {
            "clean": clean_img,
            "degraded": degraded_img,
            "clean_path": str(clean_path),
            "degraded_path": str(degraded_path),
        }


class ResNet18Encoder(nn.Module):
    """
    ResNet18 encoder that outputs global feature vector [B, 512].
    The final classification layer is removed.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = models.resnet18(weights=weights)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class PredictorMLP(nn.Module):
    """
    Predictor head.
    It maps student degraded features to clean-feature space.
    """

    def __init__(self, dim=512, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def cosine_prediction_loss(z_pred, z_clean):
    """
    z_pred:  predicted clean feature from degraded image
    z_clean: teacher clean feature

    Both are [B, C].
    """
    z_pred = F.normalize(z_pred, dim=1)
    z_clean = F.normalize(z_clean.detach(), dim=1)

    loss = 1.0 - (z_pred * z_clean).sum(dim=1).mean()

    return loss


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_loss_csv(loss_history, path: Path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()

        for row in loss_history:
            writer.writerow(row)


def save_loss_plot(loss_history, path: Path):
    epochs = [row["epoch"] for row in loss_history]
    losses = [row["train_loss"] for row in loss_history]

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Prediction Loss")
    plt.title("Encoder Pretraining Loss")
    plt.grid(True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def train_one_epoch(
    teacher,
    student,
    predictor,
    dataloader,
    optimizer,
    device,
    epoch,
):
    teacher.eval()
    student.train()
    predictor.train()

    running_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress:
        clean = batch["clean"].to(device, non_blocking=True)
        degraded = batch["degraded"].to(device, non_blocking=True)

        with torch.no_grad():
            z_clean = teacher(clean)

        z_degraded = student(degraded)
        z_pred = predictor(z_degraded)

        loss = cosine_prediction_loss(z_pred, z_clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{running_loss / max(1, num_batches):.4f}",
        })

    avg_loss = running_loss / max(1, num_batches)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to KITTI YOLO dataset root, e.g. D:/New_research/datasets/kitti_yolo"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for encoder pretraining results"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )

    parser.add_argument(
        "--no_imagenet",
        action="store_true",
        help="If set, do not use ImageNet pretrained ResNet18."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    dataset_root = Path(args.dataset).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("Degraded-to-Clean Feature Prediction Encoder Pretraining")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    pretrained = not args.no_imagenet

    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = KITTIDeg2CleanDataset(
        dataset_root=dataset_root,
        degraded_splits=DEGRADED_TRAIN_SPLITS,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    teacher = ResNet18Encoder(pretrained=pretrained).to(device)
    student = ResNet18Encoder(pretrained=pretrained).to(device)
    predictor = PredictorMLP(dim=512, hidden_dim=512).to(device)

    # Freeze teacher encoder.
    for p in teacher.parameters():
        p.requires_grad = False

    teacher.eval()

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        "method": "degraded_to_clean_feature_prediction",
        "encoder": "resnet18",
        "teacher": "frozen_resnet18",
        "student": "trainable_resnet18",
        "predictor": "2-layer MLP",
        "loss": "1 - cosine_similarity(predicted_clean_feature, clean_teacher_feature)",
        "dataset": str(dataset_root),
        "degraded_train_splits": DEGRADED_TRAIN_SPLITS,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "device": str(device),
        "imagenet_pretrained": pretrained,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(dataset),
    }

    save_json(config, output_dir / "pretrain_config.json")

    loss_history = []

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            teacher=teacher,
            student=student,
            predictor=predictor,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {avg_loss:.6f}")

        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
        }

        loss_history.append(row)

        save_loss_csv(loss_history, output_dir / "loss_curve.csv")
        save_loss_plot(loss_history, output_dir / "loss_curve.png")

        # Save latest
        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "config": config,
            },
            output_dir / "latest_checkpoint.pth"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": student.state_dict(),
                    "predictor_state_dict": predictor.state_dict(),
                    "train_loss": avg_loss,
                    "config": config,
                },
                output_dir / "best_checkpoint.pth"
            )

            torch.save(student.state_dict(), output_dir / "student_encoder.pth")
            torch.save(predictor.state_dict(), output_dir / "predictor.pth")

            print(f"Saved best checkpoint. Best loss: {best_loss:.6f}")

    config["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config["best_loss"] = best_loss
    save_json(config, output_dir / "pretrain_config.json")

    print("=" * 80)
    print("Encoder pretraining completed.")
    print(f"Output directory: {output_dir}")
    print(f"Best loss: {best_loss:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()