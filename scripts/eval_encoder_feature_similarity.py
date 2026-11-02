import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


EVAL_CONDITIONS = {
    "fog_heavy": "val_fog_heavy",
    "rain_heavy": "val_rain_heavy",
    "lowlight_medium": "val_lowlight_medium",
    "blur_medium": "val_blur_medium",
    "snow_medium": "val_snow_medium",
}


class ResNet18Encoder(nn.Module):
    """
    ResNet18 encoder that outputs global feature vector [B, 512].
    Must match the definition used during pretraining.
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


def collect_images(image_dir: Path):
    image_files = []

    for ext in IMAGE_EXTS:
        image_files.extend(image_dir.glob(f"*{ext}"))

    return sorted(image_files)


def load_image(path: Path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def compute_pair_metrics(encoder, clean_tensor, degraded_tensor, device):
    """
    Compute cosine similarity and L2 distance between clean/degraded features.
    """
    clean_tensor = clean_tensor.unsqueeze(0).to(device)
    degraded_tensor = degraded_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        z_clean = encoder(clean_tensor)
        z_degraded = encoder(degraded_tensor)

        z_clean_norm = F.normalize(z_clean, dim=1)
        z_degraded_norm = F.normalize(z_degraded, dim=1)

        cosine = (z_clean_norm * z_degraded_norm).sum(dim=1).item()

        l2 = torch.norm(z_clean_norm - z_degraded_norm, p=2, dim=1).item()

    return cosine, l2


def evaluate_encoder_on_condition(
    encoder,
    encoder_name,
    dataset_root: Path,
    condition_name: str,
    split_name: str,
    transform,
    device,
    max_images=None,
):
    clean_dir = dataset_root / "images" / "val"
    degraded_dir = dataset_root / "images" / split_name

    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean val image dir not found: {clean_dir}")

    if not degraded_dir.exists():
        raise FileNotFoundError(f"Degraded val image dir not found: {degraded_dir}")

    clean_images = collect_images(clean_dir)

    if max_images is not None:
        clean_images = clean_images[:max_images]

    cosines = []
    l2s = []

    per_image_rows = []

    encoder.eval()

    for clean_path in tqdm(clean_images, desc=f"{encoder_name} | {condition_name}"):
        degraded_path = degraded_dir / clean_path.name

        if not degraded_path.exists():
            continue

        clean_tensor = load_image(clean_path, transform)
        degraded_tensor = load_image(degraded_path, transform)

        cosine, l2 = compute_pair_metrics(
            encoder=encoder,
            clean_tensor=clean_tensor,
            degraded_tensor=degraded_tensor,
            device=device,
        )

        cosines.append(cosine)
        l2s.append(l2)

        per_image_rows.append({
            "encoder": encoder_name,
            "condition": condition_name,
            "image": clean_path.name,
            "cosine_similarity": cosine,
            "normalized_l2_distance": l2,
        })

    if len(cosines) == 0:
        raise RuntimeError(f"No valid image pairs found for condition: {condition_name}")

    summary = {
        "encoder": encoder_name,
        "condition": condition_name,
        "num_pairs": len(cosines),
        "mean_cosine_similarity": float(np.mean(cosines)),
        "std_cosine_similarity": float(np.std(cosines)),
        "median_cosine_similarity": float(np.median(cosines)),
        "mean_normalized_l2_distance": float(np.mean(l2s)),
        "std_normalized_l2_distance": float(np.std(l2s)),
        "median_normalized_l2_distance": float(np.median(l2s)),
    }

    return summary, per_image_rows


def write_csv(rows, path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_pretrained_student(student_path: Path, device):
    encoder = ResNet18Encoder(pretrained=False)

    state = torch.load(student_path, map_location=device)

    # student_encoder.pth is directly state_dict.
    # best_checkpoint.pth contains "student_state_dict".
    if isinstance(state, dict) and "student_state_dict" in state:
        state_dict = state["student_state_dict"]
    else:
        state_dict = state

    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()

    return encoder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to KITTI YOLO dataset root"
    )

    parser.add_argument(
        "--student",
        type=str,
        required=True,
        help="Path to predictive-pretrained student_encoder.pth or best_checkpoint.pth"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for feature similarity evaluation"
    )

    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    student_path = Path(args.student).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not student_path.exists():
        raise FileNotFoundError(f"Student encoder not found: {student_path}")

    print("=" * 80)
    print("Encoder Feature Similarity Evaluation")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print(f"Student encoder: {student_path}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Baseline encoder: ImageNet pretrained ResNet18
    imagenet_encoder = ResNet18Encoder(pretrained=True).to(device)
    imagenet_encoder.eval()

    # Your encoder: predictive-pretrained student
    student_encoder = load_pretrained_student(student_path, device)

    encoders = [
        ("imagenet_resnet18", imagenet_encoder),
        ("predictive_student_resnet18", student_encoder),
    ]

    summary_rows = []
    all_per_image_rows = []

    for encoder_name, encoder in encoders:
        for condition_name, split_name in EVAL_CONDITIONS.items():
            degraded_dir = dataset_root / "images" / split_name

            if not degraded_dir.exists():
                print(f"Skip missing condition: {condition_name}, dir not found: {degraded_dir}")
                continue

            summary, per_image_rows = evaluate_encoder_on_condition(
                encoder=encoder,
                encoder_name=encoder_name,
                dataset_root=dataset_root,
                condition_name=condition_name,
                split_name=split_name,
                transform=transform,
                device=device,
                max_images=args.max_images,
            )

            summary_rows.append(summary)
            all_per_image_rows.extend(per_image_rows)

    write_csv(summary_rows, output_dir / "feature_similarity_summary.csv")
    write_csv(all_per_image_rows, output_dir / "feature_similarity_per_image.csv")

    save_json(
        {
            "dataset": str(dataset_root),
            "student_encoder": str(student_path),
            "imgsz": args.imgsz,
            "device": str(device),
            "conditions": EVAL_CONDITIONS,
            "summary": summary_rows,
        },
        output_dir / "feature_similarity_results.json"
    )

    # Generate comparison markdown
    report_path = output_dir / "feature_similarity_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Encoder Feature Similarity Evaluation Report\n\n")
        f.write(f"Dataset: `{dataset_root}`\n\n")
        f.write(f"Student encoder: `{student_path}`\n\n")

        f.write("## Summary\n\n")
        f.write("| Encoder | Condition | Num pairs | Mean Cosine | Mean Normalized L2 |\n")
        f.write("|---|---|---:|---:|---:|\n")

        for row in summary_rows:
            f.write(
                f"| {row['encoder']} "
                f"| {row['condition']} "
                f"| {row['num_pairs']} "
                f"| {row['mean_cosine_similarity']:.4f} "
                f"| {row['mean_normalized_l2_distance']:.4f} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write(
            "- Higher cosine similarity means clean and degraded images are closer in feature space.\n"
        )
        f.write(
            "- Lower normalized L2 distance also means better feature consistency.\n"
        )
        f.write(
            "- If `predictive_student_resnet18` shows higher cosine similarity than `imagenet_resnet18`, "
            "the predictive pretraining has improved feature-level degradation robustness.\n"
        )

    print("=" * 80)
    print("Feature similarity evaluation completed.")
    print(f"Summary CSV: {output_dir / 'feature_similarity_summary.csv'}")
    print(f"Report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()