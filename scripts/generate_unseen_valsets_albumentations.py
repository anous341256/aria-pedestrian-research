import argparse
import json
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


CLASS_NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
}


CONDITIONS = [
    "snow_medium",
    "blur_medium",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def collect_images(img_dir: Path):
    image_files = []

    for ext in IMAGE_EXTS:
        image_files.extend(img_dir.glob(f"*{ext}"))

    return sorted(image_files)


def get_transform(condition: str):
    """
    Unseen validation degradations.

    These transforms are NOT included in the current weather-augmented training set.
    They are used to test whether weather augmentation generalizes beyond the
    synthetic degradation families seen during training.
    """

    if condition == "snow_medium":
        return A.Compose([
            A.RandomSnow(
                snow_point_range=(0.20, 0.35),
                brightness_coeff=1.5,
                p=1.0
            )
        ])

    if condition == "blur_medium":
        return A.Compose([
            A.MotionBlur(
                blur_limit=(9, 15),
                allow_shifted=True,
                p=1.0
            )
        ])

    raise ValueError(f"Unknown condition: {condition}")


def copy_labels(src_label_dir: Path, dst_label_dir: Path):
    ensure_dir(dst_label_dir)

    label_files = sorted(src_label_dir.glob("*.txt"))

    for label_file in tqdm(label_files, desc=f"Copy labels -> {dst_label_dir.name}"):
        shutil.copy2(label_file, dst_label_dir / label_file.name)


def generate_condition(src_img_dir: Path, dst_img_dir: Path, condition: str):
    ensure_dir(dst_img_dir)

    transform = get_transform(condition)

    image_files = collect_images(src_img_dir)

    if not image_files:
        raise RuntimeError(f"No validation images found in: {src_img_dir}")

    for img_path in tqdm(image_files, desc=f"Generating val_{condition}"):
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Warning: failed to read image: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = transform(image=image_rgb)
        out_rgb = augmented["image"]

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(dst_img_dir / img_path.name), out_bgr)


def make_yaml(dataset_root: Path, condition: str):
    yaml_path = dataset_root / f"kitti_{condition}.yaml"

    content = [
        f"path: {str(dataset_root).replace(chr(92), '/')}",
        "",
        "train: images/train",
        f"val: images/val_{condition}",
        "",
        "names:",
    ]

    for cls_id, cls_name in CLASS_NAMES.items():
        content.append(f"  {cls_id}: {cls_name}")

    yaml_path.write_text("\n".join(content), encoding="utf-8")

    return yaml_path


def count_split(dataset_root: Path, split_name: str):
    img_dir = dataset_root / "images" / split_name
    label_dir = dataset_root / "labels" / split_name

    images = collect_images(img_dir) if img_dir.exists() else []
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

    return {
        "split": split_name,
        "image_dir": str(img_dir),
        "label_dir": str(label_dir),
        "num_images": len(images),
        "num_labels": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to KITTI YOLO dataset root, e.g. D:/New_research/datasets/kitti_yolo"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated validation sets"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    dataset_root = Path(args.dataset).resolve()

    src_img_dir = dataset_root / "images" / "val"
    src_label_dir = dataset_root / "labels" / "val"

    if not src_img_dir.exists():
        raise FileNotFoundError(f"Source val image dir not found: {src_img_dir}")

    if not src_label_dir.exists():
        raise FileNotFoundError(f"Source val label dir not found: {src_label_dir}")

    config = {
        "tool": "Albumentations",
        "purpose": "Generate unseen degradation validation sets for robustness generalization testing.",
        "seed": args.seed,
        "dataset_root": str(dataset_root),
        "source_images": str(src_img_dir),
        "source_labels": str(src_label_dir),
        "conditions": [],
        "note": (
            "Only validation images are degraded. Labels are copied unchanged because "
            "these transforms are pixel-level and do not alter bounding-box geometry. "
            "These conditions were not included in the weather-augmented training set."
        )
    }

    for condition in CONDITIONS:
        split_name = f"val_{condition}"

        dst_img_dir = dataset_root / "images" / split_name
        dst_label_dir = dataset_root / "labels" / split_name

        if dst_img_dir.exists():
            if args.overwrite:
                shutil.rmtree(dst_img_dir)
            else:
                print(f"Skip existing image directory: {dst_img_dir}")
                print("Use --overwrite if you want to regenerate it.")
        else:
            generate_condition(
                src_img_dir=src_img_dir,
                dst_img_dir=dst_img_dir,
                condition=condition
            )

        if dst_label_dir.exists():
            if args.overwrite:
                shutil.rmtree(dst_label_dir)
                copy_labels(src_label_dir, dst_label_dir)
            else:
                print(f"Skip existing label directory: {dst_label_dir}")
                print("Use --overwrite if you want to recopy it.")
        else:
            copy_labels(src_label_dir, dst_label_dir)

        yaml_path = make_yaml(dataset_root, condition)
        split_stats = count_split(dataset_root, split_name)

        config["conditions"].append({
            "condition": condition,
            "split_name": split_name,
            "image_dir": str(dst_img_dir),
            "label_dir": str(dst_label_dir),
            "yaml": str(yaml_path),
            "stats": split_stats,
        })

    config_path = dataset_root / "unseen_val_generation_albumentations_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Unseen degradation validation sets generated.")
    print(f"Dataset root: {dataset_root}")
    print(f"Config saved to: {config_path}")
    print("=" * 80)

    for item in config["conditions"]:
        stats = item["stats"]
        print(
            f"{stats['split']}: "
            f"{stats['num_images']} images, "
            f"{stats['num_labels']} labels"
        )


if __name__ == "__main__":
    main()