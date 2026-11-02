from pathlib import Path
from collections import Counter
import argparse


def inspect_label_dir(label_dir: Path):
    files = sorted(label_dir.glob("*.txt"))

    class_counter = Counter()
    empty_files = 0
    invalid_lines = 0
    total_objects = 0

    for f in files:
        text = f.read_text(encoding="utf-8").strip()

        if not text:
            empty_files += 1
            continue

        for line in text.splitlines():
            parts = line.strip().split()

            if len(parts) < 5:
                invalid_lines += 1
                continue

            cls_id = parts[0]
            class_counter[cls_id] += 1
            total_objects += 1

    print("=" * 80)
    print(f"Label dir: {label_dir}")
    print(f"Label files: {len(files)}")
    print(f"Empty label files: {empty_files}")
    print(f"Invalid lines: {invalid_lines}")
    print(f"Total objects: {total_objects}")
    print("Class id counts:")

    for cls_id, count in sorted(class_counter.items(), key=lambda x: int(float(x[0]))):
        print(f"  class {cls_id}: {count}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    for split in ["train", "val"]:
        label_dir = dataset_root / "labels" / split

        if label_dir.exists():
            inspect_label_dir(label_dir)
        else:
            print(f"Missing label dir: {label_dir}")


if __name__ == "__main__":
    main()