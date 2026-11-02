from ultralytics import YOLO
from pathlib import Path


def main():
    # ===== 路径配置 =====
    data_yaml = r"D:/New_research/datasets/bdd100k/data.yaml"
    project_dir = r"D:/New_research/runs"
    run_name = "yolov8n_baseline"

    # ===== 模型配置 =====
    model_name = "yolov8n.pt"   # 会自动下载官方预训练权重
    imgsz = 512
    epochs = 3
    batch = 4
    workers = 4
    device = 0   # 0 表示第一张 GPU；如果想用 CPU，改成 "cpu"

    # ===== 检查路径 =====
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # ===== 加载模型 =====
    model = YOLO(model_name)

    # ===== 开始训练 =====
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=project_dir,
        name=run_name,
        pretrained=True,
        exist_ok=True,
        verbose=True
    )

    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()