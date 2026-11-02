"""Copy and hash the selected original paired forecasting baseline; never train."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/forecasting/biflow_comp_dual_baseline_20261002"
RUN = "biflow_aria_v1_20261002"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError(f"Preserve existing snapshot: {DEST}")
    candidates = []
    for group in ("scripts", "outputs", "reports"):
        base = ROOT / group / RUN
        candidates.extend((p, p.relative_to(ROOT)) for p in sorted(base.rglob("*"))
                          if p.is_file() and "__pycache__" not in p.parts)
    for source, target in [
        ("plans/BIFLOW_ARIA_V1.md", "plans/BIFLOW_ARIA_V1.md"),
        ("AGENTS.md", "context/AGENTS_at_freeze.md"),
        ("scripts/archive_forecast_baseline_20261002.py", "scripts/archive_forecast_baseline_20261002.py"),
    ]:
        candidates.append((ROOT / source, Path(target)))
    assert candidates and all(p.is_file() for p, _ in candidates)
    DEST.mkdir(parents=True)
    entries = []
    for source, relative in candidates:
        before = sha(source)
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert before == sha(target) == sha(source), str(source)
        entries.append({"path": relative.as_posix(), "source": source.relative_to(ROOT).as_posix(),
                        "size": target.stat().st_size, "sha256": before})

    out = DEST / "outputs" / RUN
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    selected = summary["reviewed"]["comp_dual"]
    pairs = []
    for fold in range(6):
        for seed in (17, 29, 43):
            base = out / f"fold_{fold}" / f"seed_{seed}"
            pair = {"fold": fold, "seed": seed}
            for key, rel in [("camera", "camera.pt"), ("pedestrian", "comp_dual/checkpoint.pt")]:
                p = base / rel
                assert p.is_file(), str(p)
                pair[key] = {"path": p.relative_to(DEST).as_posix(), "sha256": sha(p)}
            complete = json.loads((base / "comp_dual/complete.json").read_text(encoding="utf-8"))
            assert complete["model_code_sha256"] == sha(DEST / "scripts" / RUN / "model.py")
            assert complete["code_sha256"] == sha(DEST / "scripts" / RUN / "train.py")
            pairs.append(pair)
    assert len(pairs) == 18

    readme = """# 旧版双任务轨迹预测固定基线

保存日期：2026 年 10 月 2 日。按用户选择，将原始实验的 `comp_dual` 固定为后续架构比较的主基线。相机网络和行人网络按原始 fold 与 seed 成对保存，保持原来的相机条件与重投影一致。

选定模型是旋转补偿轨迹输入、历史重建和未来预测两个头、未来相机预测参与行人条件及最终重投影的版本。这里有六个录制级测试划分、三个随机种子的十八组模型，不是一个在全部录制上拟合的部署模型。

## 已保存内容

- 原始实验代码及中文注释版，训练协议、数据划分和准备后的数据集。
- 十八组旧相机与 `comp_dual` 权重，以及原实验其余三种训练模型作为对照记录。
- 训练日志、逐窗口预测、统计、报告、架构说明和展示视频。
- 原始方案与保存当时的项目记忆；每个副本的 SHA256 清单。

快照保留工作区相对目录结构。原始视频、SAM 标注源文件和环境仍保留在原工作区；其历史来源清单在 `outputs/biflow_aria_v1_20261002/sources.json`。本快照包含准备后的实验数据，并不是原始采集数据与运行环境的完整打包。重新解析原始资料或渲染依赖原工作区资产；请勿直接在快照中运行会写入结果的训练或准备脚本。

## 固定结果

在原来的 117 个有效窗口、六个录制测试划分和三个种子上，未来 1.5 秒 ADE 为 **158.0551 像素**，FDE 为 **257.3352 像素**；录制等权 ADE 为 **174.3379 像素**。输入过去六点，输出未来十五点。

这是原始四种训练网络中窗口平均 ADE 最低的版本，并作为当前选定的主基线保留。保持最后位置的 ADE 为 141.4279 像素，原始报告也一并保留；“选定主基线”不表示已经超过所有参考方法。未来参考旋转的结果属于独立诊断，不能替代这里的预测相机结果。

## 使用约定

从 `manifest.json` 的 `selected_pairs` 选择同一 fold 与 seed 的相机和行人权重。这里保存的是旧相机网络，不能替换成全量相机网络或全录制重拟合权重后继续沿用此基线名称。未来试验另建版本目录，保持此快照不变；需要恢复时先核验哈希，再复制到新的工作目录。

入口：[原始结果](reports/biflow_aria_v1_20261002/RESULTS.md)、[模型架构](reports/biflow_aria_v1_20261002/MODEL_ARCHITECTURE.md)、[中文注释代码](scripts/biflow_aria_v1_20261002/model_annotated.py)、[展示素材](outputs/biflow_aria_v1_20261002/presentation/START_HERE.md)。
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    entries.append({"path": "README.md", "source": None, "size": (DEST / "README.md").stat().st_size,
                    "sha256": sha(DEST / "README.md")})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "selected_baseline_snapshot",
        "source_workspace": str(ROOT), "source_run": RUN, "selected_variant": "comp_dual",
        "camera_version": "original paired per-fold model, not expanded-data or all-recording refit",
        "past_points": 6, "future_points": 15, "future_seconds": 1.5,
        "unique_prediction_windows": 117, "seeds": [17, 29, 43], "folds": 6,
        "window_mean_metrics": selected["window_mean"],
        "equal_recording_metrics": selected["equal_recording"],
        "selected_pairs": pairs, "files": entries,
        "file_count_excluding_manifest": len(entries), "bytes": sum(e["size"] for e in entries),
        "verification": "All copied files matched source SHA256 before and after copying; all 18 selected pairs match original training and model source hashes.",
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Independent second pass over the complete saved payload.
    reread = json.loads((DEST / "manifest.json").read_text(encoding="utf-8"))
    for e in reread["files"]:
        assert sha(DEST / e["path"]) == e["sha256"], e["path"]
    print(json.dumps({"snapshot": str(DEST), "selected_pairs": len(pairs), "files": len(entries),
                      "MiB": round(manifest["bytes"] / 2**20, 3), "verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
