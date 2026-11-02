# Aria Pedestrian Trajectory Research

第一人称行人轨迹预测研究的代码、工具与精选中日文文档。研究记录截至 **2026-10-29**，共享快照整理于 **2026-11-03**。

一人称視点の歩行者軌跡予測に関するコード、ツール、選定した中国語・日本語文書です。研究記録の基準日は **2026-10-29**、共有スナップショットの整理日は **2026-11-03** です。

## 阅读入口 / 読み始める場所

- [中文项目概览](reports/github_share_inventory_20261103/PROJECT_OVERVIEW.txt)
- [中文7份文档索引](docs/DOCS_INDEX_ZH.txt)
- [日本語の読順・原文対応](docs/ja/00_README_JA.txt)
- [日本語プロジェクト概要](docs/ja/01_PROJECT_OVERVIEW_JA.txt)
- [受領側AI向け引継ぎ](docs/ja/AI_HANDOFF_JA.txt)

## 代码入口 / コード入口

| 目录 / ディレクトリ | 用途 |
|---|---|
| `scripts/translation_forecast_20261020/` | 分离GRU与历史相机平移 / 分離GRUと過去のカメラ並進 |
| `scripts/g1_bulk_association_20261029/` | G1对应关联与窗口构造 / G1の対応付けと窓構築 |
| `scripts/g1_expanded_forecast_20261029/` | 短时预测的数据扩充对照 / 短期予測のデータ拡充対照 |
| `scripts/horizon_15_15_g1_20261029/` | 1.5秒历史→1.5秒未来 / 過去1.5秒→将来1.5秒 |
| `scripts/tract_learning_map_20261029/` | 学习过程诊断 / 学習過程の診断 |
| `scripts/scene_pedestrian_20261029/` | 场景研究前期调查 / シーン研究の予備調査 |
| `scripts/gaze_motion_20261029/` | 视线研究前期调查 / 視線研究の予備調査 |
| `tools/`, `external/`, `third_party/`, `SAM3/` | 工具及第三方源码 / ツールと第三者ソース |

各工具的依赖声明、README与LICENSE保留在相应目录。本仓库包含多个实验与工具，尚未验证统一的一键安装环境。

各ツールの依存宣言、README、LICENSEは対応するディレクトリ内にあります。複数の実験・ツールを含み、統一されたワンクリック環境は未検証です。

## 数据与复现范围 / データと再現範囲

原始录像及配套姿态、点云、视线数据计划在Google网盘单独共享；[数据索引](data_index/aria_files.csv)列出路径与体积，[网盘入口](data_index/DRIVE_LINK.txt)将在上传后补充。

原動画と付随する姿勢・点群・視線データはGoogle Driveで別途共有する予定です。[データ索引](data_index/aria_files.csv)にパスとサイズを示し、[Driveリンク](data_index/DRIVE_LINK.txt)はアップロード後に追記します。

本仓库不包含KITTI、BDD100K、PIE、`outputs/`、`runs/`、环境缓存或大型模型权重。近期标注窗口与实验权重多数位于未共享的`outputs/`，因此克隆本仓库不足以直接复现全部训练。源代码中的原环境路径仍需按接手环境配置。

KITTI、BDD100K、PIE、`outputs/`、`runs/`、環境キャッシュ、大規模モデル重みは含みません。最近の注釈窓・実験重みの多くは未共有の`outputs/`にあるため、cloneだけでは全学習を直接再現できません。元環境のパスは受領環境に合わせた設定が必要です。

这是单独整理的源码快照，不包含原项目Git历史。两个研究分支的新增脚本已按目录纳入，原分支提交号见[分支快照记录](BRANCH_CODE_SNAPSHOTS.json)。原项目和历史结果仍保留在本地。

これは元プロジェクトのGit履歴を含まない独立したソーススナップショットです。2研究分岐の追加スクリプトを取り込み、元の分岐・commitは[スナップショット記録](BRANCH_CODE_SNAPSHOTS.json)に保存しています。
