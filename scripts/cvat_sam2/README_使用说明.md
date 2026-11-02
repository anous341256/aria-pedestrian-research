# 官方 SAM 2 Tracker：安装与接入记录

日期：2026-09-27。

## 当前状态

- 已安装独立环境，已下载 SAM 2.1 Large 权重，GPU 本地连续传播测试通过。
- **已完成接入验收：私有模型 #5158 已出现，网页调用本地 agent 完成第 75–125 帧单人追踪，保存并刷新后第 115 帧结果仍在。** 这是端到端功能验收，不是标注质量验收。
- 已创建独立测试任务 `SAM2 official smoke - Aria164557 rectified - 20260927`（任务 #2623936，作业 #4515603），使用现有 60 秒去畸变视频；未修改旧任务标注，未进行模型训练。
- 测试输出不是人工标注，不是金标，也不能据此计算整段视频的通过率。

## 你现在只需要做这件事

1. 打开 <https://app.cvat.ai/profile#security>，在 Personal Access Tokens 旁点 `＋`。
2. 建议令牌名称为 `local-sam2-tracker`，有效期先设 7 天；需要读写权限（不能只读，因为服务需要注册并提交推理结果）。权限选项目前是只读或读写，不是模型级细粒度授权。
3. 复制令牌，双击项目根目录 `E:\New_research\Open_CVAT_SAM2_Tracker.cmd`。
4. 在打开的终端里粘贴令牌并回车。输入时不显示字符是正常的。**不要发到聊天里。**
5. 保持终端开启，等官方客户端进入等待请求状态。刷新 CVAT Models，寻找：

   `Aria SAM2.1 Large - official local GPU`

服务是私有的，默认注册在个人工作区。启动器只注册和运行服务，不主动发起标注。令牌不保存到脚本、配置或日志，只有本次启动进程及其官方客户端子进程使用；再次启动需要重新输入。不要分享含令牌的终端截图。测试结束可在 CVAT 撤销令牌。

不要重复开启多个终端：官方同一追踪函数只能有一个 agent，本入口带单实例锁。

## 在 CVAT 里怎么用

这不是普通的矩形框 Tracker。官方 SAM 2 Tracker 接受 **polygon（多边形）和 mask（掩码）**，不能直接接普通 rectangle。它不会自动找出整段视频中所有新出现的人。

账号接入验证之后，先在单独测试任务中试 1～3 个人、5～10 秒，不在既有人工标签上直接批量试验：

1. 到一个目标看得清的帧，用 CVAT 已有的 SAM 2/3 交互分割器点选目标，生成轮廓；也可画多边形。
2. 选中该轮廓，打开对象菜单的 `Run annotation action`（默认快捷键 `Ctrl+E`）。
3. 找到新注册模型对应的 `AI Tracker` 操作，设置结束帧。
4. 如果起点是 polygon shape 而最终需要轨迹，使用 `Convert polygon shapes to tracks` 选项。
5. 运行后逐段检查；到首次跟错处修正轮廓，从该帧再次追踪。不要把一次传播自动当成整段合格。

**入口不在左侧那个只接受矩形框的 Tracker 选择框里。** 若在那里找不到 SAM2，并不意味着安装失败。

暂停/关闭本机服务会中断未完成的追踪；已保存在 CVAT 的标注仍在。关闭窗口不会自动删除私有模型入口，下次启动会复用同名入口，不盲目重复注册。

## 本地验证结果与边界

输入：现有去畸变彩色视频 `outputs/tracktrack_reproduction/aria_164557_first60s_v1/input/aria_rgb_rectified_first60s.mp4`，2560×1920、10 fps。

- 测试范围：从 0 基第 75 帧到第 125 帧，共 51 帧，约 5 秒。
- 起点：人工视觉定位一个白衣行人所在矩形，使用同一 SAM 2.1 模型生成初始 mask；随后调用**未修改的 CVAT 官方适配器**传播。
- 循环耗时（包含解码、叠加、写预览）：10.33 秒，不含模型初次加载；不是 CVAT 云端交互速度。
- PyTorch 峰值已分配显存：1735.63 MiB；不等同于整个系统的总显存占用。
- 50 帧产生非空 mask；最后一帧为空。抽看的第 75/85/95/105/115 帧跟随同一白衣行人，第 125 帧目标已离开可见区域。只做了关键帧初看，未进行每帧独立人工验收。
- **50/51 不能解释为 98% 准确率或通过率。** 非空 mask 不代表身份或边界正确。

输出：

- `outputs/cvat_sam2_setup/local_smoke/contact_sheet.jpg`：6 个关键帧。
- `outputs/cvat_sam2_setup/local_smoke/preview.mp4`：叠加预览（本地 MPEG-4 编码，不保证所有浏览器支持）。
- `outputs/cvat_sam2_setup/local_smoke/result.json`：逐帧非空情况和框范围、显存及计时。
- `outputs/cvat_sam2_setup/pip-freeze.txt`：实际安装版本。

本轮没有编译 SAM2 可选的 connected-components CUDA 扩展，因此官方代码会警告并跳过“小孔填充”后处理。这是 SAM2 官方允许的降级路径；主模型仍在 GPU 上运行。**不能声称与启用该扩展的掩码输出逐像素一致。** CVAT 适配器和 SAM2 推理代码均未修改。若后续边界质量评估需要，单独补齐编译工具后再验证。

## CVAT 网页端验收（2026-09-27）

- 私有模型 ID：5158，Models 页面显示 `native tracker`，所有者 `anous341256`。
- [独立测试作业](https://app.cvat.ai/tasks/2623936/jobs/4515603)。600 帧视频，仅测试第 75–125 帧的一个白衣行人。
- 起点文件：`outputs/cvat_sam2_setup/cvat_seed/seed_only_frame75.xml`。该文件只有第 75 帧一个由 SAM2 生成的 16 顶点轮廓和第 76 帧 outside 结束标记，**没有后续追踪结果**。仅追加到新建的空测试任务。
- 实际入口：第 75 帧，对象 `person / POLYGON TRACK / ID 1` 的三点菜单 → `Run annotation action` → `AI Tracker #5158: Aria SAM2.1 Large - official local GPU` → Target frame `125` → Run。
- 网页显示运行进度，完成后已保存；刷新后第 115 帧仍有追踪对象，验证了结果持久化。
- 第 95 帧：轮廓位置和形状已变化，仍对应同一白衣行人，不是首帧复制。
- 第 115 帧：目标仍对应同一人，但轮廓只覆盖上半身。**不能视为完整行人标注合格。**
- 起点存在已查明的测试限制：将初始 mask 转 polygon 时只取最大连通区域，栏杆遮挡使下半身没有包含在 16 顶点起始轮廓中（其范围约 x=1690–1747、y=1147–1219）。因此本次不用于公平评价完整人物分割质量。后续质量测试应先确认完整对象提示，或保留 mask 的不连通区域，不能把提示错误直接归为模型性能。
- 没有人工逐帧验收，不报告通过率；没有启动全量标注。原有任务和人工标注未修改。

## 环境与可复现性

全部新增大文件在 E 盘项目或已存放于 E 盘的 WSL Ubuntu-24.04 中，不修改 `/opt/aria-vrs`、`/opt/aria-mps`、`/opt/sam3`、`/opt/tracktrack`。

| 项目 | 值 |
|---|---|
| Python 虚拟环境 | WSL `/opt/cvat-sam2`，Python 3.12 |
| 权重缓存 | WSL `/opt/cvat-sam2-cache/huggingface` |
| CVAT CLI / SDK | 2.76.0 / 2.76.0，与检查时线上版本一致 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| GPU | RTX 5070 Laptop，8 GB |
| 模型 | `facebook/sam2.1-hiera-large` |
| 权重 HF revision | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |
| 官方适配器仓库 | `tools/cvat-models-official`，commit `d2ebbc7f37d9c002d74fa8d392e9f3f8cb3aa419` |
| 官方 SAM2 仓库 | `tools/sam2-cvat-official`，commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`（按适配器 requirements 锁定） |
| CVAT 源码参考 | `tools/cvat-sam2-official`，commit `71b62507f32dd4f2b931f2293e63291f10d1827e` |
| 部署方式 | 官方 CLI 原生 agent，无 Docker、无新增网页服务 |

官方文档中的 `cvat/ai-models/...` 路径在当前主仓库已不存在；实际代码已迁到官方 `cvat-ai/cvat-models`。本轮使用后者，不是第三方重写。

已验证：`pip check` 无依赖冲突；官方命令帮助可用；本地 GPU 推理及适配器传播可用；启动脚本语法检查通过；账号授权、远端函数注册、网页端动作与保存后刷新读取均已验证。仍需独立评价真实标注质量与人工纠错成本。

技术复测入口（只写本地测试输出，不调用 CVAT）：

```powershell
wsl -d Ubuntu-24.04 -- /opt/cvat-sam2/bin/python /mnt/e/New_research/scripts/cvat_sam2/smoke_test.py
```

## 故障定位

- HTTP 401：令牌错误或过期；HTTP 403：权限/账号访问受限。先核对，不反复创建模型。
- 注册成功但网页不出现：刷新 Models，确认仍是同一用户的个人工作区。
- 名字出现但追踪不运行：查看启动终端是否仍在，等待 agent 开始轮询；不要同时启动第二个 agent。
- 左侧矩形 Tracker 下拉框没有 SAM2：请改用轮廓对象的 `Run annotation action`。
- OOM：减少同时追踪的对象数、关闭其他 GPU 任务后再试；不要据此承诺整段视频一次运行可行。
- `cannot import name '_C'`：本轮缺少可选孔洞填充扩展，已记录，不是 GPU 不可用。

## 官方依据

- [CVAT SAM2 Tracker 使用与限制](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/)
- [当前官方适配器与 CLI 参数](https://github.com/cvat-ai/cvat-models/tree/d2ebbc7f37d9c002d74fa8d392e9f3f8cb3aa419/tracker/sam2)
- [Meta SAM2 安装与可选 CUDA 后处理说明](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md)
- [CVAT 访问令牌说明](https://docs.cvat.ai/docs/api_sdk/access_tokens/)

如果小样本仍需频繁纠错，再根据“每条可用轨迹需要多少人工时间”判断是否转向少量精标＋伪标签/弱监督。弱监督也需要可靠、独立的测试标注；本轮不启动新的训练计划。
