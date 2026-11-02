"""Read one existing MPS result without changing source data."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.spatial.transform import Rotation

ROOT = Path("/mnt/e/New_research")
SOURCE = ROOT / "data/Aria/raw_vrs/mps_unknown_20260803_150809_vrs/slam/closed_loop_trajectory.csv"
OUT = ROOT / "outputs/mps_translation_inspection_20261020"
OUT.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = sha(SOURCE)
with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))
ts = np.array([int(r["tracking_timestamp_us"]) for r in rows], dtype=np.int64)
assert np.all(np.diff(ts) > 0)
assert len({r["graph_uid"] for r in rows}) == 1, "Do not join unrelated world frames"
def cols(names):
    return np.array([[float(r[n]) for n in names] for r in rows])
p = cols([f"t{a}_world_device" for a in "xyz"])
q = cols([f"q{a}_world_device" for a in "xyzw"])
v = cols([f"device_linear_velocity_{a}_device" for a in "xyz"])
quality = cols(["quality_score"]).ravel()
assert np.isfinite(np.column_stack([p, q, v, quality])).all()
assert np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) < 1e-5
t = (ts - ts[0]) / 1e6
# One existing native sample at/before each 0.1-second grid time.
# No interpolation, filtering, coordinate rotation or RGB alignment.
grid = ts[0] + np.arange(int((ts[-1]-ts[0]) // 100000)+1, dtype=np.int64)*100000
idx = np.unique(np.searchsorted(ts, grid, side="right")-1)
d = p-p[0]
speed = np.linalg.norm(v, axis=1)
angle = np.rad2deg((Rotation.from_quat(q[0]).inv()*Rotation.from_quat(q[idx])).magnitude())
copy = OUT / "closed_loop_trajectory_original.csv"
shutil.copy2(SOURCE, copy)
assert sha(copy) == before
fields = ["source_row_0based", "graph_uid", "tracking_timestamp_us", "elapsed_s"]
fields += [f"t{a}_world_device_m" for a in "xyz"]
fields += [f"delta_{a}_from_first_pose_m" for a in "xyz"]
fields += [f"q{a}_world_device" for a in "xyzw"]
fields += ["device_speed_mps", "orientation_change_from_first_deg", "quality_score"]
with (OUT/"trajectory_10hz.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    for k, i in enumerate(idx):
        writer.writerow([int(i), rows[i]["graph_uid"], int(ts[i]), t[i], *p[i], *d[i],
                         *q[i], speed[i], angle[k], quality[i]])
stats = {
    "recording": "unknown_20260803_150809",
    "source": str(SOURCE), "source_sha256": before,
    "native_rows": len(rows), "duration_s": float(t[-1]),
    "median_native_interval_ms": float(np.median(np.diff(ts))/1000),
    "max_native_gap_ms": float(np.max(np.diff(ts))/1000),
    "display_rows": len(idx),
    "display_max_sample_age_ms": float(np.max(grid-ts[np.searchsorted(ts,grid,side="right")-1])/1000),
    "path_length_10hz_approx_m": float(np.linalg.norm(np.diff(p[idx],axis=0),axis=1).sum()),
    "net_displacement_m": float(np.linalg.norm(d[-1])),
    "last_displacement_xyz_m": d[-1].tolist(),
    "first_world_position_m": p[0].tolist(),
    "last_world_position_m": p[-1].tolist(),
    "speed_mean_mps": float(speed.mean()),
    "speed_max_mps": float(speed.max()),
    "quality_min": float(quality.min()),
    "quality_max": float(quality.max()),
    "geo_available_values": sorted({r.get("geo_available","missing") for r in rows}),
    "first_10s_displacement_xyz_m": d[np.searchsorted(ts, ts[0]+10000000,side="right")-1].tolist(),
    "quaternion_max_norm_error": float(np.max(np.abs(np.linalg.norm(q,axis=1)-1))),
}
font = FontProperties(fname="/mnt/c/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"axes.spines.top":False, "axes.spines.right":False, "font.size":11})
fig = plt.figure(figsize=(13,7.5), facecolor="white")
gs = fig.add_gridspec(2,2, width_ratios=[1.1,1], hspace=.4, wspace=.28)
ax = fig.add_subplot(gs[:,0])
ax.plot(d[idx,0],d[idx,1], color="#cbd5e1",linewidth=1)
sc = ax.scatter(d[idx,0],d[idx,1],c=t[idx],s=8,cmap="viridis")
ax.scatter(*d[0,:2],s=110,c="#16a34a",edgecolor="white",zorder=5)
ax.scatter(*d[-1,:2],s=110,c="#ef4444",marker="X",zorder=5)
ax.annotate("起点", d[0,:2], xytext=(8,8),textcoords="offset points",fontproperties=font)
ax.annotate("终点", d[-1,:2], xytext=(8,8),textcoords="offset points",fontproperties=font)
ax.set_aspect("equal",adjustable="datalim")
ax.set_xlabel("世界 X 方向位移（米）",fontproperties=font)
ax.set_ylabel("世界 Y 方向位移（米）",fontproperties=font)
ax.set_title("从上方看：眼镜走过的路径",fontproperties=font,pad=14)
ax.grid(alpha=.2)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal",pad=.12,shrink=.8)
cb.set_label("距首条 MPS 位姿的时间（秒）",fontproperties=font)
ax2 = fig.add_subplot(gs[0,1])
for j,c in enumerate(["#2563eb","#d97706","#7c3aed"]):
    ax2.plot(t[idx],d[idx,j],c=c,label="XYZ"[j])
ax2.set_title("三维位置变化（减去起点位置）",fontproperties=font)
ax2.set_ylabel("位移（米）",fontproperties=font)
ax2.legend(ncol=3,frameon=False)
ax2.grid(alpha=.2)
ax3 = fig.add_subplot(gs[1,1])
ax3.plot(t[idx],speed[idx],color="#0f766e")
ax3.set_title("MPS 输出的设备速度大小",fontproperties=font)
ax3.set_xlabel("时间（秒）",fontproperties=font)
ax3.set_ylabel("速度（米/秒）",fontproperties=font)
ax3.grid(alpha=.2)
fig.suptitle("Aria 2 实际 MPS 轨迹｜录制 20260803_150809",fontproperties=font,fontsize=18,y=.98)
fig.text(.5,.925,f"时长 {t[-1]:.1f} 秒   ·   10 Hz 轨迹折线长度约 {stats['path_length_10hz_approx_m']:.1f} 米   ·   起终点相距 {stats['net_displacement_m']:.1f} 米",
         fontproperties=font,ha="center",fontsize=12)
fig.text(.5,.025,"闭环 MPS 估计；设备原点，非 RGB 光心。仅平移显示原点，未旋转坐标轴；X/Y 不是东/北，Z 为竖直方向。",
         ha="center",fontproperties=font,fontsize=10,color="#475569")
fig.subplots_adjust(top=.83,bottom=.14,left=.075,right=.97)
fig.savefig(OUT/"trajectory_overview.png",dpi=160)
fig.savefig(OUT/"trajectory_overview.pdf")
plt.close(fig)
assert sha(SOURCE) == before
stats["source_unchanged"] = True
(OUT/"summary.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding="utf-8")
readme = f"""MPS 单条录制检查：20260803_150809

本次读取既有官方 MPS 闭环轨迹，未重新申请云端处理。
原始 CSV 完整复制，SHA256 与来源一致：{before}
源文件未修改。

时长：{t[-1]:.3f} 秒；原生 {len(rows)} 条；显示/便读 CSV {len(idx)} 条。
轨迹折线长度（10 Hz 近似）：{stats['path_length_10hz_approx_m']:.3f} 米。
起终点三维距离：{stats['net_displacement_m']:.3f} 米。
原生最大时间间隔：{stats['max_native_gap_ms']:.3f} ms。
质量分数范围：{quality.min():.3f}–{quality.max():.3f}，它不是实测精度。

文件：
closed_loop_trajectory_original.csv：完整原始官方字段。
trajectory_10hz.csv：位置、相对起点位移、原始旋转四元数、速度、质量与设备时间戳。
trajectory_overview.png/pdf：路径、位移、速度。
summary.json：统计与校验。

10 Hz 选择在网格时刻或之前的最后一条原生样本，不插值、不平滑，不等同于 RGB 帧对齐。
elapsed_s 从首条 MPS 位姿计时，不代表 VRS/视频从零开始的时间。
位置/四元数仍是设备坐标系原点的世界位姿；尚未应用设备到 RGB 相机外参。
图仅减去第一个位置，保留世界轴方向，X/Y 不代表东/北。
闭环为离线优化的参考轨迹，不视作在线实时输出或独立实测 GT。
原文件另含 geo_available/ECEF 字段；这些字段存在不能量化 GPS 对本条轨迹的贡献。

官方资料：
https://facebookresearch.github.io/projectaria_tools/gen2/ark/mps
https://facebookresearch.github.io/projectaria_tools/gen2/technical-specs/mps/data_formats/slam/mps_trajectory
https://facebookresearch.github.io/projectaria_tools/gen2/ark/aria_studio/recordings
Aria Studio 的 Recordings on Computer 中，选择这条已处理录制，在操作菜单中打开 Rerun 可视化，
可以同时查看 VRS 图像、MPS 轨迹和点云。本次导出了静态图，没有启动官方交互界面。
"""
(OUT/"说明.txt").write_text(readme,encoding="utf-8-sig")
print(json.dumps(stats,indent=2,ensure_ascii=False))
