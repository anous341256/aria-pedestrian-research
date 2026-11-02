"""Package completed paired results without selecting or changing a model."""
from prepare import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
    for f in ["training_complete.json","verification.json","independent_audit.json","translation_audit.json","presentation/verification.json"]:
        assert read(OUT/f)["passed"]
    c=read(OUT/"comparison.json");results=c["results"]
    labels={"no_translation":"匹配对照：无平移","with_translation":"加入历史平移","previous_saved":"此前保存版本","hold_last":"保持最后位置",
        "new_person_control_camera":"新行人头＋对照相机头","control_person_new_camera":"对照行人头＋新相机头"}
    lines=["历史 MPS 平移输入消融 — 2026-10-20",
        "设置：原始分离双分支；过去1.5秒16点，预测未来0.5秒5点。117窗口、6个录制测试折、3种子17/29/43。",
        "相机训练仍为原27185窗口；36个相机＋36个行人模型。无新增标签/无PPT修改/不替换原选定模型。",
        "",
        "新增输入：过去RGB相机位置相对当前RGB相机位置的三维位移（米），表达在当前RGB相机朝向下，同时输入两个GRU。",
        "MPS设备平移＋旋转经已存factory外参得到RGB光心位置；未新增未来平移预测，也未进行平移几何补偿。",
        "原旋转补偿、未来重投影和训练监督保持一致。剩余轨迹仍包含平移/视差/行人运动/标签噪声。",
        "两组均扩展相机输入3→6、行人输入6→9；对照将新增3通道置零；两组新增GRU权重列都从零开始。",
        "两组初始化、批次、行人增强与随机噪声完全配对。模型仅按验证集选取；没有测试调参。",
        "旧保存版本作为辅助参照：输入维度改变会改变默认随机初始化顺序，因此隔离平移信息作用应比较本次匹配两组。",
        "",
        "方法 | ADE(px) | FDE(px) | 录制等权ADE(px)"]
    for arm in ["no_translation","with_translation","previous_saved","hold_last","new_person_control_camera","control_person_new_camera"]:
        r=results[arm]
        lines.append(f"{labels[arm]} | {r['window_mean']['ADE']:.4f} | {r['window_mean']['FDE']:.4f} | {r['recording_equal']['ADE']:.4f}")
    lines+=["","分支诊断：方法 | 剩余轨迹ADE(px) | 相机平均角误差(deg) | GT相机＋预测行人ADE | 预测相机＋GT行人ADE"]
    for arm in ["no_translation","with_translation","previous_saved"]:
        r=results[arm]["window_mean"]
        lines.append(f"{labels[arm]} | {r['residual_ADE']:.4f} | {r['camera_mean_deg']:.4f} | {r['gt_camera_ADE']:.4f} | {r['gt_person_ADE']:.4f}")
    contrast=c["contrasts"]["no_translation"]
    a=results["with_translation"];b=results["no_translation"]
    seeds=[s for s in ["17","29","43"] if a["per_seed"][s]["ADE"]<b["per_seed"][s]["ADE"]]
    lines+=["",f"加入平移相对匹配对照ADE变化：{contrast['ADE_difference']:+.4f}px（{contrast['relative_ADE_percent']:+.3f}%）。",
        f"改善窗口：{contrast['improved_windows']}/117；改善录制：{contrast['improved_recordings']}/6；改善种子：{len(seeds)}/3，{seeds}。",
        f"录制聚类bootstrap差值95%区间（6录制，探索性）：{contrast['recording_bootstrap_95']}px。",
        "", "种子 | 无平移ADE | 有平移ADE | 无平移FDE | 有平移FDE"]
    for s in ["17","29","43"]:
        x=b["per_seed"][s];y=a["per_seed"][s]
        lines.append(f"{s} | {x['ADE']:.4f} | {y['ADE']:.4f} | {x['FDE']:.4f} | {y['FDE']:.4f}")
    lines+=["","录制 | 无平移ADE | 有平移ADE"]
    for rec in a["per_recording"]:
        lines.append(f"{rec} | {b['per_recording'][rec]['ADE']:.4f} | {a['per_recording'][rec]['ADE']:.4f}")
    lines+=["",f"稀疏非重叠子集ADE：无平移{b['sparse']['ADE']:.4f}，有平移{a['sparse']['ADE']:.4f}。",
        f"训练计算耗时合计：{c['training_seconds']:.1f}秒（不含提取、审计、渲染）。",
        "", "有效性与边界：",
        "输入平移只来自预测起点及以前的MPS样本；记录起始处允许首条MPS对较早历史图像的<=30ms对齐，仍严格早于预测起点。详见translation_extraction.json。",
        "原始输入姿态为离线闭环MPS，仍不能等同于在线实时传感器评测。标签为经过核查的SAM参考。",
        "72个checkpoint预测重放、移除未来数据重放、无平移组忽略任意平移修改均通过。所有同一时刻目标共用完全一致的相机预测。",
        "手工四元数公式独立重算全部相机历史平移；117例物理截断原始MPS后提取不变。独立SciPy几何与分支角误差检查通过。",
        "固定原始9案例/seed17，保留失败例，左右对照：无平移／有平移／保持最后位置。视频37.8秒、378帧全部解码通过。",
        "详见comparison.json、verification.json、independent_audit.json、translation_audit.json及presentation/verification.json。",
        "", "结论边界：本次仅检验该输入方式、时域与训练预算下的预测收益；不能据此证明平移已被分离、深度无用，或所有场景具有一致收益。"]
    report=ROOT/"reports/translation_forecast_20261020";report.mkdir(parents=True,exist_ok=True)
    (report/"RESULTS.txt").write_text("\n".join(lines),encoding="utf-8-sig")
    font=FontProperties(fname="/mnt/c/Windows/Fonts/msyh.ttc")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,["ADE","residual_ADE","camera_mean_deg"],["最终画面 ADE（像素）","剩余轨迹 ADE（像素）","相机旋转平均误差（度）"]):
        vals=[b["window_mean"][key],a["window_mean"][key]]
        ax.bar([0,1],vals,color=["#64748b","#0f766e"],width=.58)
        ax.set_xticks([0,1],["无平移","加入历史平移"],fontproperties=font)
        ax.set_title(title,fontproperties=font)
        for i,v in enumerate(vals):ax.text(i,v*1.025,f"{v:.2f}",ha="center")
        ax.set_ylim(0,max(vals)*1.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("1.5 秒历史 → 0.5 秒未来｜117 窗口 × 3 种子｜数值越低越好",fontproperties=font,fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT/"presentation/comparison.png",dpi=160)
    fig.savefig(OUT/"presentation/comparison.pdf")
    plt.close(fig)
    (OUT/"presentation/说明.txt").write_text("固定9例视频：fixed_9_translation_comparison.mp4\n左：本次匹配无平移对照；中：加入历史平移；右：保持最后位置。\n三组为同一预测时刻和未来帧，白色为SAM参考中心、橙色为预测中心。\n前1.5秒历史、后0.5秒预测；半速展示。seed17，固定旧9例，非按新成绩挑选。\n总量统计见comparison.png和reports/translation_forecast_20261020/RESULTS.txt。\n",encoding="utf-8-sig")
    hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.name not in ["delivery_hashes.json","delivery_complete.json"]}
    write(OUT/"delivery_hashes.json",hashes)
    write(OUT/"delivery_complete.json",dict(passed=True,files=len(hashes),report=str(report/"RESULTS.txt")))
    print("\n".join(lines),flush=True)
if __name__=="__main__":main()
