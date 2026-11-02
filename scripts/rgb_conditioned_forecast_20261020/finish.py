from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
def main():
    for file in ["training_complete.json","verification.json","independent_audit.json","learning_diagnostics.json","presentation/verification.json"]:
        assert read(OUT/file)["passed"]
    c=read(OUT/"comparison.json");r=c["results"];names={"no_rgb":"A 无图像","local_rgb":"B 目标局部","scene_rgb":"C 全画面","both_rgb":"D 局部＋全画面","previous_separated":"旧双分支＋历史平移","hold_last":"保持最后位置"}
    lines=["历史 RGB 条件轨迹预测 — 第一版，2026-10-20",
      "设置：同117窗口、六个录制测试折、种子17/29/43。过去1.5秒16个点/帧，预测未来0.5秒5个图像中心。",
      "研究问题：在已有历史轨迹/框大小及相机旋转/平移之外，图像是否提供额外的预测信息？",
      "",
      "架构：历史数值时序 -> self-attention -> 同时刻局部/全画面视觉 cross-attention -> 时序 self-attention -> 单个未来图像位移头。",
      "预训练DINOv2 ViT-S/14被冻结；全画面448×336，目标框外扩1.5倍后裁剪224×224；各池化成4×4空间patch，保留原图空间位置。",
      "全帧特征跨目标共享；372张唯一历史画面，1872个目标历史时刻。像素输入不包含画出的标注框/轨迹。",
      "四组注册结构、初始化和抽样一致；禁用视觉内容清零并遮罩，保留固定空token；全部拥有相同轨迹/框宽高及相机信息。",
      "每组18次训练，共72次，187786可训练参数。240更新/batch48/AdamW lr0.001；录制均衡抽样。",
      "四组都不做数据增强，以保持缓存裁剪与数值轨迹一致；验证图像ADE每20步选择，初始保持最后位置(step0)也可入选。",
      "不使用未来图像、框、相机旋转或平移作为输入；不执行未来几何重投影，没有独立相机/残差预测头。",
      "",
      "方法 | ADE(px) | FDE(px) | 录制等权ADE | 稀疏窗口ADE"]
    for a in ARMS+["previous_separated","hold_last"]:
        z=r[a];lines.append(f"{names[a]} | {z['window_mean']['ADE']:.4f} | {z['window_mean']['FDE']:.4f} | {z['recording_equal']['ADE']:.4f} | {z['sparse']['ADE']:.4f}")
    lines+=["","相对新A无图像组："]
    for a in ARMS[1:]:
        z=c["contrasts"][a]
        seeds=sum(r[a]["per_seed"][s]["ADE"]<r["no_rgb"]["per_seed"][s]["ADE"] for s in ["17","29","43"])
        lines.append(f"{names[a]}：ADE差{z['ADE_difference']:+.4f}px，{z['relative_ADE_percent']:+.2f}%；改善{z['improved_windows']}/117窗口、{z['improved_recordings']}/6录制、{seeds}/3种子；录制等权探索性bootstrap95%区间{z['recording_bootstrap_95']}px。")
    lines+=["","各随机种子：种子 | 组别 | ADE | FDE"]
    for s in ["17","29","43"]:
        for a in ARMS:
            z=r[a]["per_seed"][s];lines.append(f"{s} | {names[a]} | {z['ADE']:.4f} | {z['FDE']:.4f}")
    lines+=["","各录制：录制 | A无图像 | B局部 | C全画面 | D局部和全画面"]
    for rec in r["no_rgb"]["per_recording"]:
        lines.append(rec+" | "+" | ".join(f"{r[a]['per_recording'][rec]['ADE']:.4f}" for a in ARMS))
    diag=read(OUT/"learning_diagnostics.json")["fit_equal_averages"]
    lines+=["","补充训练诊断（18次fit等权；不是测试分数）：组别 | 训练ADE | 验证ADE"]
    for a in ARMS:
        lines.append(f"{names[a]} | {diag[a]['train_ADE']:.4f} | {diag[a]['validation_ADE']:.4f}")
    lines+=["",
      "本次具体结论：局部RGB与无图像总体几乎持平，全画面及联合输入总体更差；四个新模型的总体ADE都没有超过保持最后位置。",
      "全画面组训练ADE低于无图像组，但未见录制上的测试ADE更高，提示当前视觉接法存在泛化问题；不能仅凭此区分数据规模、场景差异或特征/架构的原因。",
      "多次fit按验证规则选回初始保持最后位置，说明本预算下新直接融合模型的学习收益并不稳定。不能把失败简单归因于像素信息无用。",
      f"验证选择初始保持最后位置的次数（每组18次）：{c['step0_selected']}。",
      f"累计训练及各fit验证/保存耗时：{c['training_seconds']:.1f}秒；视觉提取{read(OUT/'extraction_complete.json')['seconds']:.1f}秒，均不含全部交互/开发/审计/展示耗时。",
      "",
      "解释边界：",
      "新A/B/C/D之间比较隔离本方案的视觉信息；旧双分支使用了额外全量相机预训练，模型、监督、增强也不同，不能把新D与旧双分支的差异全部归功于像素。",
      "数据仅117窗口/六录制，离线SAM/MPS参考；没有扩充训练数据或按测试结果调参，不能据此断言视觉信息普遍无用/有效。",
      "即便某一组平均改善，也需同时检查录制、种子、稀疏统计与失败案例。六录制bootstrap仅是探索性统计。",
      "本版冻结的逐帧视觉编码器通过后续时序模块结合；尚未比较视频预训练、不同分辨率或视觉微调。",
      "本版输出整体图像位置，不能做原双分支GT替换式运动归因；注意力权重也不是因果解释。",
      "",
      "验证与展示：72 checkpoint重放、移除未来数值数组重放、54禁用视觉输入扰动检查、配对初始化/批次核验通过。",
      "独立重算全部预测误差与1872个历史裁剪/时间戳/特征对应；源文件哈希复核通过。",
      "固定原9案例seed17六面板视频：A/B/C/D/旧双分支/保持最后位置；37.8秒378帧完整解码，保留失败案例。",
      "现有基线、标签、PPT保持原样。本实验保存为独立研究分支，不自动替换最优模型。"]
    report=ROOT/"reports"/OUT.name;report.mkdir(parents=True,exist_ok=True)
    (report/"RESULTS.txt").write_text("\n".join(lines),encoding="utf-8-sig")
    font=FontProperties(fname="/mnt/c/Windows/Fonts/msyh.ttc")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    arms=ARMS+["previous_separated","hold_last"]
    for ax,k in zip(axes,["ADE","FDE"]):
        vals=[r[a]["window_mean"][k] for a in arms]
        ax.bar(np.arange(6),vals,color=["#64748b","#2563eb","#0891b2","#0f766e","#a16207","#94a3b8"])
        ax.set_xticks(np.arange(6),[names[a] for a in arms],fontproperties=font,rotation=25,ha="right")
        ax.set_title(k+"（像素，越低越好）",fontproperties=font);ax.grid(axis="y",alpha=.15)
        for i,v in enumerate(vals):ax.text(i,v+2,f"{v:.1f}",ha="center",fontsize=9)
        ax.set_ylim(0,max(vals)*1.18)
    fig.tight_layout();fig.savefig(OUT/"presentation/comparison.png",dpi=180);fig.savefig(OUT/"presentation/comparison.pdf");plt.close(fig)
    (OUT/"presentation/说明.txt").write_text("固定原始九案例，不根据本次结果选例。seed17。\n第一排：无图像、目标局部、全画面；第二排：局部＋全画面、旧双分支＋平移、保持最后位置。\n白点是SAM参考中心，橙点是预测；先播放1.5秒历史，再播放0.5秒固定预测，整体半速37.8秒。\n完整统计请看 reports/rgb_conditioned_forecast_20261020/RESULTS.txt。\n",encoding="utf-8-sig")
    write(OUT/"delivery_complete.json",dict(passed=True,report=str(report/"RESULTS.txt"),video=str(OUT/"presentation/fixed_9_rgb_comparison.mp4"),
      training_fits=72,selected_baseline_replaced=False))
    files=[p for p in OUT.rglob("*") if p.is_file() and p.name!="delivery_hashes.json" and "feature_cache" not in str(p)]
    write(OUT/"delivery_hashes.json",{str(p.relative_to(ROOT)):sha(p) for p in files})
    print(str(report/"RESULTS.txt"),flush=True)
if __name__=="__main__":main()
