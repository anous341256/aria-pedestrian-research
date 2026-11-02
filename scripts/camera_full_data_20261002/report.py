"""Figures and a reproducible Chinese report, generated from saved test metrics."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    inventory=read(OUT/'inventory.json');scores=read(OUT/'camera_metrics.json')
    cfg=read(OUT/'protocol.json');folds=read(OUT/'folds.json');rep=read(OUT/'fixed_person_reprojection.json')
    names=inventory['record_names'];a=scores['full_recording_test'];b=scores['original_person_windows']
    dest=OUT/'presentation';dest.mkdir(exist_ok=True)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    order=['frozen_orientation','constant_angular_velocity','old_small','small_matched','full']
    labels={'frozen_orientation':'保持当前朝向','constant_angular_velocity':'恒定角速度','old_small':'上一轮小数据模型','small_matched':'小数据 相同训练预算','full':'完整录制数据模型'}
    colors={'frozen_orientation':'#888888','constant_angular_velocity':'#c49736','old_small':'#bd6d54','small_matched':'#6777a3','full':'#14776d'}
    t=np.arange(1,16)*.1
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for key in order:
        axes[0].plot(t,a[key]['curve_degrees'],label=labels[key],color=colors[key],lw=2)
    axes[0].set_xlabel('Prediction horizon (s)');axes[0].set_ylabel('Rotation error (degrees)')
    axes[0].set_title('六段留出录制等权统计',fontproperties=font);axes[0].grid(alpha=.2);axes[0].legend(prop=font,fontsize=9)
    vals=[a[k]['recording_equal_mean_degrees'] for k in order]
    axes[1].bar(np.arange(5),vals,color=[colors[k] for k in order])
    for i,v in enumerate(vals):axes[1].text(i,v+.15,f'{v:.2f}',ha='center')
    axes[1].set_xticks(np.arange(5),[labels[k].replace(' 相同','\n相同') for k in order],fontproperties=font,rotation=20,ha='right')
    axes[1].set_ylabel('Mean rotation error (degrees)');axes[1].set_title('未来 1.5 秒内平均旋转误差',fontproperties=font)
    fig.savefig(dest/'camera_results.png',dpi=170);fig.savefig(dest/'camera_results.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    tested=[names.index(f['test']) for f in folds];x=np.arange(6)
    for j,key in enumerate(['frozen_orientation','small_matched','full']):
        vals=[a[key]['per_recording'][str(i)]['mean_degrees'] for i in tested]
        axes[0].bar(x+(j-1)*.24,vals,width=.24,label=labels[key],color=colors[key])
    axes[0].set_xticks(x,[names[i][-6:] for i in tested]);axes[0].set_ylabel('Rotation error (degrees)');axes[0].set_title('不同录制的表现',fontproperties=font)
    bins=['final_rotation_under_5deg','final_rotation_5_to_15deg','final_rotation_at_least_15deg'];x=np.arange(3)
    for j,key in enumerate(['frozen_orientation','small_matched','full']):
        vals=[a[key]['motion_bins'][z]['recording_equal_mean_degrees'] for z in bins]
        axes[1].bar(x+(j-1)*.24,vals,width=.24,label=labels[key],color=colors[key])
    axes[1].set_xticks(x,['<5 degrees','5-15 degrees','>=15 degrees']);axes[1].set_ylabel('Rotation error (degrees)');axes[1].set_xlabel('Actual final rotation from the prediction cutoff')
    axes[1].set_title('按实际未来转动幅度分组',fontproperties=font)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside upper center',ncol=3,prop=font)
    fig.savefig(dest/'recordings_and_turns.png',dpi=170);plt.close(fig)
    def table(source):
        rows=['| 方法 | 平均旋转误差 ° | 0.5 秒终点 ° | 1.0 秒终点 ° | 1.5 秒终点 ° |','|---|---:|---:|---:|---:|']
        for k in order:
            v=source[k];c=v['curve_degrees']
            rows.append(f"| {labels[k]} | {v['recording_equal_mean_degrees']:.3f} | {c[4]:.3f} | {c[9]:.3f} | {c[14]:.3f} |")
        return '\n'.join(rows)
    full=a['full']['recording_equal_mean_degrees'];small=a['small_matched']['recording_equal_mean_degrees'];frozen=a['frozen_orientation']['recording_equal_mean_degrees']
    gain=(small-full)/small*100;gainhold=(frozen-full)/frozen*100
    recordtable=['| 录制 | 保持朝向 ° | 同预算小数据 ° | 完整数据 ° |','|---|---:|---:|---:|']
    for i in tested:recordtable.append('| '+names[i][-6:]+' | '+' | '.join(f"{a[k]['per_recording'][str(i)]['mean_degrees']:.3f}" for k in ['frozen_orientation','small_matched','full'])+' |')
    datatable=['| 录制 | 有效姿态秒数 | 窗口数 | 状态 |','|---|---:|---:|---|']
    for e in inventory['recordings']:
        datatable.append(f"| {e['recording'][-6:]} | {e.get('valid_seconds',0):.1f} | {e.get('windows',0)} | {'可用' if e['used'] else '缺少闭环 MPS，未使用'} |")
    training_range=[len(f['train_indices']) for f in folds];small_range=[len(f['small_train_indices']) for f in folds]
    recordtable_text='\n'.join(recordtable); datatable_text='\n'.join(datatable)
    report=f'''# 扩大相机旋转预测训练数据的实验结果

2026 年 10 月 2 日。按用户要求，将相机预测训练从十个短片段扩展到本地可用完整 Aria 录制。保持上一轮相机网络不变，输入过去 0.5 秒的 6 个旋转，预测未来 1.5 秒的 15 个旋转，不需要行人标注。

在六段留出录制上，完整数据模型的录制等权平均旋转误差为 **{full:.3f}°**，同训练预算小数据对照为 **{small:.3f}°**，保持当前朝向为 **{frozen:.3f}°**。相对两者的误差变化分别为降低 **{gain:.1f}%** 和降低 **{gainhold:.1f}%**。模型与数据均未按测试分数继续调整。

## 数据规模和划分

本地有 29 段原始 VRS，28 段具备闭环 MPS 及 RGB 时间戳。可用姿态累计约 **{inventory['valid_seconds']/60:.2f} 分钟**，生成 **{inventory['total_windows']:,} 个窗口**，原实验为 330 个窗口。162338 录制缺少闭环姿态，本轮未纳入。窗口每 0.1 秒滑动，存在高度重叠，窗口数不代表独立样本数。

相机旋转由 MPS 设备旋转乘该录制的 RGB 相机标定外参得到，与旧实验十个短片段逐矩阵核对一致。只使用质量分数至少 0.5、最近姿态时间差不超过 30 毫秒、窗口内不跨 graph_uid、时间间隔正常的窗口；四段录制开头的重复时间戳所涉及窗口已排除。数据为同一天采集，跨日期及佩戴者泛化未在本轮检验。

保留上一轮六个录制作为轮流留出的测试组。每轮一个完整录制验证、一个完整录制测试，其余 **26 个完整录制**训练。训练集每轮 {min(training_range):,}–{max(training_range):,} 个窗口；同预算小数据对照每轮 {min(small_range)}–{max(small_range)} 个窗口，来自原来的四个训练录制短片段。两个对照共用完整验证录制，因此数据量比较没有同时改变验证集。

六段完整测试录制共 **{a['full']['windows']:,} 个窗口**。同时提供完整两秒窗口互不重叠的稀疏子集；该子集仍来自同一天和六个录制。

## 模型和训练

网络保持 GRU 输入 3 维、隐藏 32 维，后接 32→64→45 的全连接层，合计 **8,589 参数**。将上一轮权重加载到本版实现后，相同输入输出最大差异为 0。未引入 RGB、眼动、相机平移或其他模型，也未改变预测时间范围。

完整数据与小数据对照各训练六个划分、三个种子，共 36 个模型。每个模型固定 2,000 次更新、batch 256；训练集完整随机遍历并重复，所有训练窗口实际至少使用一次。AdamW 学习率 0.001、权重衰减 0.01、梯度裁剪 1。损失为旋转向量均方误差，每 100 次更新用验证旋转向量误差选模。三种子分别评分后平均，没有选取测试表现最好的种子。

上一轮保存的小数据模型作为额外参照；其训练预算和验证范围与本轮不同，不能将该参照的差异全部归因于训练数据。恒角速度基线在过去 6 点相对旋转向量上做过原点的最小二乘拟合，再外推未来旋转。

历史姿态是离线 MPS 参考；未来姿态只用于训练目标和评分。本轮是姿态辅助的相机旋转预测，不是 RGB-only，也未证明闭环历史姿态可以实时获得。

## 完整测试录制上的旋转预测

误差为预测旋转与参考旋转间 SO(3) 最短角距离。先对种子平均、再对各录制内窗口平均，最后六录制等权，避免长录制支配结果。平均误差覆盖未来全部 15 点；各终点列只对应该时刻。

{table(a)}

![完整录制测试结果](E:/New_research/outputs/camera_full_data_20261002/presentation/camera_results.png)

{recordtable_text}

稀疏窗口的录制等权平均误差：保持朝向 {a['frozen_orientation']['sparse']['recording_equal_mean_degrees']:.3f}°，同预算小数据 {a['small_matched']['sparse']['recording_equal_mean_degrees']:.3f}°，完整数据 {a['full']['sparse']['recording_equal_mean_degrees']:.3f}°。完整数据模型三个种子的录制等权平均误差为 {', '.join(f'{v:.3f}°' for v in a['full']['per_seed_recording_equal_mean'])}。

![分录制和转动幅度结果](E:/New_research/outputs/camera_full_data_20261002/presentation/recordings_and_turns.png)

转动幅度分组按实际未来 1.5 秒末相对于预测起点的旋转角划分，仅用于评分分析，不是预测输入。每个分组内部仍按存在该类窗口的录制等权。

完整数据模型在六个录制上均优于同预算小数据及保持朝向，三个种子的整体方向也一致。但优势依赖场景：未来末端转动小于 5° 的组中，保持朝向误差为 {a['frozen_orientation']['motion_bins']['final_rotation_under_5deg']['recording_equal_mean_degrees']:.3f}°，完整数据为 {a['full']['motion_bins']['final_rotation_under_5deg']['recording_equal_mean_degrees']:.3f}°，仍是保持朝向更好。1.5 秒末端的整体误差仍有 {a['full']['recording_equal_final_degrees']:.2f}°，不能称为已经解决未来头部运动预测。

## 原来的人物预测窗口

以下仍是相机旋转误差，在原来 117 个人物窗口上评分；这些窗口不是重新筛选的相机测试样本，同一画面有不同人物时相机窗口可能重复。

{table(b)}

为和上一轮的 7.97° 直接对应，若按 117 窗口等权：上一轮相机为 {b['old_small']['window_mean_degrees']:.3f}°，完整数据为 {b['full']['window_mean_degrees']:.3f}°，保持朝向为 {b['frozen_orientation']['window_mean_degrees']:.3f}°。不要与上表的录制等权数值混用。

另做了固定人物预测、仅替换最终重投影相机旋转的诊断。原补偿双分支的窗口平均 ADE 从 {rep['summary']['comp_dual']['old_small']['window_ADE']:.1f} 变为 {rep['summary']['comp_dual']['full']['window_ADE']:.1f} 像素。人物预测及其原相机条件保持不变，这不是重训后的完整轨迹模型成绩，也不能代表完整替换所有相机条件的效果。

## 保存的模型和复现入口

交叉验证模型及逐窗预测保存在 `outputs/camera_full_data_20261002/fold_*/seed_*/`。另保存三个使用全部 28 段数据的模型在 [全量重训目录](E:/New_research/outputs/camera_full_data_20261002/all_recordings_refit)。它们的更新次数取相应种子六折验证选出步数的中位数，并至少完成一次全数据遍历；没有用它们报告上面的留出分数。后续若在已参与该全量重训的录制上测试，不属于未见录制泛化。

原始数据、上一轮模型和行人预测器均已保留。协议：[protocol.json](E:/New_research/outputs/camera_full_data_20261002/protocol.json)；核验：[verification.json](E:/New_research/outputs/camera_full_data_20261002/verification.json)；完整统计：[camera_metrics.json](E:/New_research/outputs/camera_full_data_20261002/camera_metrics.json)；重投影诊断：[fixed_person_reprojection.json](E:/New_research/outputs/camera_full_data_20261002/fixed_person_reprojection.json)。

## 逐录制数据清单

{datatable_text}
'''
    path=ROOT/'reports/camera_full_data_20261002/RESULTS.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(report,encoding='utf8')
    print(path)


if __name__=='__main__':main()
