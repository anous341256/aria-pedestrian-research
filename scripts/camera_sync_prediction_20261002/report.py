from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    stats=read(OUT/'summary.json');a=stats['main']['comp_dual'];c=stats['condition_effect']['comp_dual']
    data=np.load(OLD/'dataset.npz');curves={arm:[] for arm in ARMS}
    for path in OUT.glob('fold_*/seed_*/comp_dual/predictions.npz'):
        z=np.load(path);truth=data['xy'][z['indices'],6:]
        for arm in ARMS:
            error=np.linalg.norm(z[arm]-truth,axis=-1);error[~np.isfinite(error)|(z[arm+'_depth']<=1e-6)]=32000.
            curves[arm].append(error)
    curves={arm:np.concatenate(v).mean(0) for arm,v in curves.items()}
    write(OUT/'horizon_errors.json',{arm:v.tolist() for arm,v in curves.items()})
    dest=OUT/'presentation';dest.mkdir(exist_ok=True);font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.4),layout='constrained')
    chosen=['old_old','old_new','new_new'];colors=['#6c7e96','#bb7858','#15796f']
    for arm,color in zip(chosen,colors):axes[0].plot(np.arange(1,16)*.1,curves[arm],label=LABELS[arm],color=color,lw=2)
    axes[0].legend(prop=font);axes[0].set_xlabel('Prediction horizon (s)');axes[0].set_ylabel('Position error (pixels)');axes[0].grid(alpha=.2)
    axes[0].set_title('每个未来时刻的位置误差',fontproperties=font)
    x=np.arange(2);width=.25
    for j,(arm,color) in enumerate(zip(chosen,colors)):
        vals=[a[arm]['window_mean']['ADE15'],a[arm]['window_mean']['FDE15']]
        bars=axes[1].bar(x+(j-1)*width,vals,width=width,color=color)
        for bar,v in zip(bars,vals):axes[1].text(bar.get_x()+bar.get_width()/2,v+2,f'{v:.1f}',ha='center',fontsize=10)
    axes[1].set_xticks(x,['ADE 1.5 s','FDE 1.5 s']);axes[1].set_ylabel('Pixels');axes[1].set_title('平均误差与终点误差',fontproperties=font)
    fig.savefig(dest/'synchronized_results.png',dpi=170);fig.savefig(dest/'synchronized_results.pdf');plt.close(fig)
    table=['| 相机条件／重投影设置 | ADE 0.5 秒 | ADE 1.0 秒 | ADE 1.5 秒 | FDE 1.5 秒 | 录制等权 ADE |','|---|---:|---:|---:|---:|---:|']
    for arm in ['old_old','old_new','new_new','new_old']:
        r=a[arm];v=r['window_mean'];table.append(f"| {LABELS[arm]} | {v['ADE05']:.2f} | {v['ADE10']:.2f} | {v['ADE15']:.2f} | {v['FDE15']:.2f} | {r['recording_equal']['ADE15']:.2f} |")
    recordtable=['| 录制 | 旧版 ADE | 只换投影 ADE | 同步替换 ADE |','|---|---:|---:|---:|']
    for rec in a['old_old']['per_recording']:
        recordtable.append('| '+rec[-6:]+' | '+' | '.join(f"{a[arm]['per_recording'][rec]['ADE15']:.2f}" for arm in chosen)+' |')
    seedtable=['| 种子 | 旧版 ADE | 只换投影 ADE | 同步替换 ADE |','|---|---:|---:|---:|']
    for seed in ['17','29','43']:seedtable.append('| '+seed+' | '+' | '.join(f"{a[arm]['per_seed'][seed]['ADE15']:.2f}" for arm in chosen)+' |')
    no=stats['subsets']['nonoverlap']['comp_dual'];fut=stats['main']['comp_future'];paired=stats['paired']['comp_dual']
    tabletext='\n'.join(table);recordtext='\n'.join(recordtable);seedtext='\n'.join(seedtable)
    report=f'''# 同步替换未来相机条件和重投影的实验

2026 年 10 月 2 日。按用户批准，在固定人物预测模型的情况下，将新相机预测同时用于网络条件和最终重投影。使用对应测试录制留出训练的相机模型，保留原来的 117 个窗口、六个录制划分、三个种子与推理噪声。

**双分支模型同步替换后的 ADE 为 {a['new_new']['window_mean']['ADE15']:.2f} 像素，优于只换投影的 {a['old_new']['window_mean']['ADE15']:.2f}，但高于旧版本的 {a['old_old']['window_mean']['ADE15']:.2f}。** 1.5 秒终点误差却从旧版 {a['old_old']['window_mean']['FDE15']:.2f} 降到 {a['new_new']['window_mean']['FDE15']:.2f} 像素，因此平均轨迹误差与末端位置呈不同方向。

## 观看入口

- [原来九段固定案例的三组同步对比 37.8 秒](E:/New_research/outputs/camera_sync_prediction_20261002/presentation/fixed_9_examples.mp4)
- [按结果事后选择的改善和退化案例](E:/New_research/outputs/camera_sync_prediction_20261002/presentation/selected_gain_and_loss.mp4)
- [误差曲线](E:/New_research/outputs/camera_sync_prediction_20261002/presentation/synchronized_results.png)

从左到右为旧相机、只替换最终投影、同步替换网络条件与投影。使用原来相同九个固定案例及种子 17；表格平均三个种子，视频数值是种子 17 单次结果。白圈为 SAM 参考中心，橙圈为冻结预测中心。过去 6 点、未来 15 点，半速播放；预测起点后不再接收观测。另一个合集按平均分数选择极端案例，不能当作整体表现。

## 这次实际改了什么

人物网络权重完全固定。原模型条件包括历史相机旋转和预测的未来相机旋转；此次替换后，网络重新计算人物剩余轨迹，再使用同一份新相机预测重投影。设置四组：旧条件旧投影、旧条件新投影、新条件旧投影、新条件新投影。仅换条件一组用于区分两个替换位置的作用。

运行两个既有补偿变体（只预测未来、历史重建加未来预测），六个划分、三个种子，共 36 个固定人物模型。没有重新训练网络、调整阈值、改变窗口或挑选最优种子。推理保持五组固定噪声起点、十步 flow、五条输出取均值，历史重建头不参与未来推理。

相机使用 `camera_full_data_20261002/fold_*/seed_*/full/` 的对应留出预测；没有使用看过全部录制的 all_recordings_refit 模型。历史输入仍为离线 SAM 回查轨迹与离线 MPS 姿态，这不是在线跟踪或 RGB-only 结果。

## 双分支人物模型结果

单位为 2560×1920 图像中的像素，越小越好。ADE 是指定未来区间的平均中心距离，FDE 只看 1.5 秒末点。前三组是主要比较，最后一组为诊断。

{tabletext}

保持最后图像位置的 ADE 参照仍为 {stats['hold_baseline']['window_mean']['ADE15']:.2f} 像素。同步替换没有超过该基线。

![同步替换误差曲线](E:/New_research/outputs/camera_sync_prediction_20261002/presentation/synchronized_results.png)

相对旧版，同步替换的退化集中在 0.2～1.2 秒，1.3～1.5 秒的平均位置误差则略低。这说明单看最后一个点会遗漏中间预测路径的退化。

{recordtext}

{seedtext}

相对只换投影，同步替换在 {paired['new_new_vs_old_new']['improved']}/117 个窗口改善；相对旧版，在 {paired['new_new_vs_old_old']['improved']}/117 个窗口改善，均先平均同一窗口三个种子。减少重叠后的窗口 ADE：旧版 {no['old_old']['window_mean']['ADE15']:.2f}，只换投影 {no['old_new']['window_mean']['ADE15']:.2f}，同步替换 {no['new_new']['window_mean']['ADE15']:.2f}。该子集的比较方向与全窗口不同，因此不能把少量总体差异称为稳定收益。窗口存在相关性，不作独立样本显著性宣称。

## 条件替换的作用和限制

同步替换相对只换投影平均减少 {a['old_new']['window_mean']['ADE15']-a['new_new']['window_mean']['ADE15']:.2f} 像素 ADE，说明重新计算网络条件确实改变了结果，但不足以恢复旧版的平均误差。录制等权 ADE 从旧版 {a['old_old']['recording_equal']['ADE15']:.2f} 到同步替换 {a['new_new']['recording_equal']['ADE15']:.2f}，差别小于窗口等权统计，需要同时看两者。

条件改变后，统一朝向下的人物预测点平均移动 {c['mean_component_displacement_px']:.2f} 像素，不能说人物网络完全忽略相机输入。若仅作诊断，把两种人物输出都用参考未来旋转重投影，其 ADE 从 {c['old_condition_reference_camera_ADE']:.2f} 到 {c['new_condition_reference_camera_ADE']:.2f} 像素。该分支使用未来信息，不能当正式预测成绩。

只预测未来、无重建辅助的既有模型得到类似方向：旧版 {fut['old_old']['window_mean']['ADE15']:.2f}，只换投影 {fut['old_new']['window_mean']['ADE15']:.2f}，同步替换 {fut['new_new']['window_mean']['ADE15']:.2f}。因此本次现象不只出现在重建辅助版本。

结论是，旧网络条件与新投影不一致影响了结果，但不是此前 ADE 退化的充分解释。旧人物网络仍是在旧相机预测条件下训练的；同步推理替换没有检验在新条件下训练能否更好配合。下一项可比较固定新相机模型后重新训练人物预测器，但人物监督仍只有这批小规模轨迹，不能预设重训一定改善。本次到此完成同步替换检验，没有根据测试成绩进一步训练或调参。

## 核验和保存

旧模型重新推理与已保存输出逐值一致；修改未来标签及参考旋转，正式预测输出变化为 0。新条件与最终投影使用同一份对应划分的相机预测，未来推理不调用历史重建头。117 个来源文件哈希核对不变，旧代码、权重、预测与标注均保留。

[实验协议](E:/New_research/outputs/camera_sync_prediction_20261002/protocol.json) · [统计](E:/New_research/outputs/camera_sync_prediction_20261002/summary.json) · [核验](E:/New_research/outputs/camera_sync_prediction_20261002/verification.json) · [逐窗口指标](E:/New_research/outputs/camera_sync_prediction_20261002/metrics.json)
'''
    reportpath=ROOT/'reports/camera_sync_prediction_20261002/RESULTS.md';reportpath.parent.mkdir(parents=True,exist_ok=True);reportpath.write_text(report,encoding='utf8')
    print(reportpath);print('Horizon errors', {arm:curves[arm].round(2).tolist() for arm in chosen})


if __name__=='__main__':main()
