"""Package results without changing selected baseline or source labels."""
from forecast_common import *

def main():
 for name in ['training_complete.json','verification.json','independent_audit.json','presentation/verification.json']:assert read(OUT/name)['passed']
 c=read(OUT/'comparison.json');p=read(OUT/'preflight.json');meta=read(BASE/'samples.json');rows=read(OUT/'metrics.json');r=c['results'];arms=['saved_translation']+ARMS;names=['原版保存权重','原数据继续微调','质量切段＋修剪数据微调'];details=[]
 for i,m in enumerate(meta):
  means={a:float(np.mean([v['ADE'] for v in rows if v['arm']==a and v['index']==i])) for a in arms};details.append(dict(index=i,key=m['key'],recording=m['recording'],ADE=means,trimmed_minus_saved=means[ARMS[1]]-means['saved_translation']))
 dif=np.array([v['trimmed_minus_saved'] for v in details]);stats=dict(improved=int((dif<-1e-10).sum()),unchanged=int((abs(dif)<=1e-10).sum()),worsened=int((dif>1e-10).sum()));write(OUT/'window_changes.json',dict(counts=stats,windows=details))
 lines=['质量切段＋修剪数据：预测模型微调实验（2026-10-28）','',
 '结论：本次裁剪数据微调没有带来整体提升。旧版继续保留为原有参考，新权重独立保存。',
 '裁剪版 ADE 89.7863px，原版88.2420px：增加1.5443px（变差1.7501%）。',
 '对照组使用原训练数据继续微调，ADE88.4597px；裁剪版相对该对照变差1.4997%。',
 '不能据此断言裁剪一定无效，或已经证明复杂场景被删除是退步原因。此实验同时改变了样本数量、重叠程度、身份关联结果与场景分布。','',
 '1. 模型与实验设置',
 '沿用 translation_forecast_20261020 的 with_translation 保存权重：原始分离双分支＋历史相机平移特征。',
 '输入过去1.5秒的16点，预测未来0.5秒的5点；相机分支冻结，行人分支全部参数微调。',
 '仍为旋转补偿与几何重投影；平移只作为历史输入特征，没有深度输入或完整三维补偿。',
 '两组从各自相同fold/seed旧权重开始，均240步、batch48、AdamW lr0.0001、wd0.01、clip1。',
 '无输入增强；原有未来flow＋0.5历史重建flow损失，5组固定噪声、10步推理。',
 '使用原验证集残差ADE，每20步选择模型，允许保留step0；不根据测试分数选择。',
 '按录制均衡采样；两组数据/录制集合不同，不能声称实际采样窗口完全相同。',
 '原数据组选择step0 11/18次，裁剪组9/18次。所有36次训练均执行完成；step0代表继续训练未改善该折验证指标。',
 '这是旧模型的继续微调，不是删除旧训练历史后从头训练。','',
 '2. 数据与划分',
 '裁剪版只使用已锁定track_trimming_20261028的412个21帧窗口，17段连续轨迹、8个视频、5次录制；未与原117合并。',
 '412窗高度重叠，相同段按21帧步长仅30窗；其中74窗与旧117的中心点与帧完全一致。',
 '裁剪仅来自旧几何关联结果，没有叠加partial_appearance新外观版本。',
 '原验证/测试117窗、6次录制、6折、种子17/29/43全部不变；每折排除测试及验证录制的裁剪窗口。',
 '新训练窗口沿用原RGB参考内参/朝向，RGB光心MPS位置取曝光时刻以前的缓存样本。',
 '原参考朝向与原始last-before MPS朝向并不完全相同；保持原预测约定，未偷偷替换。',
 '质量筛选可观察整条离线轨迹，但网络预测输入只含历史；这不等于在线标注/MPS管线。',
 '测试集仍保留原来的较难片段，不将裁剪后较简单的数据当成新测试集。','训练窗口按录制分布：']
 for g,n in p['per_recording'].items():lines.append(f'  {g}: {n}窗')
 lines+=['每折训练窗数（原数据／裁剪数据）：']
 for f in p['fold_training_counts']:lines.append(f'  fold{f["fold"]}: {f["original_train_count"]} / {f["trimmed_train_count"]}; test={f["test"]}; validation={f["validation"]}')
 lines+=['','3. 同一测试集结果（像素，越低越好）','方法 | ADE | FDE | 行人残差ADE | 录制等权ADE | 减少窗口重叠ADE']
 for a,n in zip(arms,names):
  s=r[a];w=s['window_mean'];lines.append(f'{n} | {w["ADE"]:.4f} | {w["FDE"]:.4f} | {w["residual_ADE"]:.4f} | {s["recording_equal"]["ADE"]:.4f} | {s["sparse"]["ADE"]:.4f}')
 lines+=['保持最后位置 | ADE106.6624 | FDE152.7229',
 '三种模型的相机预测完全相同，平均角误差均4.3114°；本轮差异来自微调后的行人分支及其重投影结果。',
 f'按三个种子的窗口均值，裁剪版相对原版：改善{stats["improved"]}窗，持平{stats["unchanged"]}窗，变差{stats["worsened"]}窗。',
 '六次录制中裁剪版相对原版：3次变差，3次完全不变，0次改善。三个种子的总体ADE均变差。',
 '裁剪版对原数据微调：42/117窗改善，1/6录制改善；FDE略低0.1762px，但ADE更高。',
 '按六次录制等权bootstrap的裁剪减原版ADE区间[0.3921,6.1561]px；裁剪减原数据微调[0.3726,3.7021]px。',
 '仅六个录制群组，为探索性描述，不把大量重叠窗口视为独立重复。','',
 '各录制ADE（原版／原数据微调／裁剪微调）：']
 for g in r['saved_translation']['per_recording']:
  vals=[r[a]['per_recording'][g]['ADE'] for a in arms];lines.append(g+' | '+' / '.join(f'{v:.3f}' for v in vals))
 lines+=['','4. 固定展示与失败保留',
 '继续使用之前就固定的9例、seed17，不按这次结果换案例。视频三栏依次：原版／原数据微调／裁剪微调。',
 '37.8秒，378帧；历史1.5秒＋未来0.5秒半速播放。白色=SAM参考中心，橙色=预测中心，蓝色=已观测历史。',
 '样例10_N01_020局部改善：99.87→81.86px；07_N01_023变差：62.71→87.14px；06_N01_034：51.18→61.12px。',
 '08_N01_010仍有328.34px较大误差，原样保留。部分三栏相同是验证选择了step0，并非视频未更新。',
 '上述为单个种子的展示，不替代3种子汇总。已查看统计图、07/10末帧和全部9例末帧联系图。','',
 '5. 检查与保存',
 '36新检查点与18原检查点预测重放一致；物理删除未来输入后新模型输出完全一致。',
 '每折训练排除测试/验证录制；相机输出与旧版完全一致，同时间各行人共享同一预测旋转，0无效投影射线。',
 '独立复算412窗8652个候选中心、6592次历史位置；SciPy重投影及指标最大差<1.4e-12px。',
 '两组初始权重与初始验证误差一致，均有非零梯度。338个受保护源文件哈希不变。',
 f'36次训练累计计时{c["training_seconds"]:.2f}秒（含各次保存/验证，不是纯GPU训练benchmark）；未新训练相机。',
 '初次数据整理仅修复numpy.bool_ JSON序列化，发生于所有训练与评分之前，锁定实验设置不变。',
 '不修改原标签、原权重、PPT、CVAT或默认训练数据集；新数据与新模型保存在独立版本。','',
 '6. 如何理解这次结果',
 '质量门槛降低了错误关联风险，但筛选后的数据并未自动转化为更好的预测泛化。',
 '本轮不足以定位是场景覆盖损失、剩余标签噪声、重叠采样或微调设置造成；这些因素没有被单独控制。',
 '因此保留原版，不因为本次结果继续删难例，也不追加测试集驱动的阈值/超参搜索。']
 report=ROOT/'reports'/OUT.name/'RESULTS.txt';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (OUT/'presentation/START_HERE.txt').write_text('先看 fixed_9_trimmed_finetune_comparison.mp4：左原版，中原数据微调，右裁剪数据微调。\ncomparison.png 为完整117窗3种子结果；fixed_9_contact_sheet.jpg 为固定9例末帧。\n结论：裁剪微调没有总体收益，ADE88.24→89.79px，原版本保留。\n完整报告：E:/New_research/reports/trimmed_forecast_finetune_20261028/RESULTS.txt\n',encoding='utf-8')
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items());files=[p for folder in [OUT,report.parent,Path(__file__).parent] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['delivery_hashes.json','delivery_complete.json']];hashes={str(p.relative_to(ROOT)):sha(p) for p in files};write(OUT/'delivery_hashes.json',hashes);assert all(sha(ROOT/k)==v for k,v in hashes.items());write(OUT/'delivery_complete.json',dict(passed=True,delivery_files=len(hashes),protected_original_files=len(protected),original_baseline_preserved=True));print(dict(report=str(report),counts=stats,delivery_files=len(hashes)))
if __name__=='__main__':main()
