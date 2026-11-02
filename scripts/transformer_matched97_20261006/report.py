"""Write measured paired results and a publication-style static figure."""
from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 x=read(VERSION/'comparison.json');audit=read(VERSION/'paired_audit.json');assert audit['passed']
 REPORT.mkdir(parents=True,exist_ok=True);pres=VERSION/'presentation';pres.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
 for ax,metric,title in zip(axes,['ADE15','FDE15'],['Average displacement error','Final displacement error']):
  xx=np.arange(6)
  for j,(arm,label,color) in enumerate([('base199','Base data','#90979F'),('old','Previous batch (matched)','#D89C45'),('new','New G1 batch (matched)','#31798B')]):
   yy=[x['results'][v][arm]['window_mean'][metric] for v in VARIANTS];ax.bar(xx+(j-1)*.25,yy,.23,label=label,color=color)
  ax.set_xticks(xx,list('ABCDEF'));ax.set_ylabel('Pixels (lower is better)');ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.10),ncol=3);fig.suptitle('Fixed 199 test windows | 11 recording folds | 3 seeds',fontsize=13)
 fig.savefig(pres/'paired_results.png',dpi=180);fig.savefig(pres/'paired_results.pdf');plt.close(fig)
 rows=['# 等量新增 G1 批次替换对照','',
 '固定训练数量、模型和原199个测试窗口后，新批次使六组中的五组ADE下降：A/B/D/E/F均在三个种子上改善，C平均略差。D从163.73降至153.33像素。这个批次替换结果表明“仅仅是数据量增加”不足以解释此前的退化；但不能把批次差异单独归因于标注质量。',
 '', '## 实验设置','',
 '- 基础数据保持 199 个窗口；旧新增池和新新增池各 97 个窗口。两池不叠加。',
 '- 每个窗口为历史 6 点（约 0.5 秒）和未来 15 点（1.5 秒）；采样步长 0.1 秒，窗口步幅 5 帧。',
 '- 固定原 11 个录制划分、验证索引和全部 199 个测试索引。每轮排除测试及验证录制的新增数据，再分别对齐新增行走/静止窗口数量。',
 '- 六个原架构、约 21 万可训练参数、3 个随机种子、每模型 120×2 次更新、批量 48；相机模型使用完全相同的 33 个匹配留出权重。',
 '- 训练采样仍按录制均衡；批次的录制组成不同，因此其抽样权重分布也不同。',
 '- 所有新增样本在预测评分前冻结；标注依据 SAM2/V6、离线双向传播、显式提示修正和视觉抽查，不是独立人工 GT。',
 '', '## 固定测试集结果','',
 '单位：像素，越低越好。每个测试窗口先对三个种子等权，随后对窗口等权。旧批次列使用本次数量匹配后的训练，可能与上次全量加入旧97的分数不同。',
 '', '| 模型 | 仅基础199 ADE | 旧批次 ADE | 新批次 ADE | 新−旧 ADE | 旧批次 FDE | 新批次 FDE |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for v,r in x['results'].items():
  rows.append(f"| {NAMES[v]} | {r['base199']['window_mean']['ADE15']:.2f} | {r['old']['window_mean']['ADE15']:.2f} | {r['new']['window_mean']['ADE15']:.2f} | {r['delta_ADE']:+.2f} | {r['old']['window_mean']['FDE15']:.2f} | {r['new']['window_mean']['FDE15']:.2f} |")
 hold=x['hold_last']['window_mean'];rows+=['',f"保持最后位置参照：ADE {hold['ADE15']:.2f}，FDE {hold['FDE15']:.2f}。",'','## 稳定性','', '| 模型 | 种子17 新−旧 | 种子29 | 种子43 | 改善录制/11 | 改善窗口/199 | 录制等权 新−旧 | 稀疏窗口 新−旧 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for v,r in x['results'].items():
  d=r['per_seed_delta'];rec=r['new']['recording_equal']['ADE15']-r['old']['recording_equal']['ADE15'];sparse=r['sparse']['new']['window_mean']['ADE15']-r['sparse']['old']['window_mean']['ADE15']
  rows.append(f"| {NAMES[v]} | {d['17']:+.2f} | {d['29']:+.2f} | {d['43']:+.2f} | {r['recording_wins']}/11 | {r['window_wins']}/199 | {rec:+.2f} | {sparse:+.2f} |")
 rows+=['','稀疏子集沿用旧数据预存的 nonoverlap 标记；窗口和同一录制内的人物并非独立实验样本。三个种子不能作为三个独立数据集。','','## 同相机运动筛选的测试子集','',f"固定旧测试199中有 {len(x['same_motion_screen_test_indices'])} 个窗口也满足本次20°/1秒与50°/秒规则，仅按数值运动条件筛选，不能等同于完整视觉G1分类。该子集规则在新批次结果读取前确定。",'', '| 模型 | 旧批次 ADE | 新批次 ADE |','|---|---:|---:|']
 for v,r in x['results'].items():
  z=r['same_motion_screen_test'];rows.append(f"| {NAMES[v]} | {z['old']['ADE15']:.2f} | {z['new']['ADE15']:.2f} |")
 rows+=['','## 每轮实际新增量','','| 轮次 | 基础训练 | 新增行走（两组相同） | 新增静止（两组相同） | 总训练窗口 |','|---|---:|---:|---:|---:|']
 for f in x['protocol']['folds']:
  c=f['matched_counts'];rows.append(f"| {f['fold']} | {f['base_train']} | {c['False']} | {c['True']} | {f['base_train']+sum(c.values())} |")
 rows+=['','## 新旧批次组成','','| 指标 | 旧97 | 新97 |','|---|---:|---:|']
 for key,label in [('static','静止窗口'),('clips','片段数'),('tracks','片段内轨迹ID'),('turn_mean_deg','未来终点旋转均值（度）'),('turn_ge15_fraction','终点旋转≥15°比例'),('height_mean','历史末帧框高/图高均值')]:
  rows.append(f"| {label} | {x['distribution']['old'][key]:.4g} | {x['distribution']['new'][key]:.4g} |")
 rows+=['','新批次逐窗口要求最大1秒旋转≤20°、角速度95分位≤50°/秒，并视觉排除明显遮挡、截断和强互动。该阈值是本地操作规则，不是论文的数值定义；候选视频整体不一定都是G1。旧批次未重新按此阈值筛选，这属于有意改变的数据筛选条件。','','## 能回答与尚不能回答的内容','',
 '本实验观察到等量替换后的ADE改善，且A/B/D/E/F在三个种子中方向一致。同相机运动筛选的114个测试窗口也呈现五组改善、C略差。但稀疏窗口统计中A/E转为小幅退化，C的不同聚合方式和种子也不一致；因此不能概括为所有统计口径与指标均稳健改善。FDE仅B/D/E/F改善。',
 '', '相比仅基础199的结果，A/B/D大致回到此前水平，E/F更低；六组新模型的ADE仍未优于保持最后位置的149.67像素。本次说明替换批次可以减轻先前的退化，没有证明新增数据一定提升模型，也没有改变原先选定的基线。',
 '', '新97中终点旋转≥15°的比例从旧97的39.2%降到12.4%，人物尺度和录制组成也变化。原因可能涉及场景、相机运动分布、人物行为、有效身份数量、按录制均衡抽样及标签误差的共同作用。没有独立人工GT重标或固定样本的标签质量干预，因此不能宣称已证明旧数据质量差。',
 '', '固定更新次数也意味着新增数据改变每个样本的有效训练机会；这次只控制两批之间的预算，尚未单独检验扩充数据是否需要更长训练。',
 '', '## 核查与资产','',f"两组共验证 {audit['models_verified']} 个预测器；全部权重重放及未来后缀替换检查通过。33组相机文件逐一相同。独立几何合成最大差异 {audit['geometry_max_px']:.6f} 像素。受保护旧文件 {audit['protected_files']} 个未改变。",
 '',f"新增标注：`{EXPANSION}`；实验：`{VERSION}`；完整每窗口指标、三个输入版本和参考旋转诊断均保存。未来参考旋转诊断不属于主结果。",'',
 '[新标注实际入选窗口合集](../../outputs/sam2_g1_matched97_20261006/presentation/SELECTED_G1_WINDOWS.mp4)', '',
 '[四个固定测试案例 × 六组模型，新旧配对视频](../../outputs/transformer_matched97_20261006/presentation/FIXED4_SIX_MODELS_OLD_NEW.mp4)', '',
 '旧权重复用时，在原始批量上重放误差为零。缩小测试批量可产生约0.00015像素的CUDA舍入差异，已用原始批量重放核对；训练代码与输入没有因此调整。早期检查修正了复制文件换行及日志模块导入问题，均发生在相关结果使用前，记录保留。', '', '![结果图](../../outputs/transformer_matched97_20261006/presentation/paired_results.png)']
 (REPORT/'RESULTS.md').write_text('\n'.join(rows)+'\n',encoding='utf8')
 print(REPORT/'RESULTS.md',flush=True)
if __name__=='__main__':main()
