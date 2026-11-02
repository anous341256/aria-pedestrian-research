"""Render descriptive availability figures and a reproducible inventory report."""
from inventory import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

NAMES={'A':'累计转头角度','B':'净朝向变化','C':'相机平移速度','D':'人物框高比例','E':'人物尺度变化率','F':'补偿后横向速度','G':'补偿后纵向速度','H':'偏离匀速的残差'}
UNITS={'A':'度','B':'度','C':'米/秒','D':'框高/图高','E':'每秒（log框高）','F':'图宽/秒','G':'图高/秒','H':'残差/框高'}

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
 report=ROOT/'reports/g1_factor_inventory_20261006';report.mkdir(parents=True,exist_ok=True)
 figures=OUT/'figures';figures.mkdir(exist_ok=True)
 old=read(OUT/'existing_summary.json');new=read(OUT/'supplement_summary.json');both=read(OUT/'combined_summary.json')
 banks=read(OUT/'checkpoint_availability.json');ranges=read(OUT/'factor_ranges_and_cell_reliability.json')
 ann=read(ROOT/'outputs/sam2_g1_factor_supplement_20261006/dataset/summary.json');integ=read(OUT/'integrity.json')
 plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
 fig,axes=plt.subplots(1,2,figsize=(13.2,5),gridspec_kw={'width_ratios':[1.3,1]})
 x=np.arange(16);a=axes[0]
 a.bar(x-.19,[r['windows'] for r in old['cells']],.38,color='#8898a6',label='补充前')
 a.bar(x+.19,[r['windows'] for r in both['cells']],.38,color='#237c86',label='补充后')
 a.set(xticks=x,xticklabels=[f'{i:02d}' for i in range(1,17)],ylabel='严格匹配的窗口数（有重叠）',xlabel='原定 L16 组合',title='总量增加，组合覆盖仍不完整')
 a.legend(frameon=False);a.set_ylim(0,8.5);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
 for j,r in enumerate(both['cells']):a.text(j+.19,r['windows']+.12,str(r['windows']),ha='center',fontsize=9)
 c=np.array(both['feature_correlation']);im=axes[1].imshow(c,vmin=-1,vmax=1,cmap='RdBu_r')
 axes[1].set(xticks=range(8),xticklabels=FACTORS,yticks=range(8),yticklabels=FACTORS,title='候选特征之间的 Pearson 相关')
 for i in range(8):
  for j in range(8):axes[1].text(j,i,f'{c[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(c[i,j])>.65 else '#23313c')
 fig.colorbar(im,ax=axes[1],shrink=.78)
 fig.suptitle('G1 样本可用性盘点 · 未读取预测误差、未重新训练',fontsize=14,y=1.01)
 fig.text(.04,-.015,'A累计旋转  B净旋转  C相机平移  D框高比例  E尺度变化  F横向速度  G纵向速度  H非匀速残差\n特征取过去0.5秒；相关性受重叠窗口及录制构成影响，不是因果或因素效应。',fontsize=9)
 fig.tight_layout();fig.savefig(figures/'coverage_and_correlation.png',dpi=180,bbox_inches='tight');plt.close(fig)
 rows=[r for r in read(OUT/'combined_windows.json') if r['eligible']]
 non=pack_nonoverlap(rows,False)
 write(OUT/'recording_time_nonoverlap_keys.json',[r['key'] for r in non])
 miss=[str(r['cell']).zfill(2) for r in both['cells'] if r['windows']==0]
 main_table=table(['口径','补充前','本轮增加','补充后'],[[label,old[k],new[k],both[k]] for label,k in [('通过当前筛选的窗口','windows'),('每个片段内同一人不重叠窗口','person_nonoverlap'),('录制内时间不重叠时段','recording_time_nonoverlap'),('涉及录制数（有交集，不能相加）','recordings'),('片段内人物ID','tracks')]])
 cells=table(['组合','ABCDEFGH（−低，+高）','原有窗口','新增窗口','合计窗口','人内不重叠','录制时间不重叠','录制数'],[[f"{r['cell']:02d}",''.join('+' if n>0 else '−' for n in r['levels']),old['cells'][i]['windows'],new['cells'][i]['windows'],r['windows'],r['person_nonoverlap'],r['recording_time_nonoverlap'],r['recordings']] for i,r in enumerate(both['cells'])])
 modeltable=table(['固定权重库','已有可评价窗口','新增可评价窗口','合计','录制时间不重叠','录制数'],[[name,b['summaries']['existing']['windows'],b['summaries']['supplement']['windows'],b['summaries']['combined']['windows'],b['summaries']['combined']['recording_time_nonoverlap'],b['summaries']['combined']['recordings']] for name,b in banks.items()])
 factor_table=table(['因素','历史量及单位','固定低/高分界','低/高窗口数','实际范围'],[[k,f'{NAMES[k]}（{UNITS[k]}）',f"{both['thresholds'][k]:.6g}",f"{ranges['factors'][k]['low']} / {ranges['factors'][k]['high']}",f"{ranges['factors'][k]['min']:.4g} ～ {ranges['factors'][k]['max']:.4g}"] for k in FACTORS])
 rec_table=table(['录制尾号','合格窗口','人内不重叠','时间不重叠'],[[k.split('_')[-1],d['windows'],d['person_nonoverlap'],d['time_nonoverlap']] for k,d in both['by_recording'].items()])
 text=f'''# G1 因素筛选：可用样本盘点与补充标注

日期：2026-10-06。范围：固定已有模型，准备比较 G1 场景因素与预测误差。此轮完成样本与特征可用性盘点及新增标注；没有重新训练模型，没有读取或根据 ADE/FDE 挑选样本。

## 主要结果

通过当前统一筛选的窗口由 **{old['windows']} 增加到 {both['windows']}**，增加 **{new['windows']}（{new['windows']/old['windows']:.1%}）**。合并同录制同时段不同人物后，有 **{both['recording_time_nonoverlap']} 个不重叠时段、{both['recordings']} 个录制**。这些时段仍共享录制、环境及可能的身份，不等于独立同分布样本。

{main_table}

每个窗口含6个历史点（0.5秒跨度）及15个未来点（1.5秒），共21点、2秒跨度。窗口默认每0.5秒滑动。“人内不重叠”在同片段同ID内选择不重叠21点；“时间不重叠”在录制内进一步合并同步人物。非重叠清单是可行的贪心选择，不是独立性保证，也不是最终正交取样。

本轮补标 **13段、104秒、54个片段内ID、56个标注区间、2104个最终mask**。这不是54个经跨片段确认的独立人物。14段输入候选中的59号因近处人物缺乏清晰连续全身观察而在SAM传播前剔除。保留13段全部视频便于核查，并不表示其整个8秒都属于合格G1。

## 本轮筛选与质量记录

从原有已准备的395个窗口开始盘点，统一筛后204个通过；本轮13段产生191个宽松连续窗口、131个清晰连续窗口，再经G1相机运动和几何筛选后保留64个，来自10个片段、23个ID、9个录制。未通过本轮清晰/运动规则的轨迹仍保留原始标注，但不计入268个候选。

G1沿用本地操作性条件：21点内最大1秒净旋转≤20度，帧间角速度第95百分位≤50度/秒；结合平地通行和目标连续可见的视觉检查。这些数值是本地筛选先验，不是论文官方G1分类定义。

近处入选依据表观人物框高约≥图高12%，持续跟踪保留至8%；没有总人数上限，3人一批只用于调度。V6传播及已标注人物负点沿用之前方案。SAM2离线双向传播使用Aria参考旋转，明确记录人工式锚点选择和本代理的视觉核查，不能当作全自动在线跟踪结果。

所有入选窗口要求：21点连续、没有UNCERTAIN/PENDING、无已记录遮挡区间、框离边缘>5像素、框高≥8%、mask/引导框IoU≥0.45且越界比例≤0.1、MPS质量≥0.5、同一图和匹配误差≤30ms、时间间隔0.08–0.12秒、可投影、无相机平移异常跳跃。没有插补mask中心或不可见身体。

新增54个ID都检查了时序RGB和mask抽样图，并查看自动报警帧；撤回3个明显不可靠mask，另将4对未解决重叠涉及的8个mask隔离，留下17个人物帧待定和58个不稳定人物帧。柜台遮腿、行李遮挡和前景穿过等已写入逐ID排除区间。检查是代理抽样核查，尚不是独立人工逐帧GT。原有204个候选沿用此前抽样核查，本轮没有重新逐窗口视觉验证全部旧标签。

## L16是否可直接执行

**目前不可以作为完整正交对照直接计算效应。** 268个窗口中只有15个严格落在原定16行组合，覆盖8行；空缺：**{', '.join(miss)}**。已覆盖行也缺少跨录制重复。其他253个窗口不是坏样本，只是属于其余二水平组合。

{cells}

每行的非重叠数在该行内部计算，跨行加总可能仍有时间重叠。精确组合按补充前冻结的阈值定义，不把接近格子的样本硬塞进去，也没有为了填格移动阈值。

![组合覆盖和特征相关](../../outputs/g1_factor_inventory_20261006/figures/coverage_and_correlation.png)

八个二水平因素理论上有256种符号组合，L16只是其中16种指定组合。在自然数据中并没有操纵因素去实现这些16行，因此总窗口数增加不等于16行都能填满。本轮不能判断空格是物理不可行还是当前来源不足。净旋转≤累计旋转是明确几何约束，但不同分界值下，不能据此直接宣布某一高低格不可能。

全部候选的截距+8因素二水平设计矩阵秩为{both['design_main_rank']}（列数9）；这只说明整体主效应列不存在精确线性依赖，并不代表正交、稳定估计、足够统计功效或能辨别因果。相关系数：A/B={both['feature_correlation'][0][1]:.2f}，E/G={both['feature_correlation'][4][6]:.2f}，C/H={both['feature_correlation'][2][7]:.2f}，仍需处理共变和录制差异。

## 因素测量范围与稳定性

{factor_table}

低为≤分界值，高为>分界值。A–H全部只取过去6点。F/G保留方向，所谓高低不是绝对运动幅度；E也有正负，区分放大与缩小。C取MPS相机中心路径速度，包含相机外参杠杆臂效应，不应等同于佩戴者身体行走速度。旋转补偿后的运动仍混合平移、视差、行人运动和标注误差。

删除一个历史内点的敏感性检查中，D/E/F/G/H分别有{', '.join(str(both['level_disagreement_under_delete_one'][k]) for k in 'DEFGH')}个窗口可能越过高低分界。H为{both['level_disagreement_under_delete_one']['H']}/{both['windows']}，对短历史和mask边界抖动明显敏感。该检查使用最大变化界，是可能跨界的保守提醒，不是标签错误率，也不能作为校准后的测量可靠率。逐格连续范围与该敏感性计数已存JSON；A/B/C未通过此删除点检查评价可靠率。

## 固定模型评价资格

同时检查相机和行人权重所对应的训练、验证及测试录制，并确认6个架构×3个种子和匹配相机文件存在。此处只审计资格，没有跑误差，也没有按分数选权重库。

{modeltable}

两个库分别是扩充第二版和最近matched97新数据版，不能混成同一个固定模型结果。`expanded_v2`可覆盖228个窗口，最近`matched97_new`可覆盖214个。以最近库为准，有54个候选只能先进入分布盘点；不能用训练或验证中出现过该录制的相机权重作留出评价。未来如扩大这些录制的评分覆盖，需要另立相机/行人留出训练步骤，不属于本轮已执行工作。

注意：这是冻结的录制留出权重集合，不是一套同权重网络评价全部录制；六架构分别分析。旧留出结果已多次查看，属于开发证据。新增同录制片段是未标注过的时间段，不是全新的录制级盲测。

## 录制构成

{rec_table}

## 下一步建议（尚未执行）

保留这八个候选因素及原L16缺口记录。当前建议先用可评价样本作连续特征、录制分组的探索分析，确认关联方向和不确定性；不能将这个改法仍称为完整L16正交实验。如果坚持16格等量对照，应按缺格寻找真实候选、并同时满足留出权重覆盖，不能只按总条数继续泛化补标。

H需要先做测量敏感性分析；A/B、E/G等相关因素的联合解释应谨慎。不能凭本轮盘点宣布哪些因素决定预测误差，本轮根本没有读取误差。也尚未检验任何采样方案能提高训练效果。

## 交付与验证

- 补充标注与104秒合集：`outputs/sam2_g1_factor_supplement_20261006/presentation/`。
- 新增64个完整窗口：该版本`dataset/all_eligible_dataset.npz`及`all_eligible_samples.json`。
- 合并特征清单、固定阈值、逐格缺口、权重资格和非重叠清单：`outputs/g1_factor_inventory_20261006/`。
- 原始SAM传播、第一次核查、最终标注和排除原因分别保留；没有覆盖旧数据或把新增数据自动并入训练集。
- 原有{integ['original_files_verified']}个输入/标注/校准文件SHA256复核一致；阈值文件固定为`{integ['threshold_sha256']}`。
- 最终mask的RLE像素数、外接框、唯一ID/帧、姿态时序和投影检查通过；展示视频另有逐帧解码验证记录。
'''
 (report/'RESULTS.md').write_text(text,encoding='utf8')
 print(report/'RESULTS.md')

if __name__=='__main__':main()
