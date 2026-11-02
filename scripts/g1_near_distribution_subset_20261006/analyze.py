"""Paired fixed-test comparisons and subset exposure; no selection on results."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_near_distribution_subset_20261006'
REPORT=ROOT/'reports/g1_near_distribution_subset_20261006'
BANK=ROOT/'outputs/transformer_matched97_20261006/new'
ORDER='EDGC'
VARIANTS=['A_early_fusion','B_person_only','C_separated','D_bidirectional','E_oneway','F_interacting_geometry']
NAMES=['A 先融合','B 仅行人','C 独立分离','D 双向交互','E 单向交互','F 交互后重投影']
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf8')
def exposure(folds,n,meta,balanced=False):
    w=np.zeros(n)
    for f in folds:
        ids=f['train_indices'];counts=Counter(meta[i]['recording'] for i in ids)
        for i in ids:w[i]+=1/(len(counts)*counts[meta[i]['recording']]) if balanced else 1/len(ids)
    w/=len(folds);assert np.isclose(sum(w),1)
    return w
def describe(x,w,thresholds):
    w=w/sum(w);bins=np.digitize(x[:,3],[.2,1,1.6]);high=x>=np.array([thresholds[k] for k in ORDER])
    return dict(means=(w@x).tolist(),high_shares=(w@high).tolist(),camera_bins=np.bincount(bins,weights=w,minlength=4).tolist(),n_positive=int(sum(w>0)),max_window_weight=float(max(w)))
def distribution():
    meta=read(BANK/'data/samples.json');desc=read(ROOT/'outputs/g1_edgc_training_distribution_20261006/window_distribution.json')
    survey=[r for r in read(ROOT/'outputs/g1_population_survey_20261006/measured_features.json') if r['near']]
    thresholds=read(ROOT/'outputs/g1_population_survey_20261006/protocol_locked.json')['thresholds']
    x=np.array([[r['features'][k] for k in ORDER] for r in desc]);sx=np.array([[r['features'][k] for k in ORDER] for r in survey]);sw=np.array([r['survey_weight'] for r in survey]);sw/=sum(sw)
    weights={'original':exposure(read(BANK/'folds.json'),len(meta),meta,True)}
    for a in ['matched','random']:weights[a]=exposure(read(OUT/a/'folds.json'),len(meta),meta)
    groups={a:describe(x,w,thresholds) for a,w in weights.items()};groups['survey_near']=describe(sx,sw,thresholds)
    locked=read(OUT/'selection_locked.json');eligible=read(OUT/'design_locked.json')['eligible']
    groups['eligible_pool_equal']=describe(x[eligible],np.ones(len(eligible)),thresholds)
    selection={a:dict(union_indices=np.flatnonzero(weights[a]>0).tolist(),per_fold_counts=[len(f['train_indices']) for f in read(OUT/a/'folds.json')]) for a in ['matched','random']}
    write(OUT/'distribution_comparison.json',dict(feature_order=ORDER,groups=groups,selection=selection,expected_exposure={a:w.tolist() for a,w in weights.items()},fold_matching=[dict(fold=f['fold'],**{k:f['diagnostics'][k] for k in ['pool_distance','matched_distance','random_distance']}) for f in locked['folds']]))
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),layout='constrained')
    names=['E 框高增长率 /秒','D 框高占图像高度 %','G 补偿后纵向速度 图像高度%/秒','C 相机平移速度 米/秒'];scale=[1,100,100,1]
    sets=[(sx,sw,'近人 G1 调查','#222222'),(x,weights['original'],'原训练','#9a9a9a'),(x,weights['random'],'等量随机','#d97706'),(x,weights['matched'],'分布匹配','#168e88')]
    for j,ax in enumerate(axes.flat):
        for xx,ww,label,color in sets:
            ids=np.flatnonzero(ww>0);ids=ids[np.argsort(xx[ids,j])]
            ax.step(xx[ids,j]*scale[j],np.cumsum(ww[ids])*100,where='post',label=label,color=color,lw=1.8)
        ax.set(xlabel=names[j],ylabel='累计占比 %',ylim=(0,101));ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9);fig.suptitle('子集训练的预期分布｜11 折等权汇总',fontsize=15)
    fig.savefig(OUT/'distribution_ecdf.png',dpi=170);fig.savefig(OUT/'distribution_ecdf.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,5.4),layout='constrained');pos=np.arange(4)
    for j,(a,label,color) in enumerate([('survey_near','近人 G1 调查','#444444'),('original','原训练','#a4a4a4'),('random','等量随机','#d97706'),('matched','分布匹配','#168e88')]):
        vals=np.array(groups[a]['camera_bins'])*100;b=ax.bar(pos+(j-1.5)*.2,vals,.19,label=label,color=color);ax.bar_label(b,fmt='%.1f',fontsize=9,padding=3)
    ax.set(xticks=pos,xticklabels=['<0.2','0.2–1','1–1.6','>1.6'],xlabel='相机平移速度 米/秒',ylabel='占比 %',ylim=(0,102));ax.legend(ncol=4,fontsize=9);ax.spines[['top','right']].set_visible(False);ax.set_title('缺失的慢速样本仍缺失，常速步行的集中度有所降低',loc='left')
    fig.savefig(OUT/'camera_speed_distribution.png',dpi=170);fig.savefig(OUT/'camera_speed_distribution.pdf');plt.close(fig)
    return groups,selection
def summarize(rows):
    def avg(rr,k):return float(np.mean([r['metrics'][k] for r in rr]))
    keys=['ADE15','FDE15'];recs=sorted({r['recording'] for r in rows});seeds=sorted({r['seed'] for r in rows})
    return dict(windows=len({r['key'] for r in rows}),recordings=len(recs),**{k:avg(rows,k) for k in keys},
       recording_equal={k:float(np.mean([avg([r for r in rows if r['recording']==rec],k) for rec in recs])) for k in keys},
       per_seed={str(s):{k:avg([r for r in rows if r['seed']==s],k) for k in keys} for s in seeds},
       sparse={k:avg([r for r in rows if r['nonoverlap']],k) for k in keys} if any(r['nonoverlap'] for r in rows) else None,
       invalid_steps=sum(r['metrics']['invalid'] for r in rows))
def paired(ar,br):
    aa={(r['key'],r['seed']):r for r in ar};bb={(r['key'],r['seed']):r for r in br};assert aa.keys()==bb.keys()
    recs=sorted({r['recording'] for r in ar});rows=[]
    for key in sorted({k[0] for k in aa}):
        ids=[k for k in aa if k[0]==key]
        rows.append(dict(key=key,recording=aa[ids[0]]['recording'],delta=float(np.mean([aa[k]['metrics']['ADE15']-bb[k]['metrics']['ADE15'] for k in ids]))))
    sums=np.array([sum(r['delta'] for r in rows if r['recording']==rec) for rec in recs]);counts=np.array([sum(r['recording']==rec for r in rows) for rec in recs]);rng=np.random.default_rng(2026100622)
    draws=rng.integers(0,len(recs),(4000,len(recs)));bootstrap=sums[draws].sum(1)/counts[draws].sum(1)
    rd={rec:float(sums[j]/counts[j]) for j,rec in enumerate(recs)}
    return dict(delta_ADE=float(sums.sum()/counts.sum()),recording_equal_delta=float(np.mean(list(rd.values()))),recording_wins=sum(v<0 for v in rd.values()),per_recording=rd,window_wins=sum(r['delta']<0 for r in rows),windows=len(rows),
       cluster_bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist(),per_seed={str(s):float(np.mean([aa[k]['metrics']['ADE15']-bb[k]['metrics']['ADE15'] for k in aa if k[1]==s])) for s in [17,29,43]})
def results():
    groups,selection=distribution();meta=read(BANK/'data/samples.json');locked=read(OUT/'selection_locked.json');near={meta[i]['key'] for i in locked['test_near_G1_indices']}
    sources={'original':BANK,'matched':OUT/'matched','random':OUT/'random'};rows={a:[r for r in read(p/'metrics.json') if r['mode']=='reviewed'] for a,p in sources.items()}
    summaries={};pairs={}
    for scope in ['fixed199','strict_near_G1']:
        keep=lambda r:scope=='fixed199' or r['key'] in near
        summaries[scope]={a:{v:summarize([r for r in rr if r['method']==v and keep(r)]) for v in VARIANTS+['hold_last']} for a,rr in rows.items()}
        pairs[scope]={}
        for a,b in [('matched','original'),('random','original'),('matched','random')]:
            pairs[scope][a+'_minus_'+b]={v:paired([r for r in rows[a] if r['method']==v and keep(r)],[r for r in rows[b] if r['method']==v and keep(r)]) for v in VARIANTS}
    # Existing evaluate.py independently computes the complete 199-test aggregate.
    for arm,path in sources.items():
        existing=read(path/'summary.json')['reviewed']
        for v in VARIANTS:assert abs(existing[v]['window_mean']['ADE15']-summaries['fixed199'][arm][v]['ADE15'])<1e-10
    write(OUT/'comparison.json',dict(summary=summaries,paired=pairs,uncertainty='4000 recording-cluster bootstrap resamples of seed-averaged paired window errors, conditional on one locked subset draw. Exploratory with few recordings.',target_distribution=groups,selection=selection))
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained');pos=np.arange(6)
    for ax,scope,title in zip(axes,['fixed199','strict_near_G1'],['固定原测试集 199 窗口','其中严格近人 G1 87 窗口']):
        for j,(a,label,color) in enumerate([('original','原训练','#9a9a9a'),('random','等量随机','#d97706'),('matched','分布匹配','#168e88')]):
            vals=[summaries[scope][a][v]['ADE15'] for v in VARIANTS];bars=ax.bar(pos+(j-1)*.25,vals,.24,label=label,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3,rotation=90)
        ax.axhline(summaries[scope]['original']['hold_last']['ADE15'],color='#555',ls='--',lw=1,label='保持最后位置')
        ax.set(xticks=pos,xticklabels=list('ABCDEF'),ylabel='ADE 像素 ↓',title=title);ax.set_ylim(0,max(summaries[scope][a][v]['ADE15'] for a in sources for v in VARIANTS)*1.20);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9,ncol=2);fig.savefig(OUT/'ade_comparison.png',dpi=180);fig.savefig(OUT/'ade_comparison.pdf');plt.close(fig)
    REPORT.mkdir(parents=True,exist_ok=True)
    def wins(scope,a,b):return sum(summaries[scope][a][v]['ADE15']<summaries[scope][b][v]['ADE15'] for v in VARIANTS)
    text=['# 近距离 G1 分布子集训练对照','',
      f'固定 199 个测试窗口上，分布匹配子集相对原训练有 {wins("fixed199","matched","original")}/6 种架构的平均 ADE 改善，相对等量随机子集有 {wins("fixed199","matched","random")}/6 种改善。限制到既有的 87 个严格近人 G1 窗口，相应为 {wins("strict_near_G1","matched","original")}/6 和 {wins("strict_near_G1","matched","random")}/6。两种测试口径均保留，不能只选择有利的一组。','',
      '本轮将现有训练数据按近距离 G1 抽样调查的历史 E、D、G、C 分布重新选择，再运行相同六种 Transformer。两组新训练分别为分布匹配子集、等量随机子集；原模型结果作为固定基线。所有预测仍是过去约 0.5 秒六点预测未来 1.5 秒十五点。','',
      '## 数据和比较范围','',
      '- 来源为上一版实际使用过的 295 个训练窗口；按此前 G1 质量条件及全窗口框高至少占图像高度 8% 筛出 183 个。近距离是表观尺度，并非米制距离。没有引入抽样调查的检测框或新标注作为预测标签。',
      '- 每折只在原有训练录制内选取约 70%，得到 79–123 个窗口；测试和验证仍采用完全相同的旧 199 窗口分折。近人 G1 子组固定为其中 87 个，未根据新模型误差筛选。',
      '- 目标来自 146 个近人调查窗口的逆抽样概率加权分布；每折先排除其测试和验证录制，再计算目标。历史特征只用过去六点；全片段可见性及既有 G1 筛选是离线数据资格条件。',
      '- 匹配目标包含四维边际分布、相机速度四段和 EDGC 十六种组合；保持每个可用训练录制至少一个窗口，单录制窗口上限为子集数的 40% 向上取整。70% 保留率及匹配目标在训练前固定，未根据验证或测试误差调整。',
      '- 两组新训练均按窗口等概率采样。原模型按录制再按窗口采样，因此新组对旧组同时改变筛选、数量、分布及采样器；分布匹配与等量随机的对照才保持资格、数量、训练采样器一致。随机对照保留每个录制至少一个样本，但并非额外匹配录制占比。',
      '- 六种架构、参数、三个随机种子、120×2 次更新、48 批量、40/20/40 输入增强、验证选模规则和 33 个冻结相机网络均沿用旧版。共训练 396 个行人网络，不训练相机网络。','',
      '## 训练分布','',
      '以下为 11 折等权的训练采样期望；调查列为完整近人调查参考，训练时使用逐折排除后的参考。','',
      '| 相机速度 米每秒 | 近人调查 | 原训练 | 等量随机 | 分布匹配 |','|---|---:|---:|---:|---:|']
    for j,label in enumerate(['<0.2','0.2–1','1–1.6','>1.6']):text.append('| '+label+' | '+' | '.join(f'{100*groups[a]["camera_bins"][j]:.1f}%' for a in ['survey_near','original','random','matched'])+' |')
    text+=['','现有合格数据没有 0.2–1 米/秒窗口，部分折也没有停走窗口，因此匹配只是现有支持范围内的近似，不能说已复现完整目标分布。调查仍受可检测、可连续测量条件约束；概率加权没有消除漏检。','',
       '| 历史特征高组 | 近人调查 | 原训练 | 等量随机 | 分布匹配 |','|---|---:|---:|---:|---:|']
    for j,k in enumerate(ORDER):text.append('| '+k+' | '+' | '.join(f'{100*groups[a]["high_shares"][j]:.1f}%' for a in ['survey_near','original','random','matched'])+' |')
    for scope,title in [('fixed199','固定原测试集'),('strict_near_G1','其中严格近人 G1')]:
        text+=['','## '+title,'','| 模型 | 原训练 ADE | 等量随机 ADE | 分布匹配 ADE | 匹配较随机变化 | 原训练 FDE | 匹配 FDE |','|---|---:|---:|---:|---:|---:|---:|']
        for v,name in zip(VARIANTS,NAMES):
            a=summaries[scope]['original'][v];b=summaries[scope]['random'][v];c=summaries[scope]['matched'][v]
            text.append(f'| {name} | {a["ADE15"]:.2f} | {b["ADE15"]:.2f} | {c["ADE15"]:.2f} | {c["ADE15"]-b["ADE15"]:+.2f} | {a["FDE15"]:.2f} | {c["FDE15"]:.2f} |')
        text+=['',f'保持最后位置 ADE 为 {summaries[scope]["original"]["hold_last"]["ADE15"]:.2f} 像素。所有结果为三个训练种子平均；单位为像素，越低越好。','',
          '| 模型 | 匹配较随机 ADE 差的录制簇 95% 区间 | 改善录制 | 三种子变化 | 录制等权变化 |','|---|---:|---:|---|---:|']
        for v,name in zip(VARIANTS,NAMES):
            p=pairs[scope]['matched_minus_random'][v];lo,hi=p['cluster_bootstrap_95'];sd='/'.join(f'{z:+.1f}' for z in p['per_seed'].values())
            text.append(f'| {name} | [{lo:+.1f}, {hi:+.1f}] | {p["recording_wins"]}/{len(p["per_recording"])} | {sd} | {p["recording_equal_delta"]:+.2f} |')
        text+=['','| 模型 | 原训练录制等权 ADE | 匹配录制等权 ADE | 原训练稀疏窗口 ADE | 匹配稀疏窗口 ADE |','|---|---:|---:|---:|---:|']
        for v,name in zip(VARIANTS,NAMES):
            a=summaries[scope]['original'][v];b=summaries[scope]['matched'][v]
            text.append(f'| {name} | {a["recording_equal"]["ADE15"]:.2f} | {b["recording_equal"]["ADE15"]:.2f} | {a["sparse"]["ADE15"]:.2f} | {b["sparse"]["ADE15"]:.2f} |')
    text+=['','## 结果的解释范围','',
      '本轮比较的是一个预先固定的子集策略。在相同测试窗口上，匹配组与等量随机组的差异可作为分布整理是否有用的初步证据；两组只有一次子集抽取，三个模型种子不能代表重复子集实验。若不同架构、种子或录制汇总方向不同，应保留这些差异。','',
      '固定测试集并不是新调查人群的概率样本，因此其 ADE 不能当作全体近人 G1 的总体误差。87 窗口子组覆盖九次录制，其中 53 个来自 192702，因此同时保留录制等权及按录制重抽样的结果。该子组也继承已有自动标注和抽样核查的限制，未在本轮新增逐帧人工真值。','',
      '调查特征来自六帧独立 SAM 提示测量，训练源来自传播并修订的 mask，两者有测量方法差异。E、D、G、C 的选择也继承此前数据探索，本轮是固定旧测试的探索性对照，不是新增独立测试。相机姿态及 SAM 标注均为离线参考；没有将未来相机信息用于常规预测。参考未来旋转只在另存的诊断输出中使用。','',
      '## 复现与核验','',
      '子集名单、目标统计、匹配残差和协议在训练前锁定并记录哈希；逐折训练录制与验证、测试隔离。两组全部模型逐一复现三类输入预测，并将未来后缀物理删除后检查推理不变。原始数据、旧模型、调查结果和原代码记录保留。详细核验见各组 verification.json 及独立 audit.json。','',
      '输出目录：`outputs/g1_near_distribution_subset_20261006/`。其中 `comparison.json` 包含按录制、种子、稀疏窗口的完整统计；`selection_locked.json` 包含各折具体名单。','']
    (REPORT/'RESULTS.md').write_text('\n'.join(text),'utf8')
    print(json.dumps(dict(summary=summaries,paired=pairs),ensure_ascii=False)[:1000],flush=True)
if __name__=='__main__':
    if '--distribution-only' in sys.argv:distribution()
    else:results()
