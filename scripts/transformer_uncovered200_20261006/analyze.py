"""Fixed-test paired comparisons and historical distribution description; no fitting."""
from shared import *
from collections import Counter
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

spec=importlib.util.spec_from_file_location('previous_analysis',ROOT/'scripts/g1_near_distribution_subset_20261006/analyze.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
ORDER='EDGC'
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

def distribution():
    meta=read(DATA/'samples.json');data=dict(np.load(DATA/'dataset.npz'))
    old=read(ROOT/'outputs/g1_edgc_training_distribution_20261006/window_distribution.json')
    values=np.array([[r['features'][k] for k in ORDER] for r in old]);new=[];cache={}
    for i,m in enumerate(meta[296:],296):
        cid=m['source_clip']
        if cid not in cache:cache[cid]=dict(np.load(EXPANSION/cid/'input/reference.npz'))
        cal=cache[cid];fs=np.asarray(m['frames'][:6]);t=(cal['times_ns'][fs]-cal['times_ns'][fs[0]])/1e9;tt=t-t.mean()
        b=data['boxes'][i,:P].astype(float);height=b[:,3]-b[:,1]
        p=warp(data['xy'][i,:P].astype(float),data['H'][i,:P].astype(float))[0]
        # position is the world camera center, including device-to-camera offset.
        camera_speed=float(np.linalg.norm(np.diff(cal['position'][fs],axis=0),axis=1).sum()/(t[-1]-t[0]))
        new.append([float(np.sum(tt*np.log(height))/sum(tt*tt)),float(np.median(height)/1920),
                    float(np.sum(tt*p[:,1])/sum(tt*tt)/1920),camera_speed])
    values=np.vstack([values,new]);assert len(values)==len(meta) and np.isfinite(values).all()
    thresholds=read(ROOT/'outputs/g1_population_survey_20261006/protocol_locked.json')['thresholds']
    weights={
        'previous':prior.exposure(read(BASELINE/'folds.json'),len(meta),meta,True),
        'expanded':prior.exposure(read(OUT/'folds.json'),len(meta),meta,True),
        'previous_matched_subset':prior.exposure(read(ROOT/'outputs/g1_near_distribution_subset_20261006/matched/folds.json'),len(meta),meta,False)}
    groups={name:prior.describe(values,w,thresholds) for name,w in weights.items()}
    groups['added_equal']=prior.describe(np.array(new),np.ones(len(new)),thresholds)
    survey=[r for r in read(ROOT/'outputs/g1_population_survey_20261006/measured_features.json') if r['near']]
    sx=np.array([[r['features'][k] for k in ORDER] for r in survey]);sw=np.array([r['survey_weight'] for r in survey])
    groups['near_survey']=prior.describe(sx,sw,thresholds)
    write(OUT/'distribution_comparison.json',dict(order=ORDER,thresholds=thresholds,groups=groups,
        added_expected_training_share=float(weights['expanded'][296:].sum()),
        added_sample_share=78/373,
        features=[dict(index=i,key=m['key'],recording=m['recording'],values=dict(zip(ORDER,values[i].tolist())),
            expected_old_weight=float(weights['previous'][i]),expected_expanded_weight=float(weights['expanded'][i])) for i,m in enumerate(meta)],
        note='Equal11-fold expected sampler exposure. Survey near population uses inverse selection weights, but differs in mask measurement. Not forced distribution matching in this expansion.'))
    print('Distribution prepared; added camera bins:',groups['added_equal']['camera_bins'],flush=True)
    return groups

def main():
    groups=distribution();assert read(OUT/'verification.json')['passed']
    meta=read(DATA/'samples.json');locked=read(OUT/'selection_locked.json');near={meta[i]['key'] for i in locked['strict_near_test_indices']}
    sources={'previous':BASELINE,'expanded':OUT,'previous_matched_subset':ROOT/'outputs/g1_near_distribution_subset_20261006/matched'}
    rows={a:[r for r in read(p/'metrics.json') if r['mode']=='reviewed'] for a,p in sources.items()}
    methods=VARIANTS+['hold_last'];summaries={};pairs={};robust={}
    for scope in ['fixed199','strict_near87']:
        keep=lambda r:scope=='fixed199' or r['key'] in near
        summaries[scope]={a:{v:prior.summarize([r for r in rr if keep(r) and r['method']==v]) for v in methods} for a,rr in rows.items()}
        pairs[scope]={b:{v:prior.paired([r for r in rows['expanded'] if keep(r) and r['method']==v],
            [r for r in rows[b] if keep(r) and r['method']==v]) for v in VARIANTS} for b in ['previous','previous_matched_subset']}
        # Paired sparse-window and leave-one-recording-out sensitivity summaries.
        robust[scope]={}
        for v in VARIANTS:
            a=[r for r in rows['expanded'] if keep(r) and r['method']==v];b=[r for r in rows['previous'] if keep(r) and r['method']==v]
            rr=sorted({r['recording'] for r in a});byrec={}
            for rec in rr:
                aa=[r['metrics']['ADE15'] for r in a if r['recording']!=rec];bb=[r['metrics']['ADE15'] for r in b if r['recording']!=rec]
                byrec[rec]=float(np.mean(aa)-np.mean(bb))
            robust[scope][v]=dict(sparse=prior.paired([r for r in a if r['nonoverlap']],[r for r in b if r['nonoverlap']]),
                leave_one_recording_out_delta=byrec)
    # One row per fixed test case, seeds averaged. Sorted only for failure inspection, never tuning.
    cases=[]
    for v in VARIANTS:
        aa={(r['key'],r['seed']):r for r in rows['expanded'] if r['method']==v}
        bb={(r['key'],r['seed']):r for r in rows['previous'] if r['method']==v}
        assert aa.keys()==bb.keys()
        for key in sorted({k[0] for k in aa}):
            kk=[k for k in aa if k[0]==key];a=float(np.mean([aa[k]['metrics']['ADE15'] for k in kk]));b=float(np.mean([bb[k]['metrics']['ADE15'] for k in kk]))
            cases.append(dict(method=v,key=key,recording=aa[kk[0]]['recording'],near=key in near,before_ADE=b,after_ADE=a,delta_ADE=a-b))
    write(OUT/'paired_cases.json',cases)
    allinput={a:read(p/'metrics.json') for a,p in sources.items() if a in ['previous','expanded']}
    secondary={mode:{a:{v:prior.summarize([r for r in rr if r['mode']==mode and r['method']==v]) for v in methods} for a,rr in allinput.items()} for mode in ['pre_review','controlled_noise']}
    comparison=dict(summary=summaries,paired_expanded_minus=pairs,robustness=robust,secondary_inputs=secondary,
        scope='Exploratory fixed-test training-pool expansion, changes amount/composition/exposure jointly; not clean-label versus noisy-label causal trial.')
    selection={}
    for v in VARIANTS:
        selection[v]={}
        for name,path in [('previous',BASELINE),('expanded',OUT)]:
            selected=[read(p) for p in path.glob(f'fold_*/seed_*/{v}/complete.json')]
            assert len(selected)==33
            selection[v][name]=dict(selected_epoch_counts=dict(Counter(str(z['epoch']) for z in selected)),
                mean_validation_ADE=float(np.mean([z['validation_image_ADE'] for z in selected])),
                models_selecting_final_epoch=sum(z['epoch']==120 for z in selected))
    comparison['validation_selection']=selection
    write(OUT/'comparison.json',comparison)
    # Ready-to-review numerical figure, no generated art.
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.3),layout='constrained')
    for ax,scope,title in zip(axes,['fixed199','strict_near87'],['固定测试：199 个窗口','其中近人 G1：87 个窗口']):
        x=np.arange(6);ys=[]
        for j,(src,label,color) in enumerate([('previous','扩充前完整训练库','#919aa6'),('previous_matched_subset','上次分布匹配子集','#bdc6df'),('expanded','此次加入78个窗口','#188d87')]):
            yy=[summaries[scope][src][v]['ADE15'] for v in VARIANTS];ys+=yy
            pos=x+(j-1)*.23
            ax.scatter(pos,yy,s=42,color=color,label=label,zorder=3,marker=['s','D','o'][j])
            for px,py in zip(pos,yy):ax.annotate(f'{py:.1f}',(px,py),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8,rotation=90)
        hold=summaries[scope]['previous']['hold_last']['ADE15'];ys.append(hold)
        ax.axhline(hold,color='#b86733',ls='--',lw=1.2,label='保持最后位置')
        ax.set_xticks(x,list('ABCDEF'));ax.set_ylim(min(ys)-5,max(ys)+10)
        ax.set_title(title,fontproperties=FONT,fontsize=13);ax.set_ylabel('ADE / 像素，越低越好',fontproperties=FONT)
        ax.grid(axis='y',alpha=.12);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(prop=FontProperties(fname=FONT.get_file(),size=8),loc='upper left')
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    report(summaries,pairs,robust,groups)
    print('Fixed199:',{v:round(summaries['fixed199']['expanded'][v]['ADE15'],3) for v in methods},flush=True)
    print('Near87:',{v:round(summaries['strict_near87']['expanded'][v]['ADE15'],3) for v in methods},flush=True)

def report(summaries,pairs,robust,groups):
    REPORT.mkdir(parents=True,exist_ok=True);stats=read(DATA/'summary.json')
    lines=['# 用户筛除后的新 G1 数据扩充实验','',
        '用户指定排除 010、012、015、016、017、019、037、038、058、069、070、074、083；83 重复输入一次，按一个片段处理。原始视频与 mask 保留，排除仅体现在本轮训练名单。','',
        '**本轮结论：直接合并没有带来整体改善。** 固定 199 窗口上，六组 ADE 较扩充前完整训练库均上升 0.29–2.41 像素；录制等权汇总也全部变差。相对上次分布匹配子集，六组在全部测试和近人子组上均变差。','',
        '近人 G1 子组中，E 单向交互 ADE 从 123.04 降至 121.89，录制等权和稀疏统计也改善，但 FDE 上升 4.87 像素，三个种子中只有一个改善；仍不及上次匹配子集的 121.09。其余五组近人窗口均值上升。不能据此称扩充普遍无效，也不能称 E 已获得稳定全面提升。','',
        '多数模型的按录制重抽样区间跨过零，F 在完整测试上的区间为正。现有证据支持“此次直接合并在固定预算下未改善旧测试”的有限结论；不支持“数据太多导致变差”或“新标注差就是原因”。候选下一步是固定新旧抽样比例的对照，本轮没有追加执行。','',
        '## 数据去向','',
        f'- 原抽样为 200 个时间窗口，合并为 114 段。剔除后保留 {stats["retained_clips"]} 段 / {stats["remaining_selected_time_windows"]} 个原抽样时间窗口，其中无人、不适合当前场景或无连续轨迹的窗口不补选。',
        f'- 原有 169 个人物×21帧连续候选，用户排除涉及 39 个，余下 130 个。按既有完整边界、尺度及 mask 质量规则筛除 50 个贴边窗口，再排除 3 个存在被拒负点供体的窗口；两项有 1 个重叠，最终新增 **{stats["accepted_person_windows"]} 个人物预测窗口**。',
        f'- 新样本来自 {stats["contributing_clips"]} 段、{stats["contributing_time_windows"]} 个时间窗口、{stats["clip_local_ids"]} 个片段内 ID、{stats["contributing_recordings"]} 次录制。ID 不是全局人数，重叠窗口不是独立样本。新增训练录制 {len(stats["new_recordings"])} 次。',
        f'- 旧数据文件包含 {stats["old_prepared"]} 个窗口，实际参与各折训练的并集为 {stats["previous_actual_train_union"]} 个；合并文件为 {stats["combined_prepared"]} 个，实际训练并集为 **{stats["combined_actual_train_union"]} 个**。保留此前未被任何训练折使用的那一个窗口的原有状态。',
        '- 只在原来锁定的 200 个采样位置取窗口，不额外移动起点凑数量；每个人连续同一标注段 21 帧，不插值，不补造缺失中心。',
        '- 采用 review_v2 原始 mask 的独立解码外接框中心，统一右下边界为 exclusive；保持旧数组完全相同。新数据 review_v2 只筛选未改动的原 mask，因此新样本的 reviewed 与 pre_review 历史中心相同。',
        '- 每折只加入不属于该折测试和验证录制的新样本，防止同录制进入训练。各折新增 64–78 个，训练数量由 166–279 增至 237–351 个。','',
        '## 固定比较条件','',
        '主对照为扩充前 matched97_new 实际完整训练库，各折旧训练名单逐项保留，仅追加本轮合格样本。上次分布匹配子集另列为参考：它的数据量、筛选和采样方式均不同，不能把二者差异单独归因为这批新增数据。','',
        '六种架构、三个种子（17、29、43）、120×2 次更新、batch 48、录制等概率再在录制内抽窗口、输入增强及验证选模完全沿用主对照。共训练 198 个行人模型，复用相同 33 个冻结相机模型。历史 6 点约 0.5 秒，预测 15 点约 1.5 秒。','',
        '原 199 个测试窗口及各折验证窗口逐项不变；严格近人 G1 子组仍为原先固定的 87 个，没有根据本轮误差重选。未来相机参考仅用于另存的诊断与监督，不进入常规推理。','']
    for scope,title in [('fixed199','固定 199 窗口'),('strict_near87','固定近人 G1 的 87 窗口')]:
        lines += ['## '+title,'','单位像素，越低越好；三个模型种子的平均。','',
            '| 模型 | 扩充前 ADE | 此次 ADE | 变化 | 上次匹配子集 ADE | 扩充前 FDE | 此次 FDE |','|---|---:|---:|---:|---:|---:|---:|']
        for v in VARIANTS:
            a=summaries[scope]['previous'][v];b=summaries[scope]['expanded'][v];c=summaries[scope]['previous_matched_subset'][v]
            lines.append(f'| {NAMES[v]} | {a["ADE15"]:.2f} | {b["ADE15"]:.2f} | {b["ADE15"]-a["ADE15"]:+.2f} | {c["ADE15"]:.2f} | {a["FDE15"]:.2f} | {b["FDE15"]:.2f} |')
        lines += ['',f'保持最后位置 ADE：{summaries[scope]["previous"]["hold_last"]["ADE15"]:.2f}。','',
            '| 模型 | 录制等权 ADE 前→后 | 稀疏窗口 ADE 前→后 | 改善录制数 | 种子17/29/43的ADE变化 | 录制簇95%区间 |','|---|---:|---:|---:|---|---|']
        for v in VARIANTS:
            a=summaries[scope]['previous'][v];b=summaries[scope]['expanded'][v];p=pairs[scope]['previous'][v];lo,hi=p['cluster_bootstrap_95'];sd='/'.join(f'{x:+.2f}' for x in p['per_seed'].values())
            lines.append(f'| {NAMES[v]} | {a["recording_equal"]["ADE15"]:.2f}→{b["recording_equal"]["ADE15"]:.2f} | {a["sparse"]["ADE15"]:.2f}→{b["sparse"]["ADE15"]:.2f} | {p["recording_wins"]}/{len(p["per_recording"])} | {sd} | [{lo:+.2f}, {hi:+.2f}] |')
        lines+=['','区间按录制重抽样 4,000 次，对种子先平均；属于当前录制集合上的敏感性统计，不是独立新增测试的显著性证明。近人子组中 53/87 窗口来自 192702，故须同时看录制等权和稀疏结果。','']
    lines+=['## 数据分布变化','','相机平移速度只用于历史 EDGC 描述，不新增网络输入。旧/新训练列按各折录制平衡采样的期望权重平均；新增池列为窗口等权，近人调查列是概率权重参考。','',
        '| 相机速度 米/秒 | 近人调查 | 扩充前训练 | 新增池 | 合并后训练 |','|---|---:|---:|---:|---:|']
    for j,label in enumerate(['<0.2','0.2–1','1–1.6','≥1.6']):
        lines.append('| '+label+' | '+' | '.join(f'{100*groups[a]["camera_bins"][j]:.1f}%' for a in ['near_survey','previous','added_equal','expanded'])+' |')
    lines+=['','| 历史特征高组占比 | 近人调查 | 扩充前训练 | 合并后训练 |','|---|---:|---:|---:|']
    for j,k in enumerate(ORDER):lines.append('| '+k+' | '+' | '.join(f'{100*groups[a]["high_shares"][j]:.1f}%' for a in ['near_survey','previous','expanded'])+' |')
    lines+=['','E：框高对数增长率；D：相对框高；G：补偿后纵向速度；C：相机平移速度。阈值沿用此前锁定版本。此次没有重新匹配分布，新增录制也会改变录制平衡采样下旧样本的曝光概率。',
        '',f'新增窗口占合并训练并集的 {100*78/373:.1f}%，但按未改变的录制平衡采样器，11 折等权平均的预期训练抽样占比为 {100*read(OUT/"distribution_comparison.json")["added_expected_training_share"]:.1f}%。原因是新增 7 次录制，每个录制均获得相同的第一层抽样机会。旧样本的训练曝光因此明显减少；这仍是固定总更新预算下的数据扩充实验。','',
        '## 解释范围','',
        '该实验直接回答：在固定模型、计算预算和旧测试集下，把用户筛选后能形成连续窗口的新标注加入原训练池，预测误差如何变化。数据量、场景分布、标签误差以及每个样本的训练曝光同时变化，不能仅凭改善或下降判定标签质量就是原因，也不能推断数据越多必然越好或越坏。','',
        '用户的片段级筛除和已有数值/抽样检查提高了可用性，但不构成逐帧独立真值；边界完整、同一自动 ID 连续、与提示框一致，也不保证身份与身体轮廓完全正确。排除当前负点来源标记不证明已消除所有历史记忆污染。本轮保留这些限制，没有在观察新测试结果后调整数据阈值、架构或训练预算。','',
        '## 保存与核验','',
        '- selection_locked.json / locked_hashes.json：用户排除、资格规则、最终名单及冻结哈希。',
        '- data/candidate_decisions.json：169 个人物窗口逐个纳入或排除的原因；未形成完整候选的时间窗口仍可在原标注台账中查找。',
        '- folds.json：每折原名单、追加名单、验证和测试名单。',
        '- comparison.json / paired_cases.json：六模型的配对变化、各录制/种子/稀疏敏感性和逐窗口变化。',
        '- verification.json：198 个模型、594 份预测重放，删除未来后缀的推理隔离，以及源文件哈希核验。',
        '- independent_audit.json：独立 ADE/FDE 重算、相机投影几何、排除及折隔离核查。',
        '- 独立误差重算最大差为 0，SciPy 旋转重投影最大差约 0.000455 像素；3,330 个受保护源文件哈希不变。全部 198 次训练耗时合计约 11.3 分钟，另有准备和核验时间。',
        '- 原始标注、被拒片段、旧训练结果和选定基线均保留。','']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines),'utf8')

if __name__=='__main__':
    if '--distribution-only' in __import__('sys').argv:distribution()
    else:main()
