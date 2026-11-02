"""Readable findings using candidate failure counts, including failed repairs."""
import collections
from selection import ROOT, OUT, read, write

def main():
    data=read(OUT/'diagnostics.json'); totals=data['totals']
    names={'visual':'A 视觉记忆','external_box':'B 外部预测框',
           'camera':'C 内部旋转引导','camera_person':'D 内部旋转＋目标运动'}
    rows={}
    for arm,t in totals.items():
        failed=t['propagated']-t['raw_valid']
        assert failed==t['repairs']+t['pending']
        grouped=collections.defaultdict(lambda:[0,0])
        for c in data['per_clip'][arm]:
            grouped[c['recording']][0]+=c['propagated']-c['raw_valid']
            grouped[c['recording']][1]+=c['propagated']
        rows[arm]=dict(propagated=t['propagated'], initial_failures=failed,
            initial_failure_fraction=failed/t['propagated'],
            successful_repairs=t['repairs'],pending=t['pending'],
            recording_equal_failure_fraction=sum(f/n for f,n in grouped.values())/len(grouped),
            by_recording=dict(grouped))
    contrasts={}
    for ref,new in [('visual','external_box'),('visual','camera'),('visual','camera_person'),('camera','camera_person')]:
        diffs=[rows[new]['by_recording'][r][0]-v[0] for r,v in rows[ref]['by_recording'].items()]
        contrasts[new+'_minus_'+ref]=dict(fewer=sum(d<0 for d in diffs),
            same=sum(d==0 for d in diffs),more=sum(d>0 for d in diffs))
    write(OUT/'initial_failure_summary.json',dict(arms=rows,recording_contrasts=contrasts,
        meaning='Failure of the common automatic candidate filter, not measured mask or identity error.'))
    lines=[
        'SAM2 内部运动引导第一版：结果简报',
        '',
        '结论：内部旋转引导使自动补救需求小幅减少；增加目标运动后未见额外总体收益。',
        '在这项自动诊断上，外部预测框组明显更低。尚不能据此认定任一组的真实身份准确率。',
        '',
        '191个不重叠时间段，21个录制，每段21帧（2秒首尾跨度），四组各13579次非初始化传播。',
        '1017个目标段实例，按原12%纳入/8%保留规则标注近人，无总人数上限。SAM2未训练。',
        '',
        '方法 | 首次候选未通过，需补救的人帧数 | 比例 | 补救后仍待复核',
    ]
    for a,r in rows.items():
        lines.append(f"{names[a]} | {r['initial_failures']} | {r['initial_failure_fraction']:.2%} | {r['pending']}")
    lines+=['',
        '“需补救”包括补救成功和仍待复核两种情况，不等于真实跟错人数。',
        '各录制等权统计的首次未通过比例：']
    for a,r in rows.items():
        lines.append(f"{names[a]}：{r['recording_equal_failure_fraction']:.2%}")
    lines+=['', '按录制比较补救需求（减少/相同/增加）：']
    for k,v in contrasts.items():
        lines.append(f"{k}: {v['fewer']}/{v['same']}/{v['more']}")
    lines+=['',
        '如何解释：',
        '1. 内部空间先验能够改变输出，但当前固定强度、无需训练的实现只带来小幅自动指标变化。',
        '2. C与D总体接近，不能认为多加目标运动必然更好。',
        '3. B有原生记忆和预测框两种候选机会，C/D主要调整记忆读取。这里比较方案效果，并非等计算量实验。',
        '4. 修复后的最终mask数量接近，因为共同离线修复会填补不少传播失败；要同时看raw视频。',
        '5. 起始mask碎片、部分身体和遮挡仍是问题：057/N06是共享的低置信度碎片初始化；050/N05保留为身份待核查。',
        '',
        '数值限制：全零dense attention偏置也会改变计算后端，两个探针分别有39/31个像素变化；',
        '实际运动偏置相对同一dense后端有2889/709个像素变化。前者很小但并非严格为零，',
        '小范围数值检查不能排除闭环传播放大，因此不将A与C/D的全部差异纯粹归因于几何先验。',
        '',
        '验收：462个原始输入文件和原生SAM2源码未变；402个视频逐帧解码通过。',
        '共检查54316个raw mask、57743个最终mask；4068次起始mask导出与共享seed完全一致。',
        '未修改旧标签，未合并预测训练集。已保留成功、无明显变化及失败样例。',
        '',
        '先看：outputs/sam2_motion_memory_20261020/presentation/FIRST_21_RECORDINGS_raw.mp4',
        '修复后：同目录 FIRST_21_RECORDINGS_final.mp4',
        '完整索引：同目录 START_HERE.txt',
        '技术报告：reports/sam2_motion_memory_20261020/RESULTS.txt',
    ]
    path=ROOT/'reports/sam2_motion_memory_20261020/SUMMARY.txt'
    path.write_text('\n'.join(lines),encoding='utf8')
    print('\n'.join(lines[:20]))

if __name__=='__main__':
    main()
