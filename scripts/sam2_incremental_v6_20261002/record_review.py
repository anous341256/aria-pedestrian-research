from render import *
from common import digest
def main():
    seed_checks=0
    for k in range(1,11):
        cid=f'{k:02d}';rows={(r['frame'],r['id']):r for r in load(OUT/cid/'masks_rle.jsonl.gz')};batch_of={}
        files=sorted((OUT/cid).glob('batch_*/input.json'))
        if cid=='10':files.extend(sorted((OUT/'reviewed'/cid).glob('batch_*/input.json')))
        for path in files:
            info=read_json(path)
            for e in info['entries']:batch_of[e['id']]=info['batch']
            for seed in read_json(path.parent/'seeds.json'):
                for p in seed['negative_points']:
                    assert p['frame']==seed['frame'];assert batch_of[p['donor_id']]<info['batch'];assert p['donor_id']!=seed['id'];mask=decode(rows[(p['frame'],p['donor_id'])]['rle']);x,y=p['xy'];assert mask[y,x];seed_checks+=1
        original=SRC/cid/'input';p=read_json(original/'protocol.json');assert p['initial_masks_sha256']==digest(original/'initial_masks.json')
    base={(r['frame'],r['id']):r['rle'] for r in load(OUT/'10/masks_rle.jsonl.gz')};refined={(r['frame'],r['id']):r['rle'] for r in load(OUT/'reviewed/10/masks_rle.jsonl.gz')};assert all(refined[k]==v for k,v in base.items());assert len(refined)==len(base)+2
    write_json(OUT/'review_checks.json',dict(status='passed',seed_negative_points_checked=seed_checks,original_seed_files_unchanged=10,original_clip10_mask_rows_preserved=len(base),added_person='P028',added_frames=[0,1]))
    report=ROOT/'reports/sam2_incremental_v6_20261002/RESULTS.md';body=report.read_text(encoding='utf8')
    update='''
## 二次回查与同条件负点对照

二次回查补了 **10 号 P028**：原图第 0、1 帧贴近相机的黑上衣行人，第 2 帧已完全出画。初轮“至少三次检测”的启动规则漏掉了这个清晰但短暂的目标。人工核查可见范围并给出首帧全可见身体框，仍用第 6 版传播到第 1 帧，随后按已核查出画停止。两个 mask 都已检查；原来的 134 条候选轨迹未改动。**合计 135 条候选轨迹、52 批，其中最后一批为人工回查补入的一人两帧**，并非去重真值人数。

优先观看：[回查补标后的 53 秒完整合集](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_all_people_reviewed.mp4)、[对应前后对比](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_before_after_reviewed.mp4)、[对应负点与预测框](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_prompts_predictions_reviewed.mp4)。补标数据在 reviewed/10，初轮目录和视频全部保留。逐批增长视频仍展示原始自动轮次。

另在 04 号固定相同 10 个候选、相同初始化框/帧和四批顺序，只关闭跨批负点作对照。开负点的跨 ID 高重叠告警为 **8 对-帧**，关负点为 **19 对-帧**；非空 mask 分别为 244 和 245。说明在这个固定计划的个案中，负点减少了部分重复占据，但不能据此宣称身份准确率提高：两边的 A 末段仍有多人合并，B 仍漏跟。312 条输出中 257 条完全一致，改变主要集中在后批候选。

[04 号仅开关负点的同步对照](E:/New_research/outputs/sam2_incremental_v6_20261002/controls/04_negative_on_off.mp4)。这是一例历史困难片段诊断，不是无偏数据集指标。

最终共生成并完整解码核验 43 个视频文件（含逐段、合集、分批重放、对照与回查版）。主轮负点、初始化负点、原始种子文件不变性、回查前后原始 mask 不变性均已核验，记录见 review_checks.json、presentation/verification.json 和 additional_media_verification.json。
'''
    if '## 二次回查与同条件负点对照' not in body:report.write_text(body+update,encoding='utf8')
    start=OUT/'presentation/START_HERE.md';txt=start.read_text(encoding='utf8')
    prefix='''# 优先看回查后的版本

[完整多人合集（补入两帧短暂人物）](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_all_people_reviewed.mp4)

[负点与预测位置框](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_prompts_predictions_reviewed.mp4) · [与原少量人物同步对比](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_before_after_reviewed.mp4)

共 135 条候选轨迹、52 批，包含 1 条人工回查补入的两帧轨迹。人数未完全去重，仍有漏跟和身份错误。以下是初轮自动结果，保留供比较。

'''
    if not txt.startswith('# 优先看'):start.write_text(prefix+txt,encoding='utf8')
    summary=read_json(OUT/'summary.json');summary['reviewed_pass']=dict(provisional_tracks=135,batches=52,added_short_person='10/P028',added_frames=2,manual_review_seed=True,mask_rows_preserved=True);write_json(OUT/'summary.json',summary)
    index=ROOT/'reports/sam2_tracking_history/INDEX.md';txt=index.read_text(encoding='utf8');note='''
## 第 6 版增量标注扩展 — 2026 年 10 月 2 日

用户批准取消人物总数限制，用第 6 版每批最多三人、前批 mask 给后批负点，处理同一十段 G1。主轮完成 51 批、134 条候选轨迹，另经人工回查补入一条两帧前景轨迹，合计 135 条候选、52 批。底层评分和记忆规则保持第 6 版；这是外层增量标注实验，不将旧版改名或替换。

04 号固定种子与批次、仅开关负点的对照中，高重叠告警从 19 降到 8 对-帧，但多人合并、漏跟、背景漂移和跨批重复身份仍存在，不宣称完成全量可靠标注。[结果与失败分析](E:/New_research/reports/sam2_incremental_v6_20261002/RESULTS.md) · [53 秒回查版合集](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_all_people_reviewed.mp4) · [所有视频入口](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/START_HERE.md)。原视频、旧版标注、初轮自动输出及本次负点对照均保留。
'''
    if '## 第 6 版增量标注扩展' not in txt:index.write_text(txt+note,encoding='utf8')
    reg=ROOT/'reports/sam2_tracking_history/registry.json';data=read_json(reg);entry=dict(id='incremental_v6_20261002',name='V6 incremental batches with previous-batch negative prompts',base_version=6,clips=10,automatic_provisional_tracks=134,automatic_batches=51,reviewed_provisional_tracks=135,reviewed_batches=52,report='reports/sam2_incremental_v6_20261002/RESULTS.md',output_directory='outputs/sam2_incremental_v6_20261002',evaluation_status='Automatic attempt plus visual review; no exhaustive GT; known ID, mask and coverage failures retained')
    data.setdefault('extensions',[])
    if not any(e['id']==entry['id'] for e in data['extensions']):data['extensions'].append(entry)
    data['latest_extension']=entry['id'];data['updated']='2026-10-02';write_json(reg,data)
    plan=ROOT/'plans/TRACKING_INCREMENTAL_V6.md';txt=plan.read_text(encoding='utf8')
    if '## 本轮完成记录' not in txt:plan.write_text(txt+'\n## 本轮完成记录\n\n十段主轮 51 批、134 条候选已运行；二次回查补入一条两帧目标，共 135 条候选、52 批。已生成并核验 43 个视频。04 号固定计划负点开关对照减少高重叠告警，但没有解决早批合并及跨批身份问题。详见 [结果](E:/New_research/reports/sam2_incremental_v6_20261002/RESULTS.md)。下一步需要双向回查、候选到身份的归并及弱小/短暂目标入口，尚未据此改动第 6 版。\n',encoding='utf8')
    memory=ROOT/'AGENTS.md';txt=memory.read_text(encoding='utf8');heading='## Approved V6 incremental all-person trial — 2026-10-02'
    if heading not in txt:
        note=heading+'''\n\nThe user authorized removing the total person cap on the same ten G1 clips, using V6 in batches of up to three and earlier-batch current masks as SAM2 negative points. The automatic trial completed 51 batches and 134 provisional tracks; a visually reviewed two-frame foreground person was added as batch 52 (135 provisional tracks in total). These are not unique-person counts or complete reliable GT. Original V6 scoring/memory and old outputs are preserved. Known failures include early-batch merged masks, small-target background drift, duplicate identities, and missed short people. A fixed-seed negative-on/off diagnostic on clip 04 reduced overlap alerts (19 to 8 pair-frames), not proven ID accuracy. See `reports/sam2_incremental_v6_20261002/RESULTS.md`, `plans/TRACKING_INCREMENTAL_V6.md`, and the history index. Do not silently treat these outputs as validated training labels.\n\n'''
        txt=txt.replace('\n\n','\n\n'+note,1);memory.write_text(txt,encoding='utf8')
    print('review record, checks and history saved',seed_checks,flush=True)
if __name__=='__main__':main()
