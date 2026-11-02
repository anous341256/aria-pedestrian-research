"""Persist transparent census, review findings, and artifact navigation."""
from render import *
def main():
    allrows=[]
    for i in range(1,11):
        cid=f'{i:02d}';audit=read_json(OUT/cid/'audit.json');media=read_json(OUT/'presentation'/f'{cid}_media.json');inv=read_json(OUT/cid/'inventory.json')
        counts=collections.Counter(d['status'] for d in inv['proposal_disposition'].values())
        allrows.append(dict(**audit,max_simultaneous_masks=max(media['visible_per_frame']),proposal_status_counts=dict(counts),duration=media['frames']/10))
    summary=dict(clips=10,frames=530,seconds=53,recordings=6,original_initial_tracks=29,
        provisional_tracks=sum(r['provisional_ids'] for r in allrows),batches=sum(r['batches'] for r in allrows),negative_candidate_calls=sum(r['negative_candidate_calls'] for r in allrows),negative_point_checks=sum(r['verified_negative_points'] for r in allrows),nonempty_masks=sum(r['nonempty_masks'] for r in allrows),clips_detail=allrows,
        status='all ten clips processed; persistent seedable candidates exhausted; experimental annotation, unresolved quality failures retained',
        caveat='Provisional tracks can duplicate identities or include drift. Nonempty masks, discovered candidates, and overlap are not GT accuracy or full annotation recall.')
    write_json(OUT/'summary.json',summary)
    report=ROOT/'reports/sam2_incremental_v6_20261002';report.mkdir(parents=True,exist_ok=True)
    table='\n'.join(f'| {r["clip"]} | {r["provisional_ids"]} | {r["batches"]} | {r["max_simultaneous_masks"]} | {r["negative_candidate_calls"]} | {r["high_overlap_pairs_frames"]} |' for r in allrows)
    body=f'''# 第 6 版不限人数、分批负点标注实验

同一批十段 G1 视频已经完整运行。每批最多 3 人，前批当帧的可靠 mask 向后批提供 SAM2 负点；总人数不设 3/5 人上限。共 {summary['batches']} 批，尝试 {summary['provisional_tracks']} 条候选轨迹，原来是 29 条首帧初始化轨迹。覆盖 530 帧、53 秒、6 次录制。候选轨迹数不等于去重后的真实人数；存在重复、漂移及身份不明的结果。

## 先看视频

- [十段完整增量标注合集，53 秒](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_all_people.mp4)
- [原来少量人物与本次增量的同步对比，53 秒](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_before_after.mp4)
- [实际负点＋预测位置框，53 秒](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_prompts_predictions.mp4)
- [07 号逐批累加，整段反复播放](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/07_batch_growth.mp4)
- [10 号逐批累加，整段反复播放](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/10_batch_growth.mp4)

彩色实线与填色为 mask；虚线为当前帧预测位置；红叉为所选候选实际接收的前批负点。原来少量人物与本次结果的对比同时改变目标数量、初始化与分批方式，不能单凭它把变化归因于负点。

## 实际运行规模

| 片段 | 候选 ID | 批次 | 单帧最多非空 mask | 接收负点的候选调用 | 高重叠对-帧告警 |
|---|---:|---:|---:|---:|---:|
{table}

最后一列只表示不同 ID 的 mask 高度重叠，不等于身份错误数量；完全串到另一人且不重叠的错误不被它捕获。单帧 mask 数也不是准确率。06 号只发现两条可稳定初始化的候选，因此只有一批，不强凑人数。

## Pipeline 与第 6 版边界

1. 读取原先十段校正后的图像和 Aria 参考旋转。YOLO11l 在全部帧和每 0.5 秒的重叠局部裁剪中发现人物框。检测只供目标发现及初始化，最终输出来自 SAM2。
2. 对候选形成短轨迹并回查前批覆盖；沿用原来核查的 A–D 首帧种子，新人使用首次可靠检测框。初轮自动启动条件是至少 3 次检测、至少一次置信度 ≥0.35、框高 ≥24 像素；不是人数上限。短弱候选另列复核清单，未把这些清单都当新人。
3. 每批最多三人，第 6 版仍计算相机旋转补偿后的运动先验，生成原生记忆与运动框两类候选。对两个候选分别用负点和原候选 logits 再提示 SAM2。
4. 负点来自前批当前帧通过原版记忆写入规则的 mask，并要求当前检测支持。每个邻近供体取最多两个内部点，排除被多个供体同时占据的点。原图不抹黑，输出 mask 不相减。
5. 原版加权评分、硬阈值、批内候选一对一分配、运动更新、记忆门控不变；保存本批结果后继续发现尚未覆盖的候选。前批结果在初轮中冻结。

整段候选的持续性和跨批覆盖用于发现及去重，因此这是**离线标注流程**。单条第 6 版传播只用当前/历史图像及参考旋转，没有未来帧负点输入；不能把整体流程说成已经完成在线跟踪器。新增目标不是训练出的新检测/重识别网络。

## 已核查的现象与失败

- **覆盖扩大**：01、03、07、08 的抽样画面中可以看到原先未初始化的远处行人和中途入场人物；07 中也加入了近处儿童。07/10 适合观察较多目标逐批加入。
- **早批错误不会被单向负点自动修复**：04 的 A 在末段仍出现混合两个人的 mask，B 仍可能漏跟。早批没有接收后批提供的排除信息，这是当前数据流的局限。
- **小目标丢失后转向背景**：01 的 P008 后段覆盖车辆，02 的 P007 后段扩展到对侧建筑，03 也有远处候选偏到背景。核查的初始化 mask 很小（例如 02/P007 初始 2,140 像素）；高 SAM 质量分和 V6 运动/外观规则并不保证后续仍是人。
- **重复及部分 mask 仍存在**：同一个人可能被局部框、新轨迹片段或错误恢复重新编号，04 末段也有头部小 mask 与其他身份输出同时存在。负点并不是跨批全局身份分配。
- **负点的可靠性仍是先验判断**：当前供体规则能够拒绝部分可疑 mask，但不证明供体身份正确。错误供体仍可能排斥真正目标。自动告警和视觉确认分别保留，不把告警原因当作已证实的根因。

每段已检查五个均匀时刻的完整视图，并检查候选初始化及短弱候选裁剪。没有逐像素人工真值。实验已经跑到没有待启动的合格持续候选；**尚未达到所有可见人物逐帧正确、无重复的完整标注**。短弱候选及未解决漏跟均保留，不以强行赋 ID 冒充标完。

## 验证与保存

04 第一批三个人在没有前批负点时，120 条人物帧 RLE 与旧第 6 版逐项完全一致；核心代码哈希与原版一致。累计 {summary['negative_candidate_calls']} 次候选生成实际使用负点，{summary['negative_point_checks']} 个候选负点通过“更早批次、相同帧、供体 mask 内部”检查。拒绝记忆写入时，记忆帧集合保持不变；保存的先验只来自此前观测。逐视频完整解码结果见 presentation/verification.json。

每段目录包含 detections.json、proposal_tracklets.json、inventory.json、batch_XX/、masks_rle.jsonl.gz、decisions.jsonl.gz、audit.json、overlap_flags.json 和 weak_review.json。每批保留输入种子、SAM2 mask、候选评分、实际负点、记忆状态及执行时间。脚本位于 scripts/sam2_incremental_v6_20261002，协议与代码哈希随输出保存。原数据与全部旧版结果保留。

## 由本次失败提出的后续改动（尚未改动本次第 6 版）

先把“发现一个候选”与“建立新身份”分开；对局部框、已退出身份和重新出现的目标进行跨批核对。再做一次双向回查：后批完成后复核早批重叠/合并错误，限制不可靠标注充当负点。还需要检查远处小目标的 seed 质量和人物存在性，不能只通过增加标注人数获得可靠密集标签。
'''
    (report/'RESULTS.md').write_text(body,encoding='utf8')
    start=['# 本次增量标注入口','',f'十段视频，共 {summary["provisional_tracks"]} 条候选轨迹，{summary["batches"]} 批。候选身份仍需核查。','', '[完整多人合集](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_all_people.mp4)','[与原少量人物同步对比](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_before_after.mp4)','[负点和预测框版](E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/all_10_prompts_predictions.mp4)','[结果和失败分析](E:/New_research/reports/sam2_incremental_v6_20261002/RESULTS.md)','', '| 片段 | 全部 mask | 负点＋预测框 | 前后对比 |','|---|---|---|---|']
    for i in range(1,11):
        cid=f'{i:02d}';prefix=f'E:/New_research/outputs/sam2_incremental_v6_20261002/presentation/{cid}'
        start.append(f'| {cid} | [播放]({prefix}_all_people.mp4) | [播放]({prefix}_prompts_predictions.mp4) | [播放]({prefix}_before_after.mp4) |')
    (OUT/'presentation/START_HERE.md').write_text('\n'.join(start),encoding='utf8');print(summary['provisional_tracks'],summary['batches'],summary['negative_candidate_calls'],flush=True)
if __name__=='__main__':main()
