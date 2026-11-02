from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import subprocess, cv2, json, textwrap
from PIL import Image,ImageDraw,ImageFont
from audit import ROOT,OUT,read,save

NAMES={'G1':'相对简单 / 基础对照','G2':'相机运动突出','G3':'多因素混合'}
SHORTLIST=['150809_110_115','165112_090_100','201129_015_025']

def clip(s):
    out=OUT/'clips'/f"{s['key']}.mp4"; out.parent.mkdir(exist_ok=True)
    src=s['video']; dest=str(out)
    ffmpeg='E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
    args=[ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-n','-ss',str(s['start_s']),'-i',src,'-t',str(s['end_s']-s['start_s']),'-an','-vf','scale=960:720','-c:v','libx264','-threads','3','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',dest]
    if not out.exists():subprocess.run(args,check=True,capture_output=True)
    cap=cv2.VideoCapture(str(out)); frames=0
    while cap.grab():frames+=1
    fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    assert frames==round((s['end_s']-s['start_s'])*10),(out,frames)
    return {'key':s['key'],'path':str(out),'frames':frames,'fps':fps,'source':s['video'],'start_s':s['start_s'],'end_s':s['end_s'],'domain':'original fisheye RGB, resized only; no annotations or stabilization'}

def figure(segments):
    im=Image.new('RGB',(1500,1010),'#f4f6f8');d=ImageDraw.Draw(im)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    d.text((32,18),'现有 Aria 数据：三类片段初步划分',font=font(32),fill='#142438')
    d.text((32,66),'29 段录制已抽样复核；以下为代表候选，同一录制可包含不同类别。',font=font(20),fill='#536477')
    colors=['#29836b','#b37616','#ad4960']
    labels=['相对简单（边界候选）','相机运动突出','多因素混合']
    captions=['目标较远、多在候车；用于弱交互对照','主要目标连续前行；转头明显','多人、转头、夜间及出入视野叠加']
    times=[[110.4,114.4],[92.6,97.8],[15.5,21.2]]
    for i,key in enumerate(SHORTLIST):
        s=next(s for s in segments if s['key']==key);x=24+i*496
        d.rounded_rectangle((x,112,x+472,980),radius=14,fill='white')
        d.text((x+14,128),labels[i],font=font(24),fill=colors[i])
        d.text((x+14,165),f"{s['recording']} · {s['start_s']}–{s['end_s']} 秒",font=font(20),fill='#34465d')
        cap=cv2.VideoCapture(s['video'])
        for j,t in enumerate(times[i]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*10));ok,bgr=cap.read();assert ok
            tile=Image.fromarray(cv2.cvtColor(cv2.resize(bgr,(448,336)),cv2.COLOR_BGR2RGB))
            y=207+j*364;im.paste(tile,(x+12,y));d.text((x+20,y+8),f'{t:.1f}s',font=font(20),fill='white',stroke_width=2,stroke_fill='black')
        cap.release()
        # Manually split by length to keep the fixed figure compact.
        msg=captions[i]
        for k,line in enumerate(textwrap.wrap(msg,width=19)):
            d.text((x+14,934+k*25),line,font=font(19),fill='#34465d')
    im.save(OUT/'three_classes.jpg',quality=94)

def main():
    data=read(OUT/'classification.json');inventory=read(OUT/'motion_inventory.json')
    segs=data['segments']; selected=[next(s for s in segs if s['key']==k) for k in SHORTLIST]
    with ThreadPoolExecutor(max_workers=2) as pool: clips=list(pool.map(clip,selected))
    save(OUT/'clip_manifest.json',clips);figure(segs)
    nvisual=29*12+sum(s['reviewed_sample_count'] for s in segs)
    bands={b:round(sum(r['motion_band_seconds'][b] for r in inventory),1) for b in ['low','moderate','strong','unknown']}
    sensitivity=[]
    for angle,omega in [(10,25),(15,40),(20,50)]:
        seconds=sum(w['end_s']-w['start_s'] for r in inventory for w in r['windows'] if w['pose_coverage']>=.8 and w['end_s']-w['start_s']>=5 and w['max_1s_rotation_deg'] is not None and w['max_1s_rotation_deg']<=angle and w['omega_p95_deg_s']<=omega)
        sensitivity.append({'max_1s_rotation_deg':angle,'p95_speed_deg_s':omega,'duration_s':seconds})
    summary={'recordings':29,'duration_s':sum(r['duration_s'] for r in data['recordings']),'existing_sample_frames_reinspected':348,'additional_segment_tiles_inspected':nvisual-348,'dense_segments':18,'mps_recordings':sum(r['mps_source'] is not None for r in inventory),'whole_recording_routing_counts':dict(Counter(r['primary_class'] for r in data['recordings'])),'motion_band_seconds':bands,'low_motion_threshold_sensitivity':sensitivity,'note':'Whole-recording routing counts and rotation-band durations are NOT final three-class frame proportions. No exhaustive segmentation, model run, or accuracy evaluation.'}
    save(OUT/'summary.json',summary)
    rep=ROOT/'reports/dataset_threeway_20261001';rep.mkdir(exist_ok=True)
    md=['# Aria 数据集三分类初步审查 — 2026-10-01','',
    '**结论：现有数据确实偏向混合困难场景。应按时间片段分类；三类素材明显不均衡，不能把整段视频按地点硬分为 Easy / Hard / Uncontrolled。**','',
    '本次覆盖全部 **29 段、2853.6 秒（47.56 分钟）**。重新检查既有的每段12帧，共348帧；另对18个候选时段生成并检查较密集的画面序列（共'+str(nvisual-348)+'个画面，包含少量与原抽样重合的时刻）。读取28段录制的离线MPS姿态，162338缺SLAM。没有逐帧观看47分钟全部视频；以下是可修订的回顾性分类，未生成新的行人GT，也未重新运行跟踪模型。','',
    '## 分类定义与决策顺序','',
    '| 类别 | 判定依据 | 研究用途 |', '|---|---|---|',
    '| G1 相对简单/基础对照 | 相机相对稳定，选定目标周围交互和遮挡弱，可持续观察 | 基础跟踪、标注质量、无强转头时的预测 |',
    '| G2 相机运动突出 | 选定目标交互较弱，但转头/俯仰等明显 | 检查运动补偿与位置提示是否有帮助 |',
    '| G3 多因素混合 | 多人交叉、明显遮挡/截断、低照度/曝光、人像干扰、上下楼等与相机运动共同出现 | 检查身份保护、丢失恢复及真实场景鲁棒性 |','',
    '先判断是否存在持续可用的真实行人；无人、只看导览牌或只剩模特的片段记为“不适用”，不强塞进G1。再分别记录相机运动、目标周围交互、可见性和场景因素，最后给三类主标签。画面总人数不等于目标周围干扰；只标注5个人也不等于场景只有5个人。','',
    '分类借鉴HEADS-UP第III-B节和图3，但这是自然录制的事后划分，不是该论文的受控采集协议，也不是已证明错误来源的因果分类。G2仍含相机平移、视差和行人运动，不能称为纯旋转场景。','',
    '## 优先查看的三个例子','',
    f'![三类示例]({(OUT/"three_classes.jpg").as_posix()})','',
    '| 暂定类别 | 录制与区间 | 观察与限制 | 预览 |','|---|---|---|---|']
    for s,c in zip(selected,clips):
        md.append(f"| {NAMES[s['class']]} | {s['recording']}，{s['start_s']}–{s['end_s']}秒 | {s['observation']} {s['caveats']} | [视频]({Path(c['path']).as_posix()}) |")
    md+=['','**G1最欠缺。** 150809的110–115秒只是相对简单的远人/候车对照，不能充当已经充分具备的“稳定相机＋清晰近人自由步行”组。G2首选165112的90–100秒。G3有丰富素材，当前201129五人片段明确属于这一组。','',
    '## 为什么不能只用相机稳定度筛选','',
    '- 164557的180–185秒：镜头很稳，但主要在看导览牌，不能作为简单行人预测样本。',
    '- 201451的40–50秒、55–60秒：旋转较小，但佩戴者低头看手机，前人背包和下半身占据视野，上身严重截断，且在扶梯上。',
    '- 151153的120–125秒：相对稳定，但近人很快出画，随后主要为空站厅。',
    '- 170022的90–100秒：检测人数少不能说明简单；行人短暂经过后，画面主要是橱窗、模特和空走廊。','',
    '## 相机运动证据及其限度','',
    '使用原始导出视频的VRS时间映射，最近邻对齐MPS闭环设备姿态（≤30ms、质量>0.5）；同一图内计算四元数相对旋转角，除以相邻RGB时间差。固定设备—相机外参不改变旋转角的模长。重复时间戳不参与速度计算。每5秒统计95%角速度及窗口内最大1秒旋转角；有效覆盖不足80%记未知。没有测量相机平移，也未将低头、侧转和滚转分别统计。','',
    '| 运动初筛 | 时长（秒） | 条件 |','|---|---:|---|',
    f'| 低旋转 | {bands["low"]} | 最大1秒旋转≤10°，且95%角速度≤25°/s |',
    f'| 中等 | {bands["moderate"]} | 位于低与强之间 |',
    f'| 强旋转 | {bands["strong"]} | 最大1秒旋转≥25°，或95%角速度≥60°/s |',
    f'| 未知/覆盖不足 | {bands["unknown"]} | 不能当作稳定或不稳定 |','',
    '**这些是自定筛选阈值，不是HEADS-UP的标准，也不是最终三类比例。** 仅考虑完整5秒且姿态覆盖足够的窗口，将低旋转门槛从10°/25°每秒放宽到15°/40°每秒、20°/50°每秒时，候选时长分别为20、40、105秒。说明数值很依赖定义；本次还要结合人物可见性复核，不能宣称全数据只有20秒简单场景。','',
    '## 全部29段的录制级暂归类','',
    '录制级标签仅用于安排查看顺序，不表示整段每一帧属于同一类。保守路由为：G1无整段，G2一段，G3二十八段；局部G1/G2候选保留在下表和片段清单。没有把该计数换算为时间分布。','',
    '| 编号 | 时长秒 | 暂归类 | 场景及依据 |','|---|---:|---|---|']
    for r in data['recordings']:md.append(f"| {r['id']} | {r['duration_s']:.1f} | {r['primary_class']} | {r['observation']} |")
    md+=['','## 18个候选时段的逐段判断','','所有时间相对于各自 `E:/Aria/parsed/<recording>/logs/214-1/video.mp4` 的视频起点。当前实验的校正视频来自另一次导出，使用时需按VRS时间对齐，不能盲目复用帧号。片段边界是审查区间，不是逐帧标注的事件起止。','','| 录制/区间 | 类别 | 用途状态 | 观察与保留条件 |','|---|---|---|---|']
    for s in segs:
        md.append(f"| [{s['recording']} {s['start_s']}–{s['end_s']}秒]({Path(s['sheet']).as_posix()}) | {s['class'] or '不适用'} | {s['suitability']} | {s['observation']} {s['caveats']} |")
    md+=['','## 对当前研究的影响','',
    '最近整段SAM2和旋转补偿可行性实验使用的151153、162538、164557、193709、201129，都包含混合因素。这不推翻已有案例中观察到的收益，但它们不是三类均衡样本，不能直接回答收益究竟主要出现在哪类场景。','',
    '现阶段至少同时面对：①相机引起的位置/尺度变化；②多人身份交叉、遮挡与再出现；③观测质量、几何可见性和数据筛选。当前可靠性先验通过减少部分错认改善了身份保护；不能仅据这一点归因于旋转补偿。实际跟踪错误也可能来自控制规则自身，例如过严记忆门槛，场景类别不会自动解释这些实现问题。','',
    '下一轮比较可分别在G1、G2、G3中统计错认/ID转移、漏跟、恢复时间与有效连续长度，再看预测ADE/FDE；选段和标签要在查看新方法结果前固定。若同源录制被切成多段，应按录制（必要时按同次路线/会话）分训练测试，不能随机把相邻片段分到两边。以上是研究建议，本次没有启动新实验或调整已有数据划分。','',
    '优先缺口是稳定相机、清楚可见且确实行走的低交互样本；其次是相近光照/场景下的单纯转头样本。当前自然场景素材可继续作为混合组，不必废弃或整体重新采集。','',
    '## 文件与复现','',
    f'- [结构化分类清单]({(OUT/"classification.json").as_posix()})',
    f'- [运动统计与完整5秒窗口]({(OUT/"motion_inventory.json").as_posix()})',
    f'- [筛选规则]({(OUT/"protocol.json").as_posix()})',
    f'- [摘要和阈值敏感性]({(OUT/"summary.json").as_posix()})',
    f'- [视频预览清单]({(OUT/"clip_manifest.json").as_posix()})',
    '- 原始VRS、导出视频、全部旧标注和实验输出保持原位。新增文件仅在本次独立的scripts/outputs/reports目录。','']
    (rep/'RESULTS.md').write_text('\n'.join(md),encoding='utf-8')
    verification={'recordings':len(data['recordings']),'unique_ids':len({r['id'] for r in data['recordings']}),'all_videos_exist':all(r['video_exists'] for r in inventory),'all_review_sheets_exist':all(Path(s['sheet']).exists() for s in segs),'all_segment_ranges_valid':all(0<=s['start_s']<s['end_s']<=next(r['prior_export_duration_s'] for r in inventory if r['id']==s['recording']) for s in segs),'all_pose_graph_counts_single':all(r.get('pose_graph_count',1)==1 for r in inventory),'duration_partition_matches':abs(sum(bands.values())-summary['duration_s'])<1e-6,'clips_decoded':[{'key':c['key'],'frames':c['frames'],'fps':c['fps']} for c in clips],'labels_not_ground_truth':True}
    assert all(v for k,v in verification.items() if k.startswith('all_') or k=='duration_partition_matches')
    save(OUT/'verification.json',verification)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
