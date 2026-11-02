"""Export review assets and an auditable target-specific shortlist; no new masks."""
from screen import *
import subprocess

FF=Path('E:/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe')
REPORT=ROOT/'reports/g1_street_candidates_20261001'
# Labels are assistant visual judgements, NOT model predictions or ground truth.
NOTES={
1:('白上衣、黑长裤、浅色肩包，同向步行','目标前方偏中，沿人行道连续向前走；所看采样帧中没有明显遮挡。','黄昏、目标中远距离；与03为同一人物的不同时段。','192702_white_top_follow'),
2:('浅色帽、深色横条上衣、短裤，同向步行','选择前方偏右的戴帽者，几人前后错开，目标持续可见。','同伴近邻，末尾距离缩小；不能说完全没有互动。','192702_cap_group'),
3:('白色无袖上衣、黑长裤，同向步行','与01同一人物，这次距离更近，位于右前方，保留作尺度变化对照。','不是新增独立身份；末尾靠近其他行人。','192702_white_top_follow'),
4:('迎面黑上衣、蓝色长裤行人','两人并排行走时优先选左侧黑上衣者，截在近身通过之前。','仅4秒、开头目标较小，旁边有同行者。','192702_approach_pair'),
5:('中间浅灰上衣、深色长裤，背向步行','选择中间独立前行者，沿树木与建筑之间的人行道走。','树荫较暗、目标偏小，另有远处行人；身份尚未跨片段确认。','192702_far_follow'),
6:('迎面两名浅色上衣者，优先右侧一人','两人沿建筑边缘接近，横向分离，窗口内没有近身穿越。','夜色较暗、开头目标小，有车辆灯光。','193311_approach_pair'),
7:('深蓝上衣、浅色短裤，同向步行','白天沿街跟随，目标躯干清楚，同伴在左侧、儿童在右侧。','头部起伏较大；家庭结伴、推拉儿童乘坐物及车行出入口，保留边界。','164557_family'),
8:('白上衣黑长裤或白上衣灰裙，两人同向','白天目标大而清晰，沿人行道走，适合检查身份保持。','1秒内最大相机旋转约29°；底部有手机，末尾双人距离接近。','164535_pair'),
9:('黑上衣、浅色长裤、黑色斜挎包，同向','桥下步行跟随，前方单个目标持续可见。','桥下较暗、步行抖动较大；按原运动分档为strong。','193709_black_top_follow'),
10:('前方偏左黑上衣、浅粉长裤，背向步行','选沿墙一侧向前走的目标；周围主要为迎面经过者。','沿街有顶走廊，整体人流较多，G1/G3边界；不选近处迎面人或贴墙站立者。','155313_pink_trousers_follow'),
}
def encoder(path,width,height):
    return subprocess.Popen([str(FF),'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    for sub in ['clips','keyframes','frame_maps']:(OUT/sub).mkdir(exist_ok=True)
    rows=json.loads((OUT/'refined_candidates.json').read_text(encoding='utf8'))
    rows[0]['end_s']=189;rows[9]['end_s']=178
    rows[2]=json.loads((OUT/'replacement_03_final.json').read_text(encoding='utf8'))
    total=0;chapters=[];verification=[]
    montage_path=OUT/'street_walk_10_review.mp4';montage=encoder(montage_path,960,816)
    big=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    for row in rows:
        n=row['candidate'];sid=row['id'];s=row['start_s'];e=row['end_s'];row['duration_s']=e-s
        a=np.load(OUT/f'pose_{sid}.npz');sig=tuple(a[k] for k in ['rel','stamp','q','valid','pos'])
        row['metrics']=metrics(sig,s,e);row['sample_frames']=[f for f in row['sample_frames'] if s*10<=f<e*10]
        target,observation,caveat,episode=NOTES[n]
        row.update({'status':'G1步行候选（相对简单，待确认）' if n<=6 else '边界备选，不计入纯G1', 'target':target,'visual_observation':observation,'remaining_factors':caveat,'episode_group':episode,'recording_group':f'unknown_20260803_{sid}','source_video':INV[sid]['video'],'source_timestamp_map':INV[sid]['timestamp_map'],'source_mps':INV[sid]['mps_source'],'source_start_frame':s*10,'source_end_frame_exclusive':e*10,'preview_domain':'Unrectified original fisheye, resize only, 960x720 @10fps, muted. Read original full-resolution source for experiments.'})
        ts=list(csv.DictReader(open(INV[sid]['timestamp_map'],encoding='utf-8-sig')))
        fmap=[{'clip_frame':i,'source_frame':f,'mp4_time_ns':ts[f]['mp4_time_ns'],'vrs_device_time_ns':ts[f]['vrs_device_time_ns']} for i,f in enumerate(range(s*10,e*10))]
        row['frame_map']=str(OUT/'frame_maps'/f'{n:02d}.json');save(Path(row['frame_map']),fmap)
        row['clip']=str(OUT/'clips'/f'{n:02d}_{sid}_{s:03d}_{e:03d}.mp4')
        proc=encoder(Path(row['clip']),960,720)
        cap=cv2.VideoCapture(INV[sid]['video']);assert cap.get(cv2.CAP_PROP_FPS)==10
        cap.set(cv2.CAP_PROP_POS_FRAMES,s*10)
        band='优先候选' if n<=6 else '边界备选'
        banner=Image.new('RGB',(960,96),(19,28,40));d=ImageDraw.Draw(banner)
        d.text((18,9),f'{n:02d} / 10   {band}   |   {sid}   {s}–{e} 秒',font=big,fill='#8be1c5' if n<=6 else '#ffd18b')
        d.text((18,49),target,font=small,fill='white')
        for i in range((e-s)*10):
            ok,bgr=cap.read();assert ok,(n,i)
            rgb=cv2.cvtColor(cv2.resize(bgr,(960,720),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            proc.stdin.write(rgb.tobytes())
            combined=np.concatenate([np.asarray(banner),rgb],axis=0);montage.stdin.write(combined.tobytes())
            if i==(e-s)*5:
                row['keyframe']=str(OUT/'keyframes'/f'{n:02d}.jpg');Image.fromarray(combined).save(row['keyframe'],quality=93)
        cap.release();proc.stdin.close();assert proc.wait()==0
        chapters.append({'candidate':n,'montage_start_s':total,'montage_end_s':total+e-s});total+=e-s
        print(f'Exported {n:02d}: {sid} {s}-{e}',flush=True)
    montage.stdin.close();assert montage.wait()==0
    for path,expected in [(Path(r['clip']),r['duration_s']*10) for r in rows]+[(montage_path,total*10)]:
        cap=cv2.VideoCapture(str(path));count=0;fps=cap.get(cv2.CAP_PROP_FPS);size=[int(cap.get(k)) for k in [cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT]]
        while cap.grab():count+=1
        cap.release();assert count==expected,(path,count,expected);assert fps==10
        verification.append({'path':str(path),'decoded_frames':count,'expected_frames':expected,'fps':fps,'size':size})
    # Every source interval is disjoint, but recording and identity repetitions remain.
    for i,r in enumerate(rows):
        for z in rows[i+1:]:
            if r['id']==z['id']:assert min(r['end_s'],z['end_s'])<=max(r['start_s'],z['start_s'])
    overview=Image.new('RGB',(1600,780),(19,28,40));d=ImageDraw.Draw(overview);title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    d.text((22,15),'街道步行筛选：6 条优先候选 + 4 条边界备选',font=title,fill='white')
    for k,r in enumerate(rows):
        x=(k%4)*400;y=65+(k//4)*235
        im=Image.open(r['keyframe']).crop((0,96,960,816)).resize((304,228))
        # A 4:3 image occupies 272x204; labels stay outside image pixels.
        im=im.resize((272,204));overview.paste(im,(x+120,y+24))
        color='#8be1c5' if r['candidate']<=6 else '#ffd18b'
        d.text((x+7,y+30),f"{r['candidate']:02d}",font=title,fill=color)
        d.text((x+7,y+72),r['id'],font=font,fill='white')
        d.text((x+7,y+99),f"{r['start_s']}–{r['end_s']}s",font=font,fill='white')
        d.text((x+7,y+126),'优先候选' if r['candidate']<=6 else '边界备选',font=font,fill=color)
    overview.save(OUT/'overview.jpg',quality=92)
    save(OUT/'selected_clips.json',rows);save(OUT/'chapters.json',chapters)
    save(OUT/'verification.json',{'status':'passed','videos':verification,'disjoint_source_intervals':True,'source_recordings':len(set(r['id'] for r in rows)),'total_seconds':total,'dense_visual_sampling_hz':5,'not_ground_truth':True})
    summary={'candidate_clips':10,'priority_g1_like_candidates':6,'borderline_backups':4,'source_recordings':6,'duration_s':total,'daylight_clips':3,'dusk_clips':7,'same_identity_pair':[1,3],'no_new_SAM_or_evaluation':True,'all_previews_decoded':True}
    save(OUT/'summary.json',summary)
    lines=['# 街道步行 G1 候选扩充（2026-10-01）','',f'已整理 **10 条预览，来自 6 段录制，共 {total} 秒**。01–06 为较接近 G1 的步行候选；07–10 为边界备选，不能直接并入纯 G1。3 条白天、7 条黄昏。', '',f'[播放全部 10 条（{total} 秒，编号与目标说明）]({montage_path.as_posix()})','',f'![候选总览]({(OUT/"overview.jpg").as_posix()})','',
    '优先看 **01、02、03** 的同向跟随，再看 **04、06** 的迎面行走。若想看白天大目标，可看 **08**，但其相机转向更明显。05 用于较小、较暗的目标条件。', '',
    '## 口径与实际限制','',
    '- 这是按“目标周围干扰较少、连续可见、相机没有剧烈大转向”筛出的街道步行候选。自然步行仍含相机平移、头部起伏、视差和他人活动，不等同于相机静止或纯行人运动。',
    '- 原审查中 low 运动档要求 1 秒最大旋转不超过 10°，且角速度 P95 不超过 25°/s；本批没有片段同时满足这两个严格条件。01–06 属于原先的 moderate 档（1 秒旋转小于 25°且角速度 P95 小于 60°/s）。因此这里写“G1 步行候选”，并未把原来的 low 定义偷偷放宽后统计成严格 G1。',
    '- 05、06 有低照度及小目标问题；07–09 有较明显相机抖动或转向，10 的整体人流较多。这些因素已保留在清单中。最终三分类还需结合你想控制的难度因素确认。',
    '- **01 和 03 是同一白上衣人物的不同时间段**，用于观察距离/尺度变化，不能作为两个独立身份。前五条都出自 192702；10 条片段不等于 10 次独立采集，也不保证 10 个不同身份。后续统计按录制和连续人物事件分组，不能把所有窗口当独立样本。',
    '- 所有片段时间互不重叠。本批是素材筛选，尚未产生 SAM2 标注、人工 GT 或新效果结论。', '',
    '## 逐条清单','',
    '|编号|录制 / 原视频秒数|时长|建议目标|分组|', '|---|---|---:|---|---|']
    for r in rows:lines.append(f"|[{r['candidate']:02d}]({Path(r['clip']).as_posix()})|{r['id']} / {r['start_s']}–{r['end_s']}|{r['duration_s']}秒|{r['target']}|{'优先候选' if r['candidate']<=6 else '边界备选'}|")
    lines+=['','## 选择理由和剩余问题','']
    for r in rows:lines += [f"### {r['candidate']:02d} · {r['id']} · {r['start_s']}–{r['end_s']} 秒",'',r['visual_observation']+' '+r['remaining_factors'],'',f"[单段视频]({Path(r['clip']).as_posix()}) · [代表帧]({Path(r['keyframe']).as_posix()})",'']
    lines+=['## 相机参考运动（完整所选时间段）','','仅用于离线描述与筛选，来自 MPS closed-loop 姿态；不是在线可用性证明，也未送入任何新跟踪实验。1 秒旋转只使用窗口内部成对姿态。位移是设备端点间直线距离，不是完整路程，也不声称已分离视差。','', '|编号|1 秒最大旋转 °|角速度 P95 °/s|有效姿态覆盖|设备端点位移 m|','|---|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'];lines.append(f"|{r['candidate']:02d}|{m['max_1s_rotation_deg']:.1f}|{m['omega_p95_deg_s']:.1f}|{m['pose_coverage']:.0%}|{m['device_endpoint_displacement_m']:.2f}|")
    lines+=['','## 筛选与复核记录','','从 9 段含街道/户外路段的录制做 5 秒滑窗、1 秒步长运动预筛，生成 48 个分散候选的联系表；结合其中已查看的片段及此前场景审查，细查选中窗口。最终片段按每秒 5 帧查看，共 265 个采样时刻；这是密集抽样，不是逐帧人工 GT。','',
    '进一步查看原分辨率局部后，排除了 192702 的 146–152 秒：白上衣者主要站着看手机，前方还有弯腰者，不符合本次持续走路的筛选目标。245–250 秒有近身离场/目标持续性问题，也未采用。01 的初始 184–194 秒和10的173–181秒都裁去了后续转头较大的部分。','',
    '预览保留原鱼眼几何，仅缩小到 960×720，10 fps，无音频。53秒合集额外增加顶部文字栏，不覆盖图像。后续实验应读取原始2560×1920视频；selected_clips.json 与 frame_maps/*.json 给出原视频帧号及 VRS 时间戳，不能直接将这些帧号套到其他去畸变/重导出视频上。','',
    f"[片段清单和原始路径]({(OUT/'selected_clips.json').as_posix()}) · [视频完整解码检查]({(OUT/'verification.json').as_posix()})",'']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    (OUT/'START_HERE.md').write_text(f'# 街道步行候选\n\n[53秒十条合集]({montage_path.as_posix()})\n\n[逐条清单、目标与限制]({(REPORT/"RESULTS.md").as_posix()})\n\n01–06 优先候选；07–10 边界备选。01和03为同一人物的不同时段。\n',encoding='utf8')
    print(summary,flush=True)
if __name__=='__main__':main()
