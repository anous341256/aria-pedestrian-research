"""Render aligned model outputs and coverage, with no accuracy claim."""
import csv
from PIL import Image,ImageDraw,ImageFont
from shared import *

RUN=OUT/'sam3_seeded_hf'
DEST=OUT/'presentation'
LABELS={
    'raw':'原始画面',
    'sam2_native':'SAM 2.1 原始传播',
    'sam3_seeded_hf':'SAM 3 同初始 mask',
    'sam2_negative_memory':'SAM 2 + 后续负提示/记忆',
    'sam2_motion_no_camera':'SAM 2 + 运动规则（无姿态）',
    'sam2_motion_camera':'SAM 2 + 相机补偿（20%）',
    'sam2_motion_weight10_camera':'SAM 2 + 相机补偿（10%）',
    'sam2_reliability_camera':'SAM 2 + 可靠性先验',
}
FONT_PATH='/mnt/c/Windows/Fonts/msyh.ttc'
FONTS={size:ImageFont.truetype(FONT_PATH,size) for size in (16,18,20,24,28)}


def tile(frame,masks,name,frame_idx,width):
    im=panel(frame,masks,'',frame_idx,width)
    pil=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(pil)
    draw.text((10,5),LABELS[name],font=FONTS[24 if width>=900 else 18],fill=(245,245,245))
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def footer(width,series=False):
    im=Image.new('RGB',(width,72),(20,22,27))
    d=ImageDraw.Draw(im)
    d.text((14,8),'A 黑衣金色背图   B 灰格纹衬衫   C 白色层叠裙   D 远处深色长装   E 远处浅衣短裤',font=FONTS[18],fill=(230,230,230))
    note='各身份颜色跨版本固定；白斜线表示多个 ID 的 mask 重叠。非空 mask 不等于正确身份。'
    if series:note='相机版本含离线 MPS 参考旋转；负提示版含额外审阅信息，输入条件不同。白斜线 = ID 重叠。'
    d.text((14,40),note,font=FONTS[18],fill=(175,187,204))
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)


def presence_runs(rows,identity):
    runs=[];begin=None
    for frame_idx in range(START,STOP+1):
        present=frame_idx<STOP and mask_area(rows[frame_idx,identity])>0
        if present and begin is None:begin=frame_idx
        if not present and begin is not None:runs.append([begin,frame_idx-1]);begin=None
    return runs


def main():
    runtime=read_json(RUN/'runtime.json')
    assert runtime['status']=='completed' and runtime['frames']==STOP-START
    DEST.mkdir(exist_ok=False)
    methods={}
    for name,folder,field,value,info in BASELINES:
        methods[name]=archive(ROOT/'outputs'/folder/'masks_rle.jsonl.gz',field,value)
    methods['sam3_seeded_hf']=archive(RUN/'masks_rle.jsonl.gz')
    expected={(f,i) for f in range(START,STOP) for i in IDS}
    seeds=read_json(OUT/'initial_masks.json')
    summaries={}
    for name,rows in methods.items():
        assert set(rows)==expected,name
        for identity in IDS:
            assert np.array_equal(decode(rows[START,identity]),decode(seeds[identity]))
        summaries[name]={i:dict(nonempty_frames=sum(mask_area(rows[f,i])>0 for f in range(START,STOP)),
                               total_frames=STOP-START,presence_intervals=presence_runs(rows,i)) for i in IDS}
    write_json(OUT/'comparison_summary.json',dict(metric='Nonempty-mask coverage only; no dense ground truth',methods=summaries))
    with (OUT/'coverage.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.writer(handle);writer.writerow(['method',*IDS,'total_nonempty','total_object_frames'])
        for name,info in summaries.items():
            values=[info[i]['nonempty_frames'] for i in IDS]
            writer.writerow([name,*values,sum(values),len(expected)])
    series=['raw','sam2_native','sam3_seeded_hf','sam2_negative_memory',
            'sam2_motion_no_camera','sam2_motion_camera','sam2_motion_weight10_camera','sam2_reliability_camera']
    duo_footer=footer(1920)
    all_footer=footer(2048,series=True)
    duo_enc,duo_log=encoder(DEST/'sam2_vs_sam3.mp4',1920,848,FPS,crf=18)
    all_enc,all_log=encoder(DEST/'all_variants.mp4',2048,952,FPS,crf=18)
    sam_enc,sam_log=encoder(DEST/'sam3_masks.mp4',1280,1088,FPS,crf=18)
    sam_footer=footer(1280)
    keyframes=set(read_json(OUT/'protocol.json')['predeclared_review_frames'])|set(range(START,STOP,10))
    contact_tiles=[];contact_indices=[]
    for local_idx,frame_idx in enumerate(range(START,STOP)):
        frame=cv2.imread(str(OUT/'frames'/f'{local_idx:05d}.png'))
        masks={name:{i:rows[frame_idx,i] for i in IDS} for name,rows in methods.items()}
        main_pair=np.hstack([tile(frame,masks[name],name,frame_idx,960) for name in ('sam2_native','sam3_seeded_hf')])
        main_pair=np.vstack([main_pair,duo_footer]);duo_enc.stdin.write(main_pair.tobytes())
        tiles=[tile(frame,masks.get(name,{}),name,frame_idx,512) for name in series]
        all_canvas=np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:]),all_footer])
        all_enc.stdin.write(all_canvas.tobytes())
        single=np.vstack([tile(frame,masks['sam3_seeded_hf'],'sam3_seeded_hf',frame_idx,1280),sam_footer])
        sam_enc.stdin.write(single.tobytes())
        if frame_idx in keyframes:
            cv2.imwrite(str(DEST/f'compare_{frame_idx:04d}.jpg'),main_pair,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(DEST/f'series_{frame_idx:04d}.jpg'),all_canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
        if frame_idx%10==0 or frame_idx==STOP-1:
            contact_tiles.append(tile(frame,masks['sam3_seeded_hf'],'sam3_seeded_hf',frame_idx,800))
            contact_indices.append(frame_idx)
        if local_idx%40==0:print('RENDERED',frame_idx,flush=True)
    for proc,log in [(duo_enc,duo_log),(all_enc,all_log),(sam_enc,sam_log)]:finish_encoder(proc,log)
    for page_idx,offset in enumerate(range(0,len(contact_tiles),6)):
        page=contact_tiles[offset:offset+6]
        while len(page)<6:page.append(np.zeros_like(page[0]))
        contact=np.vstack([np.hstack(page[j:j+2]) for j in (0,2,4)])
        cv2.imwrite(str(DEST/f'sam3_contact_{page_idx:02d}.jpg'),contact,[cv2.IMWRITE_JPEG_QUALITY,94])
    verified=[]
    for name in ('sam2_vs_sam3.mp4','all_variants.mp4','sam3_masks.mp4'):
        cap=cv2.VideoCapture(str(DEST/name));count=0
        size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps=cap.get(cv2.CAP_PROP_FPS)
        while cap.read()[0]:count+=1
        cap.release()
        assert count==STOP-START and abs(fps-FPS)<.001,(name,count,fps)
        verified.append(dict(file=name,decoded_frames=count,fps=fps,size=size))
    write_json(OUT/'verification.json',dict(archive_rows_per_method=len(expected),
               initial_masks_match=True,methods=list(methods),videos=verified,
               keyframes=sorted(keyframes),systematic_review_frames=contact_indices))
    print('VISUALS_COMPLETE',json.dumps(verified),flush=True)


if __name__=='__main__':main()
