from render_cases import *
from PIL import Image,ImageDraw,ImageFont
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
FF='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
NOTES={
'A':('外观门槛造成短暂断轨',['位置只差 18.5 px；深度 6.33 vs 6.30 m。','外观相似度 0.429，低于门槛 0.45。','T1 → T2 → T1，单帧外观变化触发新 ID。','只移除该帧外观门槛，可保留 T1。']),
'B':('运动门槛与旧轨迹竞争；原 mask 存疑',['原轨迹差 69.5 px，但超过运动门槛。','失联 0.6 秒的另一轨迹差 90.4 px，却通过。','两条轨迹均无可用三维状态。','原 N07 mask 混入他人，身份报警须谨慎。']),
'C':('外观分数让新目标抢走已有 ID',['T2 对原目标只差 11.5 px，相似度 0.518。','对新目标差 56.8 px，相似度却达 0.733。','总成本 0.573 vs 0.544，选中新目标。','已判为不可靠，但仍输出该 ID；仅冻结记忆。']),
'D':('失联旧轨迹重新参与，竞争掉当前轨迹',['T8 已失联 0.6 秒；T10 上一帧仍在。','T8 在位置和外观分数上均更占优。','只在本帧排除失联 >0.3 秒轨迹，会选 T10。','这是局部决策诊断，不是已验证的改进规则。']),
'E':('三维更新停滞，切换二维后断轨',['N04 连续第22—25帧跳过三维状态更新。','第26帧三维相对深度不确定度约 39%。','退回二维后位置差 371 px，T6 → T13。','当帧深度 0.881 m；未见明显测量突跳。']),
'F':('位置范围过宽，深度冲突没有阻止串人',['T13 原跟随粉衣人，现接收另一位白衣人。','错轨迹位置差 540 px，仍通过运动门槛。','深度预计 0.50 m，候选测量 1.92 m。','深度差项最多加 0.06，冲突仍不足以拒绝。']),
'G':('夜间 mask 异常，叠加失联轨迹竞争',['N03 mask 形状、范围在第27帧明显改变。','旧 N04 轨迹失联 0.8 秒后接收该候选。','原轨迹位置差 229 px，总成本超过 0.72。','需要先核查候选质量，不能只归因于运动。']),
'H':('候选断供超过轨迹保留时间',['N06 最后一次候选在第24帧，再现于第37帧。','中间12帧无该候选；两次间隔约1.3秒。','轨迹只保留1秒，原 T3 已被删除。','因此新建 T8：本帧没有进入旧身份匹配。'])}
def paint(im,text,xy,size=23,color=(232,238,246)):
 d=ImageDraw.Draw(im);d.text(xy,text,font=ImageFont.truetype(FONT,size),fill=color)
def winpath(p):return 'E:/'+str(p.relative_to(Path('/mnt/e')))
def encoder(path):
 return subprocess.Popen([FF,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x900','-r','10','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',winpath(path)],stdin=subprocess.PIPE)
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');cases=read(OUT/'selected_cases.json');diagnostics=read(OUT/'depth_and_mask_diagnostics.json');manifest=[]
 for case in cases:
  code=case['code'];cid=case['clip'];f=case['frame'];evid=case['catalog_event'];title,notes=NOTES[code];entries={e['frame']:e for e in inv['entries'] if e['clip']==cid};trace=read(OUT/'traces'/f'{cid}.json');start=max(0,min(case['previous_source_last_frame']-2,f-6));end=min(len(entries)-1,f+4);cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));frames={}
  for t in range(end+1):
   ok,im=cap.read();assert ok
   if t>=start:frames[t]=im
  cap.release();dest=PRES/f'{code}_clip{cid}_frame{f:02d}.mp4';enc=encoder(dest);written=0
  for t in range(start,end+1):
   src=frames[t];panel=np.full((900,1600,3),19,np.uint8);scene=cv2.resize(src,(1000,750));e=entries[t];rr={r['uid']:r for r in trace[t]['rows']};sid=case['source_id'];old=case['previous_source'];target=next((o for o in e['targets'] if ref[o['target_key']]['source_id']==sid),None)
   for o in e['targets']:
    uid=o['target_key'];s=ref[uid]['source_id']
    if s not in {sid,old}:continue
    col=(255,220,50) if s==sid else (40,210,255);m=cv2.resize(decode(o['rle']),(1000,750),interpolation=cv2.INTER_NEAREST).astype(bool);scene[m]=(scene[m]*.72+np.array(col)*.28).astype(np.uint8);b=(np.array(o['box'])*1000/2560).astype(int);cv2.rectangle(scene,tuple(b[:2]),tuple(b[2:]),col,2);txt(scene,f'{s} / T{rr[uid]["track_id"]}',max(0,b[0]),max(17,b[1]-6),.65,col)
   # Overlay the actual pair's pre-assignment position, never a future estimate.
   if target:
    uv=(np.array(target['box'][:2])+target['box'][2:])/2;center=tuple((uv*1000/2560).astype(int));cv2.circle(scene,center,6,(255,255,255),-1)
    for tid,col,label in [(evid['previous_person_track'],(40,210,255),'previous'),(evid['assigned']['track_id'],(230,50,235),'assigned')]:
     p=trace[t]['priors'].get(str(tid))
     if p is None or p['uv'] is None:continue
     xy=tuple(np.clip(np.array(p['uv'])*1000/2560,[-10000,-10000],[10000,10000]).astype(int));cv2.line(scene,xy,center,col,2);cv2.drawMarker(scene,xy,col,cv2.MARKER_CROSS,24,3);txt(scene,f'T{tid} {label}',max(0,min(850,xy[0]+8)),max(20,min(730,xy[1]-8)),.5,col)
   panel[70:820,:1000]=scene;pil=Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB));paint(pil,f'{code}  片段 {cid}  |  {title}',(20,13),29);paint(pil,f'当前帧 {t}  /  问题帧 {f}'+('  ← 停顿检查' if t==f else ''),(1020,75),26,(255,185,90));paint(pil,'问题帧的决策证据（固定说明）',(1020,124),24)
   for j,n in enumerate(notes):paint(pil,n,(1020,170+j*45),21)
   paint(pil,'白点：当前候选中心；黄十字：原身份预测',(20,834),22);paint(pil,'紫十字：本次输出身份的预测；连线表示位置偏差',(20,867),22);paint(pil,'T 为关联输出；N 为原标注核查编号',(1020,375),21);paint(pil,'下方为当前目标的 RGB 与原始 SAM mask',(1020,415),21)
   panel=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
   if target:
    mask=decode(target['rle']).astype(bool);masked=src.copy();masked[~mask]=127
    panel[460:770,1020:1300]=fit(crop(src,target['box']),280,310);panel[460:770,1310:1590]=fit(crop(masked,target['box']),280,310)
   pil=Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB));paint(pil,'慢放约 1/3 速；问题帧额外停留 2 秒',(1020,803),21);paint(pil,'只回放原结果，没有修改标签或算法',(1020,842),21);panel=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
   if t==f:cv2.imwrite(str(PRES/f'{code}_event.jpg'),panel)
   for _ in range(23 if t==f else 3):enc.stdin.write(panel.tobytes());written+=1
  enc.stdin.close();assert enc.wait()==0
  manifest.append(dict(code=code,clip=cid,event_frame=f,source_start=start,source_end=end,video=dest.name,frames=written,seconds=written/10));print('RENDERED',code,written,flush=True)
 concat=PRES/'concat.txt';concat.write_text(''.join("file '"+winpath(PRES/m['video'])+"'\n" for m in manifest),encoding='utf8');allpath=PRES/'all8_failure_review.mp4';subprocess.run([FF,'-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',winpath(concat),'-c','copy','-movflags','+faststart',winpath(allpath)],check=True)
 checks=[]
 for path,expected in [(PRES/m['video'],m['frames']) for m in manifest]+[(allpath,sum(m['frames'] for m in manifest))]:
  cap=cv2.VideoCapture(str(path));count=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape==(900,1600,3);count+=1
  cap.release();assert count==expected;checks.append(dict(video=path.name,decoded_frames=count))
 write(OUT/'video_manifest.json',manifest);write(OUT/'video_verification.json',dict(passed=True,checks=checks))
 print('ALL DONE',sum(m['seconds'] for m in manifest),flush=True)
if __name__=='__main__':main()
