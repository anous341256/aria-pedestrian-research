from partial_common import *
import cv2,subprocess
from PIL import Image,ImageDraw,ImageFont
FF='/mnt/e/下载/ffmpeg-2026-06-15-git-44d082edc8-full_build/bin/ffmpeg.exe'
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
def wp(p):return 'E:/'+str(p.relative_to(Path('/mnt/e')))
def txt(a,s,xy,col=(240,240,240),size=.5):cv2.putText(a,s,xy,cv2.FONT_HERSHEY_SIMPLEX,size,(0,0,0),3,cv2.LINE_AA);cv2.putText(a,s,xy,cv2.FONT_HERSHEY_SIMPLEX,size,col,1,cv2.LINE_AA)
def zh(a,s,xy,size=22):
 im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));ImageDraw.Draw(im).text(xy,s,font=ImageFont.truetype(FONT,size),fill=(235,238,245));return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
def color(t):return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[t*47%180,200,255]]]),cv2.COLOR_HSV2BGR)[0,0])
def encoder(path,w=1280,h=640,fps=10):return subprocess.Popen([FF,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',wp(path)],stdin=subprocess.PIPE)
def finish(p):p.stdin.close();assert p.wait()==0
def render(mode):
 dest=(TRIM if mode=='trim' else OUT);pres=dest/'presentation';pres.mkdir(exist_ok=True);inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');montage=pres/f'all10_{mode}_comparison.mp4';enc=encoder(montage);videos=[]
 for cl in inv['clips']:
  cid=cl['clip'];entries=[e for e in inv['entries'] if e['clip']==cid];old={r['uid']:r for r in read(BASE/'tracking/natural'/ARM/f'{cid}.json')['rows']};new=old if mode=='trim' else {r['uid']:r for r in read(OUT/'tracking'/f'{cid}.json')['rows']};keep={};reject={}
  if mode=='trim':
   data=read(TRIM/'clips'/f'{cid}.json');keep={u:s['segment_id'].rsplit('_',1)[-1] for s in data['methods']['quality_trim']['segments'] for u in s['uids']};reject={a['uid']:a for a in data['audit']}
  path=pres/f'{cid}_{mode}.mp4';ce=encoder(path);cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'))
  for e in entries:
   ok,im=cap.read();assert ok;out=np.full((640,1280,3),20,np.uint8);out=zh(out,f'片段 {cid}  帧 {e["frame"]}   |   '+('轨迹裁剪版：原跟踪算法完全不变' if mode=='trim' else '局部外观版：同一批 SAM2 候选与相机/深度输入'),(15,8));out=zh(out,'原版：整体外观＋几何关联',(15,51));out=zh(out,'保留的连续轨迹（绿色）' if mode=='trim' else '新版：整体外观＋局部对应＋几何关联',(655,51))
   for j,lookup in enumerate([old,new]):
    panel=cv2.resize(im,(640,480))
    for t in e['targets']:
     u=t['target_key'];r=lookup[u];b=(np.array(t['box'])/4).astype(int);col=color(r['track_id']);mask=cv2.resize(decode(t['rle']),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool);label=f'T{r["track_id"]}'
     if mode=='trim' and j==1:
      col=(70,225,70) if u in keep else ((60,60,235) if reject[u]['hard_exclusion'] else (145,145,145));label=('S'+keep[u] if u in keep else 'CUT')+f' / T{r["track_id"]}'
     if not (mode=='trim' and j==1 and u not in keep):panel[mask]=(panel[mask]*.75+np.array(col)*.25).astype(np.uint8)
     cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),col,2);txt(panel,label,(max(0,b[0]),max(16,b[1]-3)),col)
     if mode!='trim' and r.get('local_weight',0)>0 and j==1:txt(panel,f'G{r["global_appearance"]:.2f} L{r["local_appearance"]:.2f}',(max(0,min(490,b[0])),min(470,b[3]+14)),col,.37)
    out[95:575,j*640:(j+1)*640]=panel
   note='绿：保留；红：明确质量报警；灰：边界/长度等原因剔除。原 mask 未修改。' if mode=='trim' else 'T：本方法独立分配的 ID，左右数字不要求相同；G：整体分数，L：局部匹配分数。'
   out=zh(out,note,(15,590),20);ce.stdin.write(out.tobytes());enc.stdin.write(out.tobytes())
   if e['frame']==cl['frames']//2:cv2.imwrite(str(pres/f'{cid}_midframe.jpg'),out)
  cap.release();finish(ce);videos.append(dict(path=str(path.relative_to(ROOT)),frames=cl['frames']));print('VIDEO',mode,cid,flush=True)
 finish(enc);videos.append(dict(path=str(montage.relative_to(ROOT)),frames=530))
 for v in videos:
  cap=cv2.VideoCapture(str(ROOT/v['path']));n=0
  while True:
   ok,im=cap.read()
   if not ok:break
   assert im.shape==(640,1280,3);n+=1
  cap.release();assert n==v['frames'];v['decoded']=n
 write(dest/'video_verification.json',dict(passed=True,videos=videos));print('VERIFIED',mode)
if __name__=='__main__':render(sys.argv[1])
