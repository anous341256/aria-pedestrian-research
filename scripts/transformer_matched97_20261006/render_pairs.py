"""Four pre-score fixed old-test cases, old/new paired for each of six variants."""
from shared import *
import cv2,subprocess,argparse,importlib.util
spec=importlib.util.spec_from_file_location('drawing_reference',ROOT/'scripts/transformer_expanded_v2_20261005/render.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def prepare():
 mm=read(BASELINE/'data/samples.json');recs=sorted({m['recording'] for m in mm});keys=[]
 for j in np.linspace(0,len(recs)-1,4,dtype=int):
  rows=sorted([m for m in mm if m['recording']==recs[j]],key=lambda m:(m['clip'],m['cutoff'],m['id']));keys.append(rows[len(rows)//2]['key'])
 path=VERSION/'visual_cases.json'
 if path.exists():assert read(path)['keys']==keys
 else:write(path,dict(keys=keys,rule='Four evenly spaced recording names, temporally middle existing test window in each. Selected before new forecast scores.',display_seed=17))
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
 if args.prepare:prepare();return
 dest=VERSION/'presentation';dest.mkdir(exist_ok=True);mm=read(BASELINE/'data/samples.json');data=dict(np.load(BASELINE/'data/dataset.npz'));lookup={m['key']:i for i,m in enumerate(mm)};keys=read(VERSION/'visual_cases.json')['keys'];pred={}
 for arm in ['old','new']:
  for fold in read(VERSION/arm/'folds.json'):
   for v in VARIANTS:
    z=np.load(VERSION/arm/f'fold_{fold["fold"]}/seed_17/{v}/predictions_reviewed.npz')
    for j,i in enumerate(z['indices']):
     if mm[i]['key'] in keys:pred[arm,int(i),v]=(z['pred'][j],metrics(z['pred'][j],data['xy'][i,P:],z['depth'][j])['ADE15'])
 path=dest/'FIXED4_SIX_MODELS_OLD_NEW.mp4';assert not path.exists();log=path.with_suffix('.ffmpeg.log').open('wb');W,H=1280,650
 enc=subprocess.Popen([r.FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','10','-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=log)
 for key in keys:
  i=lookup[key];m=mm[i];cid=m['clip'];source='sam2_g1_v6_v8_20261001' if int(cid)<=10 else 'sam2_g1_expansion_20261005';labels='sam2_near_people_20261002' if int(cid)<=10 else source
  cap=cv2.VideoCapture(str(ROOT/f'outputs/{source}/{cid}/input/rgb.mp4'));ff=[]
  for frame in range(m['cutoff']-5,m['cutoff']+16):
   cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok;ff.append(cv2.resize(im,(640,480)))
  cap.release();maskrows={(a['id'],a['frame']):a for a in gz(ROOT/f'outputs/{labels}/{cid}/final/masks_rle.jsonl.gz')}
  for variant in VARIANTS:
   for j,im in enumerate(ff):
    canvas=np.full((H,W,3),23,np.uint8);texts=[(15,8,f'{NAMES[variant]} | 固定旧测试案例 {key} | 种子17',25)]
    for column,arm in enumerate(['old','new']):
     panel=im.copy();mask=maskrows.get((m['id'],m['cutoff']-5+j))
     if mask:
      mask=r.decode_mask(mask['rle']);panel[mask]=panel[mask]*.84+.16*np.array([255,220,80])
     pp,ade=pred[arm,i,variant];r.point(panel,data['xy'][i,j],(255,255,255),6)
     if j>=6:
      r.point(panel,pp[j-6],(0,170,255),9);a=np.clip(data['xy'][i,j]*.25,-5000,5000).astype(int);b=np.clip(pp[j-6]*.25,-5000,5000).astype(int);cv2.line(panel,tuple(a),tuple(b),(0,170,255),2,cv2.LINE_AA)
     canvas[100:580,column*640:(column+1)*640]=panel;texts.append((column*640+12,60,f'{"旧新增批次" if arm=="old" else "新G1批次"} | ADE {ade:.1f} px',26))
    stage='观察过去0.5秒' if j<6 else f'预测未来 {(j-5)/10:.1f}秒'
    texts.append((15,600,stage+' · 白圈参考中心，橙圈预测 · 两侧同一测试窗口及训练数量',22));canvas=r.label(canvas,texts)
    enc.stdin.write(canvas.tobytes());enc.stdin.write(canvas.tobytes())
    if j==20:cv2.imwrite(str(dest/f'{key}_{variant[0]}_pair.jpg'),canvas)
 enc.stdin.close();assert enc.wait(timeout=60)==0;log.close();cap=cv2.VideoCapture(str(path));n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==(H,W,3);n+=1
 cap.release();assert n==4*6*42;write(dest/'paired_video_audit.json',dict(passed=True,frames=n,seconds=n/10,keys=keys,display_seed=17));print('Paired video verified',n,flush=True)
if __name__=='__main__':main()
