"""Find unused low-complexity source intervals across all recordings."""
from inventory import *
import cv2,csv
from PIL import Image,ImageDraw,ImageFont
from concurrent.futures import ThreadPoolExecutor
def main():
 used=[]
 for v in ['g1_street_candidates_20261001','sam2_g1_expansion_20261005','sam2_g1_expansion_v2_20261005','sam2_g1_matched97_20261006']:
  used+=read(ROOT/'outputs'/v/'selected_clips.json')
 used.append(dict(id='201129',start_s=10,end_s=34))
 thresholds=read(OUT/'thresholds_locked.json')['thresholds'];allrows=[];chosen=[]
 for sid,rec in INV.items():
  if not rec.get('mps_source') or not Path(rec['mps_source']).exists():continue
  mm=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
  times=np.array([int(x['vrs_device_time_ns']) for x in mm],np.int64);rel=np.arange(len(times))/10
  ts,df=pose(sid);ix=np.searchsorted(ts,times).clip(1,len(ts)-1);ix=np.where(abs(ts[ix]-times)<abs(ts[ix-1]-times),ix,ix-1)
  q=df.iloc[ix][['qx_world_device','qy_world_device','qz_world_device','qw_world_device']].to_numpy();rr=Rotation.from_quat(q).as_matrix();pos=df.iloc[ix][['tx_world_device','ty_world_device','tz_world_device']].to_numpy()
  valid=(abs(ts[ix]-times)<30e6)&(df.quality_score.to_numpy()[ix]>=.5);steps=angle(rr[:-1],rr[1:]);sec=angle(rr[:-10],rr[10:]);rows=[]
  for st in range(1,len(times)//10-8,2):
   en=st+8;lo=st*10;hi=en*10
   if any(x['id']==sid and max(st-.5,x['start_s'])<min(en+.5,x['end_s']) for x in used):continue
   if not valid[lo:hi].all():continue
   mx=float(max(sec[lo:hi-10]));om=float(np.percentile(steps[lo:hi-1]*10,95))
   good=[j for j in range(lo,hi-20,5) if max(sec[j:j+11])<=20 and np.percentile(steps[j:j+20]*10,95)<=50]
   if not good:continue
   a=float(np.median([sum(steps[i:i+5]) for i in range(lo,hi-5,5)]));b=float(np.median(angle(rr[lo:hi-5:5],rr[lo+5:hi:5])));c=float(np.linalg.norm(np.diff(pos[lo:hi],axis=0),axis=1).mean()*10)
   row=dict(id=sid,start_s=st,end_s=en,A=a,B=b,C=c,abc=[int(x>thresholds[k]) for k,x in zip('ABC',[a,b,c])],max_1s_rotation=mx,omega_p95=om,g1_window_starts=[x/10 for x in good],recording=rec['recording']);rows.append(row)
  allrows+=rows;keep=[]
  # Spread time and camera states, without using any forecast outcomes.
  for group in sorted({tuple(x['abc']) for x in rows}):
   cand=[x for x in rows if tuple(x['abc'])==group]
   for x in sorted(cand,key=lambda x:(-len(x['g1_window_starts']),x['max_1s_rotation']+.15*x['omega_p95'])):
    if any(max(x['start_s'],z['start_s'])<min(x['end_s'],z['end_s']) for z in keep):continue
    keep.append(x);break
  for x in sorted(rows,key=lambda x:(-len(x['g1_window_starts']),x['max_1s_rotation']+.15*x['omega_p95'])):
   if len(keep)>=3:break
   if any(max(x['start_s']-5,z['start_s'])<min(x['end_s']+5,z['end_s']) for z in keep):continue
   keep.append(x)
  chosen+=keep[:4];print(sid,len(rows),len(keep[:4]),flush=True)
 dest=OUT/'screen';dest.mkdir(exist_ok=True)
 for i,x in enumerate(chosen):x['screen_id']=i+1
 def render(row):
  cap=cv2.VideoCapture(INV[row['id']]['video']);ims=[]
  for f in [row['start_s']*10,(row['start_s']+4)*10,row['end_s']*10-1]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;ims.append(cv2.cvtColor(cv2.resize(bgr,(400,300)),cv2.COLOR_BGR2RGB))
  cap.release();canvas=Image.new('RGB',(1200,335),(242,242,242));draw=ImageDraw.Draw(canvas)
  for j,im in enumerate(ims):canvas.paste(Image.fromarray(im),(400*j,35))
  draw.text((5,3),f"#{row['screen_id']:02d} {row['id']} {row['start_s']}-{row['end_s']} sec  ABC={row['abc']}  turn={row['max_1s_rotation']:.1f}",font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22),fill='black')
  canvas.save(dest/f"candidate_{row['screen_id']:02d}.jpg",quality=90);return canvas
 with ThreadPoolExecutor(max_workers=3) as ex:ims=list(ex.map(render,chosen))
 for k in range(0,len(ims),5):
  c=Image.new('RGB',(1200,335*len(ims[k:k+5])))
  for j,im in enumerate(ims[k:k+5]):c.paste(im,(0,j*335))
  c.save(dest/f'page_{k//5+1:02d}.jpg',quality=90)
 write(OUT/'unused_screen_candidates.json',chosen);write(OUT/'unused_all_candidates.json',allrows);write(OUT/'used_intervals.json',used)
 print('screened',len(allrows),'selected',len(chosen),flush=True)
if __name__=='__main__':main()
