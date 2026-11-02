"""Temporal mask QA: sampled identity strips plus every automatic anomaly.

Diagnostics are review aids, not calibrated accuracy. No labels are changed.
"""
from pathlib import Path
import sys,importlib.util,json,gzip,collections,time,argparse,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_g1_matched97_20261006'
spec=importlib.util.spec_from_file_location('old_render',ROOT/'scripts/sam2_incremental_v6_20261002/render.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
from common import read_json,write_json,digest
from reviewed_people import PEOPLE
CLIPS=list(PEOPLE)
FONT=ImageFont.truetype(v.FONT,16)
def crop(frame,guide,row,identity,f,status):
 b=np.array(guide,int);pad=max(18,int(.13*(b[3]-b[1])));x1,y1=max(0,b[0]-pad),max(0,b[1]-pad);x2,y2=min(2560,b[2]+pad),min(1920,b[3]+pad)
 im=frame[y1:y2,x1:x2].copy();c=v.color(int(identity[1:])-1)
 if row:
  m=v.decode(row['rle'])[y1:y2,x1:x2].astype(bool);im[m]=(.7*im[m]+.3*np.array(c)).astype(np.uint8)
  contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,c,2)
 tile=Image.new('RGB',(200,240),(235,235,235));cr=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));cr.thumbnail((196,196));tile.paste(cr,(0,0));d=ImageDraw.Draw(tile)
 d.text((0,200),f'{identity} f{f} '+('MASK' if row else 'GAP'),fill='black',font=FONT);d.text((0,218),status.replace('CENSUS_PROMPT_REPAIR','REPAIR').replace('V6_','')[:22],fill='black',font=FONT)
 return tile
def sheets(cells,dest,prefix,cols=8,rows=4):
 paths=[]
 for page in range(math.ceil(len(cells)/(cols*rows))):
  batch=cells[page*cols*rows:(page+1)*cols*rows];canvas=Image.new('RGB',(cols*200,math.ceil(len(batch)/cols)*240),(245,245,245))
  for j,im in enumerate(batch):canvas.paste(im,(j%cols*200,j//cols*240))
  path=dest/f'{prefix}_{page+1}.jpg';canvas.save(path,quality=93);paths.append(path.name)
 return paths
def run(cid,revision):
 folder=OUT/cid;res=folder if revision=='raw' else folder/('final' if revision=='delivery' else revision);dest=folder/f'qa_{revision}';dest.mkdir(exist_ok=True)
 if (dest/'audit.json').exists():return
 while not (res/'complete.json').exists():time.sleep(2)
 rows=v.load(res/'masks_rle.jsonl.gz');ds=v.load(res/'decisions.jsonl.gz');census=read_json(res/'census.json' if (res/'census.json').exists() else folder/'census.json');entries=census['segments']
 lookup={(r['id'],r['frame']):r for r in rows};logs={(r['id'],r['frame']):r for r in ds};byf=collections.defaultdict(list)
 assert len(lookup)==len(rows)
 for r in rows:byf[r['frame']].append(r)
 cap=cv2.VideoCapture(str(folder/'input/rgb.mp4'));frames=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  frames.append(im)
 cap.release();alerts=[];stats=[];allcells=[]
 for e in entries:
  last=None;samples=set([e['start'],e['end'],e['anchor_frame']]+list(range(e['start'],e['end']+1,5)));cells=[]
  for guide in e['rows']:
   f=guide['frame'];key=(e['id'],f);r=lookup.get(key);d=logs.get(key,{});reasons=[];status=d.get('status','NO_MASK')
   if not r:reasons.append('missing')
   elif 'UNCERTAIN' in status:reasons.append('uncertain')
   if r:
    a=np.array(r['box']);center=(a[:2]+a[2:])/2;area=r['quality']['pixels'];height=a[3]-a[1];q=r['quality']
    if q['guide_iou']<.5:reasons.append('guide_disagreement')
    if q['outside']>.1:reasons.append('mask_outside_guide')
    if last and f-last['frame']==1:
     ar=area/max(1,last['area']);jump=np.linalg.norm(center-last['center'])/max(80,height,last['height'])
     if ar>1.8 or ar<1/1.8:reasons.append('area_change')
     if jump>.45:reasons.append('center_jump')
    last=dict(frame=f,area=area,center=center,height=height)
    stats.append(dict(id=e['id'],frame=f,segment=e['segment'],area=area,box=r['box'],status=status,guide_iou=q['guide_iou']))
   else:last=None
   if reasons:alerts.append(dict(clip=cid,id=e['id'],segment=e['segment'],frame=f,reasons=reasons));samples.add(f)
  for f in sorted(samples):
   guide=next(r['box'] for r in e['rows'] if r['frame']==f);r=lookup.get((e['id'],f));d=logs.get((e['id'],f),{})
   cells.append(crop(frames[f],guide,r,e['id'],f,d.get('status','NO_MASK')))
  paths=sheets(cells,dest,e['segment']);allcells.extend(cells)
 conflicts=[]
 for f,rr in byf.items():
  masks=[cv2.resize(v.decode(r['rle']),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool) for r in rr]
  for i,a in enumerate(masks):
   for j in range(i):
    b=masks[j];inter=np.count_nonzero(a&b);ratio=inter/max(1,min(a.sum(),b.sum()))
    if ratio>.15 and inter>=10:conflicts.append(dict(frame=f,ids=[rr[j]['id'],rr[i]['id']],containment=float(ratio)))
 sheets(allcells,dest,'all_tracks',cols=8,rows=6)
 # Full contexts catch omitted nearby people and masks outside expected crops.
 context=[];cols={e['id']:v.color(int(e['id'][1:])-1) for e in entries}
 for f in sorted(set(range(0,len(frames),10))|{len(frames)-1}):
  im,_=v.overlay(frames[f],byf[f],cols,{},f);im=cv2.resize(im,(640,480));im=v.text(im,[(5,5,f'{cid} frame {f}')],22);context.append(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
 for page in range(math.ceil(len(context)/6)):
  cells=context[page*6:page*6+6];canvas=Image.new('RGB',(1280,math.ceil(len(cells)/2)*480))
  for j,im in enumerate(cells):canvas.paste(im,(j%2*640,j//2*480))
  canvas.save(dest/f'context_{page+1}.jpg',quality=94)
 result=dict(clip=cid,revision=revision,people=census['people'],segments=len(entries),requested_frames=sum(len(e['rows']) for e in entries),masks=len(rows),statuses=dict(collections.Counter(d['status'] for d in ds)),alerts=alerts,conflicts=conflicts,sampled_crops=len(allcells),negative_points=sum(len(d.get('negative_points',[])) for d in ds))
 write_json(dest/'frame_stats.json',stats);write_json(dest/'audit.json',result)
 print(cid,'QA',len(rows),'masks',len(alerts),'alerts',len(conflicts),'overlaps',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='+',default=CLIPS);p.add_argument('--revision',default='raw');a=p.parse_args()
 for cid in a.clips:run(cid,a.revision)
