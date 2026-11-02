"""SAM masks for survey measurement, independent short-frame box prompts."""
from pathlib import Path
import os
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface';os.environ['HF_HUB_OFFLINE']='1';os.environ['OMP_NUM_THREADS']='2'
import json,time,cv2,numpy as np,torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from cvat_sdk.masks import encode_mask
import gzip
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/g1_population_survey_20261006'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def quality(mask,box):
 y,x=np.where(mask)
 if not len(x):return dict(valid=False,box=None,guide_iou=0.,outside=1.,pixels=0)
 b=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1];a=np.array(b);g=np.array(box)
 inter=np.prod(np.maximum(0,np.minimum(a[2:],g[2:])-np.maximum(a[:2],g[:2])));iou=inter/max(1,np.prod(a[2:]-a[:2])+np.prod(g[2:]-g[:2])-inter)
 pad=max(12,.1*(g[3]-g[1]));x1,y1,x2,y2=np.rint(g+[-pad,-pad,pad,pad]).astype(int);outside=1-mask[max(0,y1):min(1920,y2),max(0,x1):min(2560,x2)].sum()/len(x)
 return dict(valid=bool(iou>=.45 and outside<=.1),box=b,guide_iou=float(iou),outside=float(outside),pixels=len(x))
def main():
 torch.set_num_threads(2);predictor=SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-large',device='cuda');start=time.monotonic()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for sample in read(OUT/'sample_locked.json'):
   dest=OUT/'samples'/sample['sample_id']
   if (dest/'segmentation.json').exists():continue
   while not (dest/'proposals.json').exists():time.sleep(1)
   tracks=[t for t in read(dest/'proposals.json') if t['automatic_eligible']];results={t['key']:[] for t in tracks};masksout=[]
   for f in range(6):
    if not tracks:break
    im=cv2.imread(str(dest/f'f{f:02d}.jpg'));predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    for t in tracks:
     guide=next(r['box'] for r in t['rows'] if r['frame']==f)
     masks,scores,_=predictor.predict(box=np.array(guide),multimask_output=False)
     mask=masks[0].astype(bool);q=quality(mask,guide);score=float(scores[0])
     results[t['key']].append(dict(frame=f,score=score,quality=q))
     masksout.append(dict(key=t['key'],frame=f,rle=encode_mask(mask)))
     if f in [0,5]:
      # Contextual mask crop for measurement QC.
      x1,y1,x2,y2=np.rint(guide).astype(int);pad=max(25,(y2-y1)//6);x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(2560,x2+pad);y2=min(1920,y2+pad)
      vis=im[y1:y2,x1:x2].copy();m=mask[y1:y2,x1:x2];vis[m]=(vis[m]*.6+np.array([80,220,80])*.4).astype(np.uint8)
      cv2.imwrite(str(dest/f'mask_{t["track"]:03d}_{f:02d}.jpg'),vis,[cv2.IMWRITE_JPEG_QUALITY,94])
   with gzip.open(dest/'survey_masks.jsonl.gz','wt',encoding='utf8') as h:
    for r in masksout:h.write(json.dumps(r)+'\n')
   write(dest/'segmentation.json',results);print(sample['sample_id'],'people',len(tracks),'elapsed',round(time.monotonic()-start,1),flush=True)
 write(OUT/'segmentation_complete.json',dict(seconds=time.monotonic()-start,model='facebook/sam2.1-hiera-large',frames_per_target=6))
if __name__=='__main__':main()
