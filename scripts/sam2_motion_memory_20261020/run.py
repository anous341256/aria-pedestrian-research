"""Four matched near-person offline annotation arms; pretrained weights frozen."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['HF_HOME']='/opt/cvat-sam2-cache/huggingface'
os.environ['HF_HUB_OFFLINE']='1'
import sys,time,argparse,importlib.util,collections,copy,json,gzip
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/sam2_motion_memory_20261020'
spec=importlib.util.spec_from_file_location('near',ROOT/'scripts/sam2_near_people_20261002/run.py')
near=importlib.util.module_from_spec(spec);spec.loader.exec_module(near)
from attention import MotionMemory,metadata
from selection import read,write,digest
ARMS=['visual','external_box','camera','camera_person']
W,H=2560,1920
torch.set_num_threads(2)

def image_candidate(model,pp,box,negatives=()):
 coords=np.array(list(np.array(box).reshape(2,2))+[n['xy'] for n in negatives],np.float32)
 coords*=np.array([model.image_size/W,model.image_size/H],np.float32)
 points=dict(point_coords=torch.tensor(coords[None],device='cuda'),
             point_labels=torch.tensor([[2,3]+[0]*len(negatives)],device='cuda'))
 current,sam,_,_=model._track_step(frame_idx=0,is_init_cond_frame=True,
    current_vision_feats=pp.vision_feats,current_vision_pos_embeds=pp.vision_pos_embeds,
    feat_sizes=pp.feat_sizes,point_inputs=points,mask_inputs=None,
    output_dict=dict(cond_frame_outputs={},non_cond_frame_outputs={}),
    num_frames=1,track_in_reverse=False,prev_sam_mask_logits=None)
 _,_,ious,low,high,ptr,presence=sam
 mask=(torch.nn.functional.interpolate(low,size=(H,W),mode='bilinear',align_corners=False,antialias=True)[0,0]>0).cpu().numpy()
 return mask,float(ious.max()),float(presence.sigmoid().item())

def evaluate(c,state,guide):
 q=near.quality(c['mask'],guide)
 app=near.appearance_similarity(c['descriptor'],state['bank'])
 score=(.45*app+.25*np.clip(c['iou_score'],0,1)+.10*c['presence'])/.8
 valid=q['valid'] and c['pixels']>=200 and c['iou_score']>=.4 and c['presence']>=.35 and app>=.52 and score>=.66
 return dict(valid=bool(valid),score=float(score),appearance=float(app),quality=q,
             write=bool(valid and score>=.76 and app>=.62))

def add_meta(state,step,mask,box,R,time,reliability):
 for key in ['cond_frame_outputs','non_cond_frame_outputs']:
  if step in state['memory'][key]:
   state['memory'][key][step]['motion_meta']=metadata(mask,box,R,time,reliability)
 state['identity_reliability']=float(reliability)

def load_frames(c,tracker):
 inp=ROOT/c['input'];cal=np.load(inp/'reference.npz')
 cap=cv2.VideoCapture(str(inp/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,c['lo'])
 cached={}
 for f in range(c['lo'],c['hi']):
  ok,bgr=cap.read();assert ok,(c['id'],f)
  rgb,pp,feat,hsv=near.frame_data(tracker,bgr)
  # Feature cache is shared across four arms, without masks or target state.
  cached[f]=(pp,feat,hsv)
 cap.release()
 return cal,cached

def run_segment(tracker,control,c,entry,arm,seed,seedscore,cached,cal,archive):
 model=tracker._predictor;R,K=cal['R'],cal['K'];times=cal['times_ns']/1e9;ki=np.linalg.inv(K)
 anchor=entry['anchor_frame'];identity=entry['id'];guides={r['frame']:r for r in entry['rows']}
 seedbox=near.describe_mask(seed)['bbox_xyxy'];rows=[];logs=[]
 if seedbox is None:return rows,[dict(status='SEED_EMPTY',id=identity,segment=entry['segment'],frame=anchor)]
 pp,feat,hsv=cached[anchor];seeddesc=near.inc.base.descriptor(seed,feat,hsv)
 rows.append(dict(id=identity,segment=entry['segment'],frame=anchor,batch=entry['batch'],
    rle=near.encode_mask(seed),box=seedbox,donor_eligible=seedscore>=.7,quality=near.quality(seed,entry['anchor_box'])))
 logs.append(dict(id=identity,segment=entry['segment'],frame=anchor,status='ANCHOR',memory_written=True,seed_score=seedscore))
 for direction in [1,-1]:
  target=list(range(anchor+direction,entry['end']+1 if direction==1 else entry['start']-1,direction))
  if not target:continue
  control.active=None
  state=near.init_state(tracker,pp,seed,feat,hsv,0,0.,np.eye(3))
  add_meta(state,0,seed,seedbox,R[anchor],times[anchor],1.)
  for step,f in enumerate(target,1):
   if f not in guides:
    logs.append(dict(id=identity,segment=entry['segment'],frame=f,status='NO_GUIDE'));continue
   box=guides[f]['box'];ppf,featf,hsvf=cached[f]
   Hto=K@R[f].T@R[anchor]@ki;Hfrom=K@R[anchor].T@R[f]@ki
   now=abs(times[f]-times[anchor]);state['motion'].predict(now)
   prior=state['motion'].image_prior(Hto,W,H)
   donor=near.donors_at(archive,f,identity,box)
   options=[('memory',None)]
   if arm=='external_box' and prior['search_box'] is not None:
    # Reliability-limited external box, avoiding the old large-lost-box failure.
    size=np.maximum(np.asarray(prior['box'])[2:]-np.asarray(prior['box'])[:2],2)
    if state['motion'].missed<3 and max(prior['sigma_xy'])<.4*max(size):
     options.append(('motion_box',prior['search_box']))
   candidates=[]
   for source,prompt in options:
    control.set(arm,K,R[f],f,times[f],prior,state['identity_reliability'],state['motion'].missed)
    candidate=near.inc.corrected_candidate(model,ppf,state,step,identity,source,prompt,featf,hsvf,donor)
    ev=evaluate(candidate,state,box)
    candidates.append((candidate,ev))
   raw=[dict(source=x['source'],box=x['box'],pixels=x['pixels'],sam_score=x['iou_score'],
      presence=x['presence'],score=e['score'],appearance=e['appearance'],valid=e['valid'],
      guide_iou=e['quality']['guide_iou'],outside=e['quality']['outside']) for x,e in candidates]
   selected=max([x for x in candidates if x[1]['valid']],key=lambda x:x[1]['score'],default=None)
   mask=None;written=False;negative=[];status='PENDING';quality=None;info={}
   if selected:
    candidate,ev=selected;mask=candidate['mask'];quality=ev['quality'];negative=candidate['negative_points']
    # The propagation prior is updated only from memory-accepted observations.
    z=near.transform_box(candidate['box'],Hfrom)
    if ev['write'] and z is not None:
     state['motion'].update(near.box_to_z(z),step)
     # Fixed visual-only promotion policy, identical for all four arms.
     promote=step-state['last_reference']>=5 and ev['appearance']>=.72 and candidate['iou_score']>=.75
     near.inc.commit(model,ppf,state,candidate,step,promote);written=True
     reliability=np.clip((ev['appearance']-.4)/.5,0,1)
     add_meta(state,step,mask,candidate['box'],R[f],times[f],reliability)
     if promote and ev['score']>.85:
      state['bank'].append(candidate['descriptor'])
      if len(state['bank'])>3:del state['bank'][1]
    else:
     state['motion'].missed+=1
    status='TRACKED' if written else 'UNCERTAIN'
    info=dict(selected=candidate['source'],sam_score=candidate['iou_score'],appearance=ev['appearance'])
   if mask is None:
    # Same offline proposal-box correction in every arm; count it explicitly.
    control.active=None
    negative=near.inc.near_points(donor,box)
    repaired,score,presence=image_candidate(model,ppf,box,negative)
    rq=near.quality(repaired,box)
    desc=near.inc.base.descriptor(repaired,featf,hsvf)
    app=near.appearance_similarity(desc,[seeddesc])
    if rq['valid'] and score>=.6 and app>=.5:
     mask=repaired;quality=rq;written=True;status='OFFLINE_REPAIR'
     state=near.init_state(tracker,ppf,mask,featf,hsvf,step,now,Hfrom)
     add_meta(state,step,mask,rq['box'],R[f],times[f],float(np.clip((app-.4)/.5,0,1)))
    else:
     state['motion'].missed+=1;status='PENDING_REVIEW'
    info=dict(sam_score=score,appearance=app,repair_quality=rq)
   if mask is not None:
    donor_ok=written and quality['guide_iou']>=.45 and quality['outside']<.08 and info['sam_score']>=.7
    # Retention threshold only affects exported annotation/negative donation.
    b=quality['box'];near_ok=b[3]-b[1]>=.08*H
    if near_ok:
     rows.append(dict(id=identity,segment=entry['segment'],frame=f,batch=entry['batch'],
        rle=near.encode_mask(mask),box=b,quality=quality,donor_eligible=bool(donor_ok)))
    else:status+='__BELOW_RETENTION'
   raw_best=selected[0] if selected else max(candidates,key=lambda x:x[1]['score'])[0]
   logs.append(near.clean(dict(id=identity,segment=entry['segment'],frame=f,direction=direction,
      step=step,status=status,memory_written=written,raw_candidates=raw,negative_points=negative,
      raw_rle=near.encode_mask(raw_best['mask']),raw_source=raw_best['source'],
      prior=prior,quality=quality,**info)))
   control.active=None;control.bias=None
 return rows,logs

def run_clip(tracker,control,c):
 dest=OUT/c['id'];dest.mkdir(parents=True,exist_ok=True)
 if all((dest/a/'complete.json').exists() for a in ARMS):return
 start=time.monotonic();cal,cached=load_frames(c,tracker)
 # Freeze shared seeds before any arm-specific donor masks exist.
 seeds={}
 for e in c['entries']:
  pp,_,_=cached[e['anchor_frame']]
  control.active=None
  mask,score,_=image_candidate(tracker._predictor,pp,e['anchor_box'])
  seeds[e['segment']]=(mask,score)
 near.inc.dumpgz(dest/'seeds.jsonl.gz',[dict(segment=k,rle=near.encode_mask(v[0]),score=v[1]) for k,v in seeds.items()])
 for arm in ARMS:
  if (dest/arm/'complete.json').exists():continue
  begin=time.monotonic();rows=[];logs=[];archive={}
  start_biased=control.biased_calls;start_calls=control.calls
  for batch in sorted({e['batch'] for e in c['entries']}):
   pending=[]
   for entry in [e for e in c['entries'] if e['batch']==batch]:
    seed,score=seeds[entry['segment']]
    rr,ll=run_segment(tracker,control,c,entry,arm,seed,score,cached,cal,archive)
    rows+=rr;logs+=ll;pending+=rr
   for r in pending:archive.setdefault(r['id'],{})[r['frame']]=r
  (dest/arm).mkdir(exist_ok=True)
  near.inc.dumpgz(dest/arm/'masks_rle.jsonl.gz',rows);near.inc.dumpgz(dest/arm/'decisions.jsonl.gz',logs)
  summary=dict(seconds=time.monotonic()-begin,masks=len(rows),segments=len(c['entries']),
     statuses=dict(collections.Counter(x['status'] for x in logs)),
     attention_calls=control.calls-start_calls,biased_attention_calls=control.biased_calls-start_biased,max_bias=control.max_bias,
     peak_cuda_gb=torch.cuda.max_memory_allocated()/1024**3)
  write(dest/arm/'complete.json',summary)
  print(c['id'],arm,json.dumps(summary),flush=True)
 del seeds,cached;control.active=None;control.bias=None;torch.cuda.empty_cache()
 write(dest/'complete.json',dict(seconds=time.monotonic()-start))

def main():
 p=argparse.ArgumentParser();p.add_argument('--clips',nargs='*');args=p.parse_args()
 selection=read(OUT/'selection.json')
 if args.clips:selection=[c for c in selection if c['id'] in args.clips]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  tracker=near.load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
  tracker._predictor.eval();control=MotionMemory(tracker._predictor)
  for c in selection:
   run_clip(tracker,control,c)
   write(OUT/'progress.json',dict(last_clip=c['id'],completed=sum((OUT/x['id']/'complete.json').exists() for x in read(OUT/'selection.json')),total=191))
 if not args.clips:write(OUT/'inference_complete.json',dict(clips=len(selection),arms=ARMS))
if __name__=='__main__':main()
