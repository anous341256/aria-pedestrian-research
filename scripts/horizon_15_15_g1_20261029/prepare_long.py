"""Build real contiguous 31-frame targets; never pad or join source IDs."""
from pathlib import Path
import sys,json,hashlib,collections,time
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/horizon_15_15_g1_20261029'
BULK=ROOT/'outputs/g1_bulk_association_20261029'
OLD=ROOT/'outputs/translation_forecast_20261020'
CAM=ROOT/'outputs/camera_full_data_20261002'
sys.path.insert(0,str(ROOT/'scripts/g1_bulk_association_20261029'))
from associate_bulk import load_clip,infer,frame_rule,contiguous
from bulk_common import read,write,sha
def geometry(xy,boxes,R,K,ts):
 rel=R[15].T@R;H=K@rel@np.linalg.inv(K)
 return dict(xy=np.asarray(xy),boxes=np.asarray(boxes),R=R,K=K,H=H,cam=Rotation.from_matrix(rel).as_rotvec(),seconds=(ts-ts[15])/1e9)
def old_windows():
 d=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');groups=collections.defaultdict(dict);templates={}
 for i,m in enumerate(meta):
  key=(m['recording'],m['clip'],m['id'],m['segment'],m['run']);templates[key]=m
  for j,f in enumerate(m['frames']):
   item={k:d[k][i,j] for k in ['xy','boxes','R']};item['ts']=m['timestamps_ns'][j];item['K']=d['K'][i]
   if f in groups[key]:assert all(np.array_equal(item[k],groups[key][f][k]) for k in item)
   groups[key][f]=item
 rows=[];mm=[]
 for key,obs in sorted(groups.items()):
  last=-100000
  for f in sorted(obs):
   if f%5 or not all(j in obs for j in range(f,f+31)):continue
   rr=[obs[j] for j in range(f,f+31)];ts=np.array([x['ts'] for x in rr]);assert ((np.diff(ts)>90e6)&(np.diff(ts)<110e6)).all()
   rows.append(geometry(np.array([r['xy'] for r in rr]),np.array([r['boxes'] for r in rr]),np.array([r['R'] for r in rr]),rr[0]['K'],ts))
   non=f-last>=31
   if non:last=f
   m=dict(templates[key]);m.update(key=f'old_{key[1]}_{key[2]}_{f:03d}',frames=list(range(f,f+31)),cutoff=f+15,timestamp_ns=int(ts[15]),timestamps_ns=ts.tolist(),nonoverlap=non,training_source='old_reviewed_union',input=f'outputs/sam2_g1_v6_v8_20261001/{key[1]}/input');mm.append(m)
 return rows,mm
def joined_bulk():
 inv=read(BULK/'inventory.json');by=collections.defaultdict(list);ee=collections.defaultdict(list)
 for c in inv['clips']:by[c['recording']].append(c)
 for e in inv['entries']:ee[e['clip']].append(e)
 runs=[]
 for rec,cs in sorted(by.items()):
  cur=[]
  for c in sorted(cs,key=lambda c:c['absolute_start']):
   if cur and c['absolute_start']!=cur[-1]['absolute_end']:
    if len(cur)>1:runs.append(cur)
    cur=[]
   cur.append(c)
  if len(cur)>1:runs.append(cur)
 rows=[];mm=[];summary=[]
 for ri,run in enumerate(runs):
  rid=f'joined_{ri+1:03d}';dest=OUT/'association'/f'{rid}.json';frames=[];features={};poses=[];entries=[];oldq={}
  for c in run:
   p,ff,ft=load_clip(c,ee[c['clip']]);p['c']=p['c']+p['world_origin'];poses.append(p);frames+=ff;features.update(ft);entries+=ee[c['clip']];oldq.update(read(BULK/'quality'/f'{c["clip"]}.json')['frame_audit'])
  assert all(np.allclose(p['K'],poses[0]['K'],atol=1e-10,rtol=0) for p in poses)
  p={k:np.concatenate([v[k] for v in poses]) for k in ['R','c','times_ns','selected_mps_ns']};p['K']=poses[0]['K'];p['c']-=p['c'][0].copy()
  assert ((np.diff(p['times_ns'])>90e6)&(np.diff(p['times_ns'])<110e6)).all()
  result=read(dest) if dest.exists() else infer(p,frames,features,True)
  if not dest.exists():write(dest,result)
  targets={t['target_key']:(e,t) for e in entries for t in e['targets']};groups=collections.defaultdict(list)
  for r in result['rows']:groups[r['track_id']].append(r)
  nall=0;nq=0
  for tid,rr in groups.items():
   for segment,track in enumerate(contiguous(rr)):
    last=-100000
    for k in range(0,len(track)-30,5):
     part=track[k:k+31];nall+=1;reasons=[];areas=[]
     for r in part:
      q=oldq[r['uid']];reasons+=frame_rule(r,q['quality'])[0];reasons+=q['manual_review_flags'];areas.append(q['quality']['area'])
     ratios=np.array(areas[1:])/np.maximum(1,areas[:-1])
     if reasons or (ratios<.4).any() or (ratios>2.5).any():continue
     fs=np.array([r['frame'] for r in part]);ts=p['times_ns'][fs];b=np.array([targets[r['uid']][1]['box'] for r in part],float);xy=(b[:,:2]+b[:,2:])/2
     R=p['R'][fs];one=Rotation.from_matrix(np.swapaxes(R[:-10],1,2)@R[10:]).magnitude()*180/np.pi;speed=Rotation.from_matrix(np.swapaxes(R[:-1],1,2)@R[1:]).magnitude()*180/np.pi/(np.diff(ts)/1e9)
     if one.max()>20 or np.percentile(speed,95)>50:continue
     rows.append(geometry(xy,b,R,p['K'],ts));non=fs[0]-last>=31
     if non:last=fs[0]
     mm.append(dict(key=f'{rid}_T{tid}_S{segment}_{fs[0]:03d}',clip=rid,id=f'T{tid}',segment=segment,recording=run[0]['recording'],frames=fs.tolist(),cutoff=int(fs[15]),timestamp_ns=int(ts[15]),timestamps_ns=ts.tolist(),nonoverlap=bool(non),training_source='bulk_continuous_reassociation',uids=[r['uid'] for r in part],source_frames=[dict(clip=targets[r['uid']][0]['clip'],frame=targets[r['uid']][0]['source_frame'],input=targets[r['uid']][0]['input']) for r in part],joined_clips=[c['clip'] for c in run]));nq+=1
  summary.append(dict(run=rid,recording=run[0]['recording'],chunks=[c['clip'] for c in run],frames=len(frames),continuous31_candidates=nall,quality31=nq));print(rid,len(frames),nall,nq,flush=True)
 write(OUT/'joined_inventory.json',summary)
 return rows,mm
def camera_and_translation(rows,meta):
 names=read(CAM/'inventory.json')['record_names'];cal=read(CAM/'calibrations.json');camrows=[];record_indices=[];starts=[];cts=[];tr=[];camera_mps=[];protected={}
 for rec,name in enumerate(names):
  pp=CAM/f'poses/{name}.npz';p=dict(np.load(pp));ts=p['times_ns'];path=ROOT/f'data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv';protected[str(pp.relative_to(ROOT))]=sha(pp);protected[str(path.relative_to(ROOT))]=sha(path)
  cols=['tracking_timestamp_us','quality_score','graph_uid']+[f't{a}_world_device' for a in 'xyz']+[f'q{a}_world_device' for a in 'xyzw'];df=pd.read_csv(path,usecols=cols);ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000;ii=np.searchsorted(ns,ts,side='right')-1;safe=ii.clip(0);age=ts-ns[safe];q=df.quality_score.to_numpy()[safe];graphs=df.graph_uid.to_numpy()[safe];rd=Rotation.from_quat(df[[f'q{a}_world_device' for a in 'xyzw']].to_numpy()[safe]).as_matrix();ext=np.array(cal[name]['T_device_camera']);pos=df[[f't{a}_world_device' for a in 'xyz']].to_numpy()[safe]+rd@ext[:3,3]
  valid=(abs(age)<=30000000)&(q>=.5)&np.isfinite(pos).all(1)
  for i,m in enumerate(meta):
   if m['recording']!=name:continue
   requested=np.array(m['timestamps_ns'][:16]);ix=np.searchsorted(ts,requested);assert np.array_equal(ts[ix],requested)
   assert valid[ix].all() and (ns[safe[ix]]<=requested[-1]).all() and len(set(graphs[ix]))==1
   rows[i]['translation']=np.einsum('ji,tj->ti',rows[i]['R'][15],pos[ix]-pos[ix[-1]]).astype(np.float32);rows[i]['translation_mps_ns']=ns[safe[ix]]
  ix=[]
  for f in range(len(ts)-30):
   fs=np.arange(f,f+31)
   if not p['valid'][fs].all() or not valid[fs[:16]].all():continue
   if not ((np.diff(ts[fs])>90e6)&(np.diff(ts[fs])<110e6)).all() or len(set(p['graph'][fs]))!=1 or len(set(graphs[fs[:16]]))!=1:continue
   if not (ns[safe[fs[:16]]]<=ts[fs[15]]).all():continue
   ix.append(fs)
  ix=np.array(ix);rel=np.einsum('nji,ntjk->ntik',p['R'][ix[:,15]],p['R'][ix]);camrows.append(Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(len(ix),31,3).astype(np.float32));tr.append(np.einsum('nji,ntj->nti',p['R'][ix[:,15]],pos[ix[:,:16]]-pos[ix[:,15,None]]).astype(np.float32));record_indices.extend([rec]*len(ix));starts.extend(ix[:,0].tolist());cts.extend(ts[ix[:,15]].tolist());camera_mps.append(ns[safe[ix[:,:16]]]);print('CAM',name,len(ix),flush=True)
 arrays=dict(rotation=np.concatenate(camrows),translation=np.concatenate(tr),record_index=np.array(record_indices),start_frame=np.array(starts),cutoff_ns=np.array(cts),translation_mps_ns=np.concatenate(camera_mps))
 np.savez_compressed(OUT/'camera_dataset.npz',**arrays);write(OUT/'camera_record_names.json',names);write(OUT/'pose_sources.json',protected)
 return arrays,names
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 old,om=old_windows();new,nm=joined_bulk();keep=[];duplicates=[]
 for j,m in enumerate(nm):
  same=[i for i,o in enumerate(om) if o['recording']==m['recording'] and o['timestamps_ns']==m['timestamps_ns'] and np.linalg.norm(old[i]['xy']-new[j]['xy'],axis=-1).max()<=2]
  if same:duplicates.append(dict(new=j,old=same))
  else:keep.append(j)
 rows=old+[new[j] for j in keep];meta=om+[nm[j] for j in keep];cd,names=camera_and_translation(rows,meta);keys=rows[0].keys();arrays={k:np.stack([r[k] for r in rows]) for k in keys}
 for k,v in arrays.items():assert np.isfinite(v).all(),k
 assert np.array_equal(arrays['translation'][:,-1],np.zeros((len(meta),3)))
 groups=sorted({m['recording'] for m in om});folds=[]
 for fi,g in enumerate(groups):
  val=groups[(fi+1)%len(groups)];oi=[i for i,m in enumerate(om) if m['recording'] not in [g,val]];ex=[i for i,m in enumerate(meta) if m['recording'] not in [g,val]]
  cf={k:np.flatnonzero(np.isin(cd['record_index'],[names.index(x) for x in rs])).tolist() for k,rs in [('train_indices',[n for n in names if n not in [g,val]]),('val_indices',[val]),('test_indices',[g])]}
  folds.append(dict(fold=fi,test=g,validation=val,train_indices=oi,expanded_train_indices=ex,val_indices=[i for i,m in enumerate(om) if m['recording']==val],test_indices=[i for i,m in enumerate(om) if m['recording']==g],camera=cf))
 protocol=dict(version=OUT.name,history_points=16,future_points=15,seeds=[17,29,43],camera_updates=2000,person_updates=240,person_batch=48,learning_rate=.001,weight_decay=.01,validation_every=20,camera_validation_every=100,augmentation=False,initialization='Train both 15-output camera and person from scratch, since old5-output heads are incompatible. Same architecture family and historical translation inputs. No old checkpoints that could include new heldout data.',arms=['original_long','expanded_long'],camera='Same shared matching camera for both arms, trained on full31-frame camera windows excluding BOTH test and validation recordings.',model='Original separated camera GRU6->32/MLP future rotation; person GRU9->64/context/future-flow/history-flow. History rotation compensation and future rotation reprojection. Translation is historical feature only. 5 draws/10flowsteps.',selection='Camera validation rotvec MSE; person validation residualADE, step0eligible. 240steps recording-balanced, no augmentation. No test tuning.',data='Old31 windows from exact consistent union of existing117 reviewed references. New31 from temporally adjacent locked bulk intervals; continuous LocalTracker rerun, no source-ID stitching, no new SAM or depth. Existing frame quality and manual exclusions, stride5. Entire new31 window must satisfy max1second rotation20deg and angularspeedP95 50deg/s, including chunk boundaries. Automatic offline labels.',comparison='Same long-window heldout cohort for original-only and expanded. Report first5 and full15 future errors. Cannot directly compare with previous117/.5s score86.68.',original_windows=len(om),added_windows=len(keep),combined_windows=len(meta),test_recordings=groups,camera_windows=len(cd['rotation']),camera_recordings=names)
 write(OUT/'protocol.json',protocol);write(ROOT/'plans/HORIZON_15_15_G1_20261029.txt',protocol);write(OUT/'samples.json',meta);write(OUT/'folds.json',folds);write(OUT/'duplicates.json',duplicates);np.savez_compressed(OUT/'dataset.npz',**arrays)
 sources=[OLD/'dataset.npz',OLD/'samples.json',BULK/'inventory.json',BULK/'protocol.json',BULK/'visual_review_notes.json']
 sources+=list((ROOT/'scripts/translation_forecast_20261020').glob('*.py'))+list((ROOT/'scripts/partial_appearance_20261028').glob('*.py'))
 for s in read(OUT/'joined_inventory.json'):
  for cid in s['chunks']:
   sources += [BULK/'poses'/f'{cid}.npz',BULK/'features'/f'{cid}.npz',BULK/'quality'/f'{cid}.json',BULK/'tracking'/f'{cid}.json']
 write(OUT/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in set(sources)})
 summary=dict(passed=True,old_windows=len(om),new_windows=len(keep),duplicates=len(duplicates),recordings=len({m['recording'] for m in meta}),fold_counts=[dict(fold=f['fold'],old=len(f['train_indices']),expanded=len(f['expanded_train_indices']),val=len(f['val_indices']),test=len(f['test_indices'])) for f in folds],camera_windows=len(cd['rotation']))
 write(OUT/'preflight.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
