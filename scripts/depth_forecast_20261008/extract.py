"""Complete fixed-cohort stereo extraction. Future measurements are archive-only."""
import sys,json,gzip,time,hashlib,traceback,gc,shutil,argparse
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008';SOURCE=ROOT/'outputs/transformer_uncovered200_20261006'
sys.path.insert(0,str(ROOT/'scripts/aria_stereo_validation_20261008'))
from validate import official,sha
from analyze import decode,project
from benchmark import paired

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    for attempt in range(100):
        try:temp.replace(p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.05)

def paths(m):
    cid=m['clip'];s=m['source_version']
    if s=='old117':return ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input',ROOT/f'outputs/sam2_near_people_20261002/{cid}/final'
    if s=='new82':base=ROOT/f'outputs/sam2_g1_expansion_20261005/{cid}'
    elif s=='g1_matched97_20261006':base=ROOT/f'outputs/sam2_g1_matched97_20261006/{cid}'
    elif s=='uncovered200_user_review_20261006':base=ROOT/f'outputs/sam2_g1_uncovered200_20261006/{m["source_clip"]}'
    else:raise ValueError(s)
    return base/'input',base/('review_v2' if s=='uncovered200_user_review_20261006' else 'final')

def inventory():
    meta=read(SOURCE/'data/samples.json');data=dict(np.load(SOURCE/'data/dataset.npz'));old=dict(np.load(ROOT/'outputs/biflow_aria_v1_20261002/dataset.npz'))
    assert len(meta)==374 and all(np.array_equal(data[k][:117],old[k]) for k in data if k in old)
    union=sorted({i for f in read(SOURCE/'folds.json') for i in f['train_indices']});assert len(union)==373
    cache={};entries={};sample_refs=[];protected={}
    for p in [SOURCE/'data/samples.json',SOURCE/'data/dataset.npz',SOURCE/'folds.json']:
        protected[str(p.relative_to(ROOT))]=sha(p)
    for i,m in enumerate(meta):
        cid=m['clip'];inp,ann=paths(m)
        if cid not in cache:
            with gzip.open(ann/'masks_rle.jsonl.gz','rt') as f:rows={(r['id'],r['frame']):r for r in map(json.loads,f)}
            ref=dict(np.load(inp/'reference.npz'));cache[cid]=(ref,rows)
            for p in [inp/'reference.npz',inp/'metadata.json',ann/'masks_rle.jsonl.gz']:protected[str(p.relative_to(ROOT))]=sha(p)
        ref,rows=cache[cid];keys=[]
        assert int(ref['times_ns'][m['cutoff']])==m['timestamp_ns']
        assert np.max(np.abs(ref['K']-data['K'][i]))<1e-3
        for j,f in enumerate(m['frames']):
            ts=int(ref['times_ns'][f]);key=f"{m['recording']}_{ts}";keys.append(key)
            if key not in entries:entries[key]=dict(key=key,recording=m['recording'],rgb_timestamp_ns=ts,historical=False,targets=[],input=str(inp.relative_to(ROOT)),reference_frame=f)
            e=entries[key];e['historical']|=j<6
            ident=f"{cid}/{m['id']}/{f}"
            if not any(t['target_key']==ident for t in e['targets']):
                assert (m['id'],f) in rows,(cid,m['id'],f)
                r=rows[(m['id'],f)];e['targets'].append(dict(target_key=ident,clip=cid,id=m['id'],frame=f,annotation=str(ann.relative_to(ROOT)),rle=r['rle']))
        sample_refs.append(dict(index=i,key=m['key'],frame_keys=keys,target_keys=[f"{cid}/{m['id']}/{f}" for f in m['frames']]))
    ordered=sorted(entries.values(),key=lambda e:(not e['historical'],e['recording'],e['rgb_timestamp_ns']))
    # Reuse only identical timestamp, intrinsics and source images from the earlier pilot.
    pilot={}
    for p in (ROOT/'outputs/aria_stereo_validation_20261008').glob('*/*/metadata.json'):
        r=read(p);pilot[f"unknown_20260803_{r['recording']}_{r['rgb_timestamp_ns']}"]=str(p.parent.relative_to(ROOT))
    for e in ordered:e['pilot_cache']=pilot.get(e['key'])
    plan=dict(version='depth_forecast_20261008',samples=374,actual_previous_train_union=union,
        unique_frames=len(ordered),historical_unique_frames=sum(e['historical'] for e in ordered),entries=ordered,sample_refs=sample_refs,
        frozen_rules=dict(min_points=30,minimum_lr_fraction=.7,z_range_m=[.5,30],maximum_relative_iqr=.25,erosion_kernel=15,stereo_timing='Both exposures no later than RGB timestamp; left queried BEFORE RGB minus1ms, max RGB gap40ms'),
        depth_features=['log(z)/log(30)','adjacent delta log(z), clipped [-1,1]','valid','left_right_fraction','IQR/z clipped [0,1]'],
        no_future_input='Only first6 per-window frames; no temporal interpolation, backward filling or future normalization. Future15 depth archived only.',
        model='Latest dual-branch cross-attention, exact original117 test/validation indices and six held-out recording pairs. Two arms initialized from same previous cross checkpoint; frozen original GRU/flow backbones.',
        training='All previous actual train-union samples outside each held-out pair, exact old validation/test117 windows. New held-out-recording rows remain unused. Identical240 updates,48 anchors,18 seeds/folds per arm. No tests used for choices.',
        missing='Keep every forecasting window, set unusable depth features to zero with explicit valid bit; no depth-based sample deletion.',
        caveat='Offline reviewed masks and MPS; depth estimates are pseudo measurements, not independent true 3D labels.')
    if (OUT/'extraction_plan.json').exists():assert read(OUT/'extraction_plan.json')==plan
    else:write(OUT/'extraction_plan.json',plan);write(OUT/'protected_sources.json',protected)
    write(OUT/'inventory_summary.json',{k:v for k,v in plan.items() if k not in ['entries','sample_refs','actual_previous_train_union']})
    print('INVENTORY',len(ordered),'frames;',sum(e['historical'] for e in ordered),'historical;',len(cache),'clips',flush=True)

def prepare_pair(provider,mps,e,ref):
    ts=e['rgb_timestamp_ns'];ls=provider.get_stream_id_from_label('slam-front-left');rs=provider.get_stream_id_from_label('slam-front-right')
    li,lr=provider.get_image_data_by_time_ns(ls,ts-1000000,official.TimeDomain.DEVICE_TIME,official.TimeQueryOptions.BEFORE);lt=int(lr.capture_timestamp_ns)
    ri,rr=provider.get_image_data_by_time_ns(rs,lt,official.TimeDomain.DEVICE_TIME,official.TimeQueryOptions.CLOSEST);rt=int(rr.capture_timestamp_ns)
    assert abs(lt-rt)<=1000000,'stereo timing invalid';assert max(lt,rt)<=ts,'future stereo exposure';assert abs(lt-ts)<=40000000,'RGB timing invalid'
    online=mps.get_online_calibration(lt,official.TimeQueryOptions.CLOSEST)
    lc=online.get_camera_calib('slam-front-left');rc=online.get_camera_calib('slam-front-right')
    tl=lc.get_transform_device_camera().inverse();tr=rc.get_transform_device_camera().inverse()
    linear=official.fisheye_to_linear_calib(lc,focal_scale=1.25,output_width=512,output_height=512,use_original_pp=True)
    rl,rrr=official.create_scanline_rectified_cameras(tl,tr)
    l=li.to_numpy_array();r=ri.to_numpy_array()
    left,right=official.rectify_stereo_pair(l,r,lc,rc,linear,linear,rl,rrr,official.InterpolationMethod.BILINEAR)
    sl,sr=official.rectify_stereo_pair(np.full_like(l,255),np.full_like(r,255),lc,rc,linear,linear,rl,rrr,official.InterpolationMethod.BILINEAR)
    baseline=float(np.linalg.norm((tl@tr.inverse()).translation()))
    world_cv=official.compute_T_world_rectCam(mps.get_interpolated_closed_loop_pose(lt).transform_world_device,tl,rl).to_matrix()
    world_rgb=mps.get_interpolated_closed_loop_pose(ts).transform_world_device.to_matrix()@ref['T_device_camera']
    return dict(left=left,right=right,support_left=sl>=254,support_right=sr>=254,intrinsics=linear.get_projection_params(),baseline_m=baseline,
        T_rgb_cv=np.linalg.inv(world_rgb)@world_cv,T_world_cv=world_cv,K_rgb=ref['K']),dict(left_timestamp_ns=lt,right_timestamp_ns=rt,rgb_timestamp_ns=ts,stereo_skew_ms=abs(lt-rt)/1e6,rgb_skew_ms=abs(lt-ts)/1e6)

def measure(e,pair,disp,lr):
    z=official.disparity_to_depth(disp,float(pair['baseline_m']),float(pair['intrinsics'][0]));y,x=np.indices(z.shape);xr=x-disp;xi=np.clip(np.nan_to_num(xr),0,511).astype(int)
    support=pair['support_left']&pair['support_right'][y,xi]&(xr>=0)&(xr<=511)
    valid=support&lr&np.isfinite(z)&(z>0)&np.isfinite(disp)&(disp>0)
    pts=project(pair,z,support,valid);out=[]
    for t in e['targets']:
        mask=cv2.erode(decode(t['rle']),np.ones((15,15),np.uint8))>0;member=mask[pts['v'],pts['u']];good=member&pts['valid'];n=int(good.sum());a=pts['z_rgb'][good]
        med=float(np.median(a)) if n>=30 else None;iqr=float(np.percentile(a,75)-np.percentile(a,25)) if n>=30 else None;ratio=n/max(1,int(member.sum()))
        reasons=[]
        if n<30:reasons.append('insufficient_points')
        if ratio<.7:reasons.append('low_LR_consistency')
        if med is not None:
            if not .5<=med<=30:reasons.append('outside_depth_range')
            if iqr/med>.25:reasons.append('broad_depth_distribution')
        out.append(dict(target_key=t['target_key'],points=n,supported_points=int(member.sum()),lr_fraction=ratio,z_median_m=med,z_iqr_m=iqr,usable=not reasons,reasons=reasons))
    return z,valid,out

def run():
    import torch
    torch.set_num_threads(2)
    plan=read(OUT/'extraction_plan.json');benchmark=read(OUT/'batch_validation.json')
    # Batch execution accepted only with close agreement before training; otherwise exact official sequential.
    use_batch=all(r['disparity_max_diff']<.05 and r['lr_mask_agreement']>.999 for r in benchmark)
    write(OUT/'runtime.json',dict(use_batched_LR=use_batch,benchmark=benchmark))
    model,cfg=official.load_foundation_stereo(str(ROOT/'models/foundation_stereo_20261008/23-51-11/model_best_bp2.pth'))
    refs={};current=None;provider=None;mps=None;start=time.time();done=0;failed=0
    for n,e in enumerate(plan['entries']):
        folder=OUT/'frames'/e['key'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists():done+=1;continue
        begin=time.time()
        try:
            ip=e['input']
            if ip not in refs:refs[ip]=dict(np.load(ROOT/ip/'reference.npz'))
            ref=refs[ip];cached=e.get('pilot_cache')
            if cached:
                timing=read(ROOT/cached/'metadata.json')
                if timing['left_timestamp_ns']>e['rgb_timestamp_ns']-1000000 or timing['right_timestamp_ns']>e['rgb_timestamp_ns']:cached=None
            if cached:
                p=ROOT/cached;pair=dict(np.load(p/'pair.npz'));dep=np.load(p/'depth.npz');disp=dep['disparity_px'];lr=dep['lr_consistent'];timing=read(p/'metadata.json')
                assert np.max(np.abs(pair['K_rgb']-ref['K']))<1e-8
            else:
                if current!=e['recording']:
                    provider=None;mps=None;gc.collect();current=e['recording']
                    provider=official.data_provider.create_vrs_data_provider(str(ROOT/f'data/Aria/raw_vrs/{current}.vrs'))
                    mps=official.MpsDataProvider(official.MpsDataPathsProvider(str(ROOT/f'data/Aria/raw_vrs/mps_{current}_vrs')).get_data_paths())
                pair,timing=prepare_pair(provider,mps,e,ref)
                disp,lr=(paired(model,pair['left'],pair['right'],cfg) if use_batch else official.run_lr_consistency(model,pair['left'],pair['right'],cfg))
            z,valid,values=measure(e,pair,disp,lr)
            np.savez_compressed(folder/'depth.npz',disparity_px=disp,depth_m=z,lr_consistent=lr,valid=valid,**pair)
            write(folder/'measurements.json',values);write(folder/'complete.json',dict(key=e['key'],historical=e['historical'],timing=timing,cache_source=cached,seconds=time.time()-begin,usable_targets=sum(x['usable'] for x in values),targets=len(values)));done+=1
        except Exception as exc:
            failed+=1;write(folder/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print('FAILED',e['key'],str(exc),flush=True)
        progress=dict(processed=n+1,total=len(plan['entries']),successful=done,failed=failed,historical=e['historical'],recording=e['recording'],elapsed_seconds=time.time()-start,last_key=e['key'])
        write(OUT/'extraction_progress.json',progress)
        if (n+1)%10==0:print('PROGRESS',json.dumps(progress),flush=True)
    write(OUT/'extraction_complete.json',dict(successful=done,failed=failed,total=len(plan['entries']),seconds=time.time()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['inventory','run']);a=ap.parse_args()
    if a.mode=='inventory':inventory()
    else:run()
