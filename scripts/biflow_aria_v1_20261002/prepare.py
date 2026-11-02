"""Freeze recording-disjoint folds and reference trajectories before training."""
from common import *
import collections

def main():
    assert not (OUT/'dataset.npz').exists(),'Existing data is preserved'
    rows=[];meta=[];sources={};excluded=[];camera=[];cammeta=[]
    for k in range(1,11):
        cid=f'{k:02d}';folder=LABELS/cid/'final';census=read(folder/'census.json');rr=gz(folder/'masks_rle.jsonl.gz');ds=gz(folder/'decisions.jsonl.gz')
        ref={(r['id'],r['frame']):r for r in rr};logs={(d['id'],d['frame']):d for d in ds};raw={(r['id'],r['frame']):r for r in gz(LABELS/cid/'masks_rle.jsonl.gz')}
        inp=VIDEOS/cid/'input';cal=np.load(inp/'reference.npz');R,K,ns=cal['R'],cal['K'],cal['times_ns'];alignment=read(inp/'alignment.json');info=read(inp/'metadata.json')['source'];record=info['recording_group'];N=len(R)
        for path in [folder/'census.json',folder/'masks_rle.jsonl.gz',folder/'decisions.jsonl.gz',LABELS/cid/'masks_rle.jsonl.gz',inp/'reference.npz',inp/'rgb.mp4']:
            sources[str(path.relative_to(ROOT))]=sha(path)
        def valid_pose(fs):return all(alignment[f]['status']=='valid' and alignment[f].get('quality',0)>=.5 for f in fs) and len({alignment[f]['graph'] for f in fs})==1
        for cutoff in range(5,N-15):
            fs=np.arange(cutoff-5,cutoff+16)
            if not valid_pose(fs):continue
            q=relative(R[fs]);camera.append(rotvec(q));cammeta.append(dict(clip=cid,cutoff=cutoff,recording=record,timestamp_ns=int(ns[cutoff])))
        for e in census['segments']:
            chunks=[]
            for f in range(e['start'],e['end']+1):
                okay=(e['id'],f) in ref and 'UNCERTAIN' not in logs[e['id'],f]['status']
                if not okay:continue
                if not chunks or f!=chunks[-1][-1]+1:chunks.append([])
                chunks[-1].append(f)
            for runnum,frames in enumerate(chunks):
                if len(frames)<21:continue
                for cutoff in range(frames[0]+5,frames[-1]-15+1,5):
                    fs=np.arange(cutoff-5,cutoff+16);key=f'{cid}_{e["id"]}_{cutoff:03d}'
                    if not valid_pose(fs):excluded.append(dict(key=key,reason='reference_pose'));continue
                    boxes=np.array([ref[e['id'],int(f)]['box'] for f in fs],float);points=center(boxes);hs=homographies(R[fs],K);comp,z=warp(points,hs)
                    if not np.isfinite(comp).all() or (z<=1e-6).any():excluded.append(dict(key=key,reason='reference_normalization_not_projectable'));continue
                    rawpoints=points[:6].copy();valid=np.ones(6)
                    for j,f in enumerate(fs[:6]):
                        r=raw.get((e['id'],int(f)))
                        if r is None:valid[j]=0.;rawpoints[j]=0.
                        else:rawpoints[j]=center(r['box'])
                    # Current position must be available to define the observation anchor.
                    if valid[-1]==0:rawpoints[-1]=points[5];valid[-1]=1.
                    edge=bool(((boxes[:,0]<=5)|(boxes[:,1]<=5)|(boxes[:,2]>=2555)|(boxes[:,3]>=1915)).any())
                    rows.append(dict(xy=points,boxes=boxes,raw=rawpoints,raw_valid=valid,H=hs,R=R[fs],K=K,cam=rotvec(relative(R[fs])),seconds=(ns[fs]-ns[cutoff])/1e9))
                    meta.append(dict(key=key,clip=cid,id=e['id'],segment=e['segment'],run=runnum,recording=record,cutoff=cutoff,frames=fs.tolist(),edge=edge,description=e['description'],source_start_s=info['start_s'],timestamp_ns=int(ns[cutoff]),nonoverlap=(cutoff-(frames[0]+5))%25==0))
    arrays={key:np.stack([r[key] for r in rows]).astype(np.float32) for key in rows[0]}
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'dataset.npz',**arrays,camera=np.stack(camera).astype(np.float32));write(OUT/'samples.json',meta);write(OUT/'camera_samples.json',cammeta)
    groups=sorted({m['recording'] for m in meta});folds=[]
    for i,test in enumerate(groups):
        val=groups[(i+1)%len(groups)];train=[g for g in groups if g not in [test,val]]
        folds.append(dict(fold=i,test=test,validation=val,train=train,train_indices=[j for j,m in enumerate(meta) if m['recording'] in train],val_indices=[j for j,m in enumerate(meta) if m['recording']==val],test_indices=[j for j,m in enumerate(meta) if m['recording']==test],camera_train_indices=[j for j,m in enumerate(cammeta) if m['recording'] in train],camera_val_indices=[j for j,m in enumerate(cammeta) if m['recording']==val]))
    config=dict(name='BiFlow-inspired Aria dual-task V1; independent compact implementation',paper='https://arxiv.org/html/2510.00405v2',inspected_official_commit='3c519c8c15139e27109d1434f9d7e70f6f8823bc',model='Per-person GRU shared encoder with historical and predicted camera conditions; two conditional flow-matching decoders; no social or candidate-attention reproduction',past_points=6,future_points=15,variants=['raw_future','raw_dual','comp_future','comp_dual'],seeds=[17,29,43],epochs=120,updates_per_epoch=2,batch_size=48,validation_every=10,learning_rate=.001,weight_decay=.01,reconstruction_weight=.5,hidden=64,flow_steps=10,inference_samples=5,noise_sigma=.15,report_prediction='Mean of five fixed Gaussian-start trajectories, not best-of-K; individual samples also saved',augmentation='Training: 40% current reviewed history, 20% same-ID pre-review history, 40% independent Gaussian jitter sd .005 of W/H + drop probability .20; final historical point always observed',evaluation_inputs=['reviewed','pre_review','controlled_noise'],camera='Separate learned history-only GRU predicts 15 relative rotation vectors; same frozen fold/seed camera predictor supplied to all four branches. Reference future rotation is diagnostic only.',selection='All continuous final-label runs >=21 frames, excluding V6_UNCERTAIN; stride 5 relative to run start; no performance-based pruning; border windows retained',validation='Six recording-disjoint outer folds; cyclic next recording validation, remaining four training; choose checkpoint by validation future pixel ADE; no test tuning',labels='Offline bidirectional SAM2 reviewed references, not independent human GT. This evaluates forecasting conditional on offline historical annotations and MPS reference histories, not online tracking or RGB-only inference.',coordinates='Rectified 2560x1920 image; compensation handles rotation only, not pure physical pedestrian motion',invalid_projection_penalty_px=32000.,implementation='Original compact experiment code; official source inspected for architecture and flow matching conventions; not copied official weights or a reproduction of published BEV scores.')
    write(OUT/'protocol.json',config);write(OUT/'folds.json',folds);write(OUT/'sources.json',sources);write(OUT/'selection.json',dict(samples=len(meta),clips=dict(collections.Counter(m['clip'] for m in meta)),recordings=dict(collections.Counter(m['recording'] for m in meta)),camera_windows=len(camera),excluded=excluded,raw_changed_history_points=int(np.sum(np.linalg.norm(arrays['raw']-arrays['xy'][:,:6],axis=-1)>1))))
    print(read(OUT/'selection.json'));print('folds',[(len(f['train_indices']),len(f['val_indices']),len(f['test_indices'])) for f in folds])
if __name__=='__main__':main()
