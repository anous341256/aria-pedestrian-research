"""Freeze user exclusions, qualify sampled person windows, append train-only data."""
from shared import *
from common import center,homographies,relative,rotvec
from collections import Counter
import copy

EXCLUDED=['010','012','015','016','017','019','037','038','058','069','070','074','083']

def decode_box(row):
    counts=np.asarray(row['rle'][:-4],np.int64)
    x1,y1,x2,y2=map(int,row['rle'][-4:])
    assert counts.sum()==(x2-x1+1)*(y2-y1+1)
    mask=np.repeat(np.arange(len(counts))%2,counts).reshape(y2-y1+1,x2-x1+1)
    yy,xx=np.where(mask)
    assert len(xx)==row['quality']['pixels']
    return [int(xx.min()+x1),int(yy.min()+y1),int(xx.max()+x1+1),int(yy.max()+y1+1)]

def angle(a,b):
    rr=np.swapaxes(a,-1,-2)@b
    return np.degrees(np.arccos(np.clip((np.trace(rr,axis1=-2,axis2=-1)-1)/2,-1,1)))

def main():
    assert not (OUT/'selection_locked.json').exists(),'Already locked; do not silently regenerate'
    DATA.mkdir(parents=True,exist_ok=True)
    windows=read(EXPANSION/'selected_windows.json');clips=read(EXPANSION/'selected_clips.json')
    clipmap={c['clip']:c for c in clips};wm={w['window_id']:w for w in windows}
    candidates=read(EXPANSION/'continuous21_person_window_candidates.json')
    oldmeta=read(BASELINE/'data/samples.json');olddata=dict(np.load(BASELINE/'data/dataset.npz'));nold=len(oldmeta)
    assert nold==296
    protected={}
    # Protect complete source annotations and old trained assets, including rejected clips.
    paths=list(BASELINE.rglob('*'))
    paths += list((ROOT/'outputs/g1_near_distribution_subset_20261006').glob('*.json'))
    paths += [ROOT/'outputs/g1_near_distribution_subset_20261006/matched/metrics.json']
    paths += list(EXPANSION.glob('*.json'))
    for c in clips:
        p=EXPANSION/c['clip']
        paths += list(p.glob('*.json'))+list(p.glob('*.gz'))+list((p/'review_v2').glob('*'))
        paths += list((p/'input').glob('*.json'))+[p/'input/reference.npz']
    paths += list((ROOT/'scripts/transformer_matched97_20261006').glob('*.py'))
    for p in sorted(set(paths)):
        if p.is_file():protected[p.relative_to(ROOT).as_posix()]=sha(p)
    write(OUT/'protected_previous_hashes.json',protected)
    cache={};newmeta=[];newdata=[];ledger=[];checked=set()
    def load_clip(cid):
        if cid not in cache:
            p=EXPANSION/cid
            cache[cid]=dict(rows={(r['id'],r['frame']):r for r in gz(p/'review_v2/masks_rle.jsonl.gz')},
                raw={(r['id'],r['frame']):r for r in gz(p/'masks_rle.jsonl.gz')},
                decisions={(r['id'],r['frame']):r for r in gz(p/'decisions.jsonl.gz')},
                numeric={(r['id'],r['frame']):r for r in read(p/'review_v2/mask_metrics.json')},
                audit=read(p/'review_v2/audit.json'),census=read(p/'review_v2/census.json'),
                cal=dict(np.load(p/'input/reference.npz')),alignment=read(p/'input/alignment.json'))
        return cache[cid]
    for candidate in candidates:
        cid=candidate['clip'];identity=candidate['id'];w=wm[candidate['window_id']]
        reasons=set();fs=np.arange(w['local_start_frame'],w['local_start_frame']+21);cutoff=int(fs[5])
        if cid in EXCLUDED:
            ledger.append(dict(**candidate,accepted=False,reasons=['user_excluded_clip']));continue
        c=load_clip(cid);cal=c['cal'];R=cal['R'][fs];K=cal['K'];ns=cal['times_ns'][fs];seconds=(ns-ns[5])/1e9
        if c['audit']['scene_exclusion']:reasons.add('scene_exclusion')
        if candidate['donor_dependency_flag']:reasons.add('rejected_negative_donor_dependency')
        boxes=[];segments=set()
        for f in fs:
            key=identity,int(f);row=c['rows'].get(key);decision=c['decisions'].get(key,{})
            if row is None:reasons.add('missing_mask');continue
            segments.add(row['segment']);q=c['numeric'][key];box=q['box'];boxes.append(box)
            ck=(cid,identity,int(f))
            if ck not in checked:
                assert decode_box(row)==box,ck
                assert c['raw'][key]['rle']==row['rle'],'Expected selection-only review'
                checked.add(ck)
            if any(s in decision.get('status','MISSING') for s in ['UNCERTAIN','PENDING','MISSING']):reasons.add('uncertain_or_missing_decision')
            if box[0]<=5 or box[1]<=5 or box[2]>=2555 or box[3]>=1915:reasons.add('image_border_truncation')
            if box[3]-box[1]<.08*1920:reasons.add('below_retention_scale')
            if q['guide_iou']<.45 or q['outside']>.1:reasons.add('mask_guide_disagreement')
        if len(segments)!=1:reasons.add('crosses_identity_segment')
        for e in c['census']['segments']:
            if e['id']!=identity:continue
            if e.get('review_only_reason'):reasons.add(e['review_only_reason'])
            for ex in e.get('forecast_excluded_intervals',[]):
                if max(fs[0],ex['start'])<=min(fs[-1],ex['end']):reasons.add(ex['reason'])
        aa=[c['alignment'][int(f)] for f in fs]
        if not all(a['status']=='valid' and a.get('quality',0)>=.5 and a['nearest_gap_ms']<=30 for a in aa) or len({a['graph'] for a in aa})!=1:reasons.add('pose_alignment')
        if not np.all((np.diff(seconds)>.08)&(np.diff(seconds)<.12)):reasons.add('time_step')
        motion=dict(max_1s_rotation_deg=float(max(angle(R[:-10],R[10:]))),omega_p95_deg_s=float(np.percentile(angle(R[:-1],R[1:])/np.diff(seconds),95)))
        if motion['max_1s_rotation_deg']>20 or motion['omega_p95_deg_s']>50:reasons.add('g1_camera_screen')
        if len(boxes)==21:
            boxes=np.asarray(boxes,float);xy=center(boxes);H=homographies(R,K);stable,z=warp(xy,H)
            if not np.isfinite(stable).all() or (z<=1e-6).any():reasons.add('stabilization_geometry')
        ledger.append(dict(**candidate,accepted=not reasons,reasons=sorted(reasons)))
        if reasons:continue
        arr=dict(xy=xy,boxes=boxes,H=H,R=R,K=K,cam=rotvec(relative(R)),seconds=seconds,
                 raw=xy[:P].copy(),raw_valid=np.ones(P))
        m=dict(key=f'u200_{cid}_{identity}_{cutoff:03d}',clip=f'u200_{cid}',source_clip=cid,id=identity,
            segment=next(iter(segments)),recording=w['recording'],cutoff=cutoff,frames=fs.tolist(),
            timestamp_ns=int(ns[5]),source_start_s=clipmap[cid]['start_s'],source_version='uncovered200_user_review_20261006',
            window_id=w['window_id'],nonoverlap=False,static_control=False,static_control_status='not_separately_adjudicated',
            edge=False,classification='G1_user_retained_clear_numeric',motion_screen=motion,
            annotation=str((EXPANSION/cid/'review_v2').relative_to(ROOT)),
            raw_history_note='V2 is a selection of unmodified raw same-ID masks; canonical exclusive boxes match exactly.')
        newmeta.append(m);newdata.append(arr)
    assert newmeta
    # Stable per-person nonoverlap flags for possible later diagnostics; added samples are train-only now.
    last={}
    for m in newmeta:
        group=(m['clip'],m['id']);start=m['frames'][0]
        if start>last.get(group,-1):m['nonoverlap']=True;last[group]=m['frames'][-1]
    newarr={k:np.stack([a[k] for a in newdata]).astype(np.float32) for k in olddata}
    combined={k:np.concatenate([a,newarr[k]]) for k,a in olddata.items()};meta=oldmeta+newmeta
    assert len({m['key'] for m in meta})==len(meta)
    for k,a in combined.items():assert np.array_equal(a[:nold],olddata[k]) and np.isfinite(a).all(),k
    np.savez_compressed(DATA/'dataset.npz',**combined);write(DATA/'samples.json',meta)
    np.savez_compressed(DATA/'added_dataset.npz',**newarr);write(DATA/'added_samples.json',newmeta)
    write(DATA/'candidate_decisions.json',ledger)
    previous=read(EXPANSION/'previous_coverage.json')
    # Exact intervals were already checked in the annotation audit; verify again against source metadata.
    assert isinstance(previous,list) and len(previous)==63
    overlaps=[]
    for m in newmeta:
        c=clipmap[m['source_clip']]
        for p in previous:
            if p['recording']==c['recording_group'] and max(p['start_frame'],c['source_start_frame'])<min(p['end_frame_exclusive'],c['source_end_frame_exclusive']):overlaps.append([m['key'],p])
    assert not overlaps
    folds=[]
    for o in read(BASELINE/'folds.json'):
        added=[i for i in range(nold,len(meta)) if meta[i]['recording'] not in [o['test'],o['validation']]]
        f={k:copy.deepcopy(o[k]) for k in ['fold','test','validation','val_indices','test_indices']}
        f.update(train_indices=o['train_indices']+added,previous_train_indices=o['train_indices'],new_train_indices=added,
            unused_new_indices=[i for i in range(nold,len(meta)) if i not in added],
            train=sorted({meta[i]['recording'] for i in o['train_indices']+added}))
        assert not {meta[i]['recording'] for i in f['train_indices']} & {f['test'],f['validation']}
        folds.append(f)
    write(OUT/'folds.json',folds)
    cfg=read(BASELINE/'protocol.json')
    cfg.update(name='User-reviewed uncovered200 expansion',approved='User exclusions plus merge remaining eligible annotations and rerun',
        plan='selection_locked.json',cohort=f'Exact previous 296 prepared samples plus {len(newmeta)} new clear person windows; only previous actual train indices retained',
        old_test_comparison='Each old train list preserved, new labels append only when outside both held-out recordings; validation/test remain exact old199',
        sampler='Unchanged uniform recording then uniform window; 240 updates at batch48',
        training_augmentation='Unchanged40/20/40; new V2 masks are selection-only, so reviewed and raw centers equal; controlled noise unchanged',
        test_rule='Exact previous199 windows only; fixed strict near87 subgroup; new labels never headline tests',
        primary='All six architectures, no deployment winner selected by test score',
        camera='Reuse exact33 matched97_new held-out camera checkpoints; no camera training',
        raw_box_convention='Decode RLE and use exclusive xmax/ymax, matching old prepared arrays')
    write(OUT/'protocol.json',cfg)
    oldunion=set(i for f in folds for i in f['previous_train_indices']);union=set(i for f in folds for i in f['train_indices'])
    stats=dict(selected_time_windows=200,selected_clips=114,user_excluded_clips=EXCLUDED,
        retained_clips=114-len(EXCLUDED),remaining_selected_time_windows=sum(w['clip'] not in EXCLUDED for w in windows),
        all_continuous_candidates=len(candidates),candidates_after_user_exclusion=sum(c['clip'] not in EXCLUDED for c in candidates),
        accepted_person_windows=len(newmeta),contributing_clips=len({m['source_clip'] for m in newmeta}),
        contributing_time_windows=len({m['window_id'] for m in newmeta}),clip_local_ids=len({(m['clip'],m['id']) for m in newmeta}),
        contributing_recordings=len({m['recording'] for m in newmeta}),new_recordings=sorted({m['recording'] for m in newmeta}-{oldmeta[i]['recording'] for i in oldunion}),
        old_prepared=nold,combined_prepared=len(meta),previous_actual_train_union=len(oldunion),combined_actual_train_union=len(union),
        rejection_counts=dict(Counter(r for m in ledger for r in m['reasons'])),
        fold_counts=[dict(fold=f['fold'],before=len(f['previous_train_indices']),added=len(f['new_train_indices']),after=len(f['train_indices'])) for f in folds],
        decoded_person_frames=len(checked),source_time_overlaps=overlaps)
    write(DATA/'summary.json',stats)
    write(OUT/'selection_locked.json',dict(user_rejection=EXCLUDED,repeated_83_deduplicated=True,
        sampling='Only the original selected200 time windows; no extra shifted windows or replacement of failed windows',
        rules='Single ID segment throughout21 frames; V2 retained masks; clear original thresholds; reject flagged negative-donor dependencies; valid geometry/G1 camera rules',
        selected_keys=[m['key'] for m in newmeta],summary=stats,
        strict_near_test_indices=read(ROOT/'outputs/g1_near_distribution_subset_20261006/selection_locked.json')['test_near_G1_indices']))
    locked=[DATA/'dataset.npz',DATA/'samples.json',DATA/'added_dataset.npz',DATA/'added_samples.json',DATA/'candidate_decisions.json',OUT/'folds.json',OUT/'protocol.json',OUT/'selection_locked.json']
    write(OUT/'locked_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in locked})
    print(stats,flush=True)

if __name__=='__main__':main()
