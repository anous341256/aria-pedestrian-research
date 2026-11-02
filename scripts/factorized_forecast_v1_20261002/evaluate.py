"""Held-out image/component metrics and error attribution; no selection/tuning."""
from shared import *


def reduce_rows(rows):
    if not rows: return None
    keys=['ADE05','ADE10','ADE15','FDE15']
    def mean(rr): return {k:float(np.mean([r['metrics'][k] for r in rr])) for k in keys}
    by_record={g:mean([r for r in rows if r['recording']==g]) for g in sorted({r['recording'] for r in rows})}
    by_track={g:mean([r for r in rows if r['track']==g]) for g in sorted({r['track'] for r in rows})}
    return dict(windows=len({r['key'] for r in rows}),prediction_rows=len(rows),window_mean=mean(rows),
                recording_equal={k:float(np.mean([v[k] for v in by_record.values()])) for k in keys},
                track_equal={k:float(np.mean([v[k] for v in by_track.values()])) for k in keys},
                per_recording=by_record,per_seed={str(s):mean([r for r in rows if r['seed']==s]) for s in sorted({r['seed'] for r in rows})},
                invalid_steps=sum(r['metrics']['invalid'] for r in rows))


def angle(a,b):
    A=rotmat(a.reshape(-1,3)); B=rotmat(b.reshape(-1,3))
    return np.degrees(np.arccos(np.clip((np.sum(A*B,axis=(1,2))-1)/2,-1,1))).reshape(a.shape[:-1])


def main():
    assert read(OUT/'training_complete.json')['passed']
    cfg=read(OUT/'protocol.json'); meta=read(OLD/'samples.json'); data=dict(np.load(OLD/'dataset.npz'))
    rows=[]; components=[]; reconstruction=[]; attribution=[]; camera_rows=[]; horizon={}; counts=[]
    def append(pred,depth,idx,seed,mode,method):
        for j,i in enumerate(idx):
            m=meta[i]; height=float((data['boxes'][i,P-1,3]-data['boxes'][i,P-1,1])/SCALE[1])
            turn=float(np.degrees(np.linalg.norm(data['cam'][i,-1])))
            rows.append(dict(index=int(i),key=m['key'],recording=m['recording'],clip=m['clip'],track=m['clip']+'_'+m['id'],
                             seed=seed,mode=mode,method=method,nonoverlap=m['nonoverlap'],edge=m['edge'],box_height_fraction=height,final_turn_degrees=turn,
                             metrics=metrics(pred[j],data['xy'][i,P:],None if depth is None else depth[j])))
        if mode=='reviewed':
            err=np.linalg.norm(pred-data['xy'][idx,P:],axis=-1)
            bad=~np.isfinite(err)
            if depth is not None:bad|=depth<=1e-6
            err[bad]=32000.;horizon.setdefault(method,[]).extend(err.tolist())
    for fold in read(OUT/'folds.json'):
        f=fold['fold']; idx=np.array(fold['test_indices'])
        truth=warp(data['xy'][idx,P:],data['H'][idx,P:])[0]
        for seed in cfg['seeds']:
            cam=cameras(f,seed)
            for arm in ['old','full']:
                angles=angle(cam[arm][idx],data['cam'][idx,P:])
                camera_only,depth=project(truth,cam[arm],data,idx)
                for j,i in enumerate(idx):
                    camera_rows.append(dict(key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,arm=arm,
                                            angular_mean=float(angles[j].mean()),angular_final=float(angles[j,-1]),
                                            reference_pedestrian_image_ADE=metrics(camera_only[j],data['xy'][i,P:],depth[j])['ADE15']))
            for mode in cfg['evaluation_inputs']:
                old=np.load(OLD/f'fold_{f}/seed_{seed}/comp_dual/predictions_{mode}.npz');assert np.array_equal(old['indices'],idx)
                append(old['pred'],old['depth'],idx,seed,mode,'old_original')
                if mode=='reviewed':
                    sync=np.load(ROOT/f'outputs/camera_sync_prediction_20261002/fold_{f}/seed_{seed}/comp_dual/predictions.npz')
                    assert np.array_equal(sync['indices'],idx)
                    append(sync['new_new'],sync['new_new_depth'],idx,seed,mode,'old_sync_full')
                # Same input corruption as the new models for non-reviewed references.
                history=input_history(data,idx,mode,np.random.default_rng(702))
                append(np.repeat(history['input_xy'][:,-1,None],F,axis=1),None,idx,seed,mode,'hold')
                for variant in VARIANTS:
                    dest=OUT/f'fold_{f}/seed_{seed}/{variant}'
                    counts.append(read(dest/'complete.json')['total_parameters'])
                    z=np.load(dest/f'predictions_{mode}.npz');assert np.array_equal(z['indices'],idx)
                    assert np.isfinite(z['component']).all()
                    for arm in ['full','old','frozen','reference']:
                        append(z[arm],z[arm+'_depth'],idx,seed,mode,variant+'/'+arm)
                    for j,i in enumerate(idx):
                        comp_error=np.linalg.norm(z['component'][j]-truth[j],axis=-1)
                        components.append(dict(key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,mode=mode,variant=variant,
                                               ADE=float(comp_error.mean()),FDE=float(comp_error[-1])))
                        if 'reconstruction' in z.files:
                            error=np.linalg.norm(z['reconstruction'][j]-data['xy'][i,:P],axis=-1)
                            input_error=np.linalg.norm(z['input_xy'][j]-data['xy'][i,:P],axis=-1)
                            valid=z['input_valid'][j].astype(bool)
                            reconstruction.append(dict(key=meta[i]['key'],seed=seed,mode=mode,variant=variant,all_point_MAE=float(error.mean()),
                                visible_input_MAE=float(input_error[valid].mean()),visible_reconstruction_MAE=float(error[valid].mean())))
                    if mode=='reviewed':
                        # Exact image error decomposition with this pedestrian output held fixed.
                        a=z['reference']-data['xy'][idx,P:]; b=z['full']-z['reference']; e=z['full']-data['xy'][idx,P:]
                        assert np.allclose(a+b,e,atol=1e-7)
                        for j,i in enumerate(idx):
                            attribution.append(dict(key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,variant=variant,
                                person_MSE=float(np.sum(a[j]**2,-1).mean()),camera_MSE=float(np.sum(b[j]**2,-1).mean()),
                                cross_term=float((2*np.sum(a[j]*b[j],-1)).mean()),total_MSE=float(np.sum(e[j]**2,-1).mean())))
    write(OUT/'metrics.json',rows);write(OUT/'component_metrics.json',components);write(OUT/'history_metrics.json',reconstruction)
    write(OUT/'camera_metrics.json',camera_rows);write(OUT/'attribution.json',attribution)
    reviewed=[r for r in rows if r['mode']=='reviewed'];methods=sorted({r['method'] for r in reviewed})
    summary={m:reduce_rows([r for r in reviewed if r['method']==m]) for m in methods}
    expected=read(OLD/'summary.json')['reviewed']
    assert abs(summary['old_original']['window_mean']['ADE15']-expected['comp_dual']['window_mean']['ADE15'])<1e-6
    assert abs(summary['hold']['window_mean']['ADE15']-expected['hold']['window_mean']['ADE15'])<1e-6
    primary='separated_dual/full'
    def keyed(method):
        return {m['key']:float(np.mean([r['metrics']['ADE15'] for r in reviewed if r['method']==method and r['key']==m['key']])) for m in meta}
    a=keyed(primary);pairs={}
    for base in ['old_original','old_sync_full','coupled_dual/full','separated_future/full','hold']:
        b=keyed(base); delta=np.array([a[m['key']]-b[m['key']] for m in meta])
        recdelta=np.array([summary[primary]['per_recording'][g]['ADE15']-summary[base]['per_recording'][g]['ADE15'] for g in summary[primary]['per_recording']])
        rng=np.random.default_rng(20261002);bootstrap=rng.choice(recdelta,(5000,len(recdelta)),replace=True).mean(1)
        pairs[base]=dict(improved=int((delta<0).sum()),windows=len(delta),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
                        recordings_improved=int((recdelta<0).sum()),recordings=len(recdelta),
                        recording_equal_delta_ci95=np.percentile(bootstrap,[2.5,97.5]).tolist(),
                        bootstrap_note='Resample six recordings, after averaging windows/seeds within recording; descriptive, not 351 independent trials.')
    subsets={}
    filters={'nonoverlap':lambda r:r['nonoverlap'],'edge':lambda r:r['edge'],'no_edge':lambda r:not r['edge'],
             'box_small_lt_012':lambda r:r['box_height_fraction']<.12,'box_medium_012_025':lambda r:.12<=r['box_height_fraction']<.25,'box_large_ge_025':lambda r:r['box_height_fraction']>=.25,
             'turn_lt_5deg':lambda r:r['final_turn_degrees']<5,'turn_5_15deg':lambda r:5<=r['final_turn_degrees']<15,'turn_ge_15deg':lambda r:r['final_turn_degrees']>=15}
    mains=['old_original','old_sync_full','hold','separated_dual/full','coupled_dual/full','separated_future/full']
    for label,predicate in filters.items():
        subsets[label]={m:reduce_rows([r for r in reviewed if r['method']==m and predicate(r)]) for m in mains}
    other={mode:{m:reduce_rows([r for r in rows if r['mode']==mode and r['method']==m]) for m in mains if m!='old_sync_full'} for mode in ['pre_review','controlled_noise']}
    comp={v:{mode:float(np.mean([r['ADE'] for r in components if r['variant']==v and r['mode']==mode])) for mode in cfg['evaluation_inputs']} for v in VARIANTS}
    hist={v:{mode:{k:float(np.mean([r[k] for r in reconstruction if r['variant']==v and r['mode']==mode])) for k in ['all_point_MAE','visible_input_MAE','visible_reconstruction_MAE']} for mode in cfg['evaluation_inputs']} for v in VARIANTS if v.endswith('dual')}
    attr={v:{k:float(np.mean([r[k] for r in attribution if r['variant']==v])) for k in ['person_MSE','camera_MSE','cross_term','total_MSE']} for v in VARIANTS}
    for v,s in attr.items(): assert abs(s['person_MSE']+s['camera_MSE']+s['cross_term']-s['total_MSE'])<1e-5
    camera_summary={a:{k:float(np.mean([r[k] for r in camera_rows if r['arm']==a])) for k in ['angular_mean','angular_final','reference_pedestrian_image_ADE']} for a in ['old','full']}
    write(OUT/'summary.json',dict(main=summary,paired_vs_primary=pairs,subsets=subsets,other_inputs=other,component_ADE=comp,history=hist,
                                error_attribution=attr,camera=camera_summary,primary=primary))
    write(OUT/'horizon_errors.json',{k:np.asarray(v).mean(0).tolist() for k,v in horizon.items()})
    print({m:{'ADE':summary[m]['window_mean']['ADE15'],'FDE':summary[m]['window_mean']['FDE15'],'macro':summary[m]['recording_equal']['ADE15']} for m in mains})
    print('paired',pairs);print('components',comp);print('camera',camera_summary)


if __name__=='__main__':main()
