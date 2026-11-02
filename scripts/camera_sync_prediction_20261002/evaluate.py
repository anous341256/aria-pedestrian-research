from shared import *
import json


def reduce_rows(rr):
    keys=['ADE05','ADE10','ADE15','FDE15']
    def mean(group):return {k:float(np.mean([r['metrics'][k] for r in group])) for k in keys}
    per_recording={g:mean([r for r in rr if r['recording']==g]) for g in sorted({r['recording'] for r in rr})}
    per_seed={str(g):mean([r for r in rr if r['seed']==g]) for g in sorted({r['seed'] for r in rr})}
    return dict(windows=len({r['key'] for r in rr}),window_mean=mean(rr),recording_equal={k:float(np.mean([v[k] for v in per_recording.values()])) for k in keys},
                per_recording=per_recording,per_seed=per_seed,invalid_steps=sum(r['metrics']['invalid'] for r in rr))


def main():
    assert read(OUT/'verification.json')['passed']
    rows=read(OUT/'metrics.json');data=np.load(OLD/'dataset.npz');meta=read(OLD/'samples.json')
    full={};paired={};condition={};camera=read(CAMERA/'fixed_person_reprojection.json')['summary']
    for variant in VARIANTS:
        full[variant]={arm:reduce_rows([r for r in rows if r['variant']==variant and r['arm']==arm]) for arm in ARMS}
        expected=read(OLD/'summary.json')['reviewed'][variant]['window_mean']['ADE15']
        assert abs(full[variant]['old_old']['window_mean']['ADE15']-expected)<1e-6
        assert abs(full[variant]['old_new']['window_mean']['ADE15']-camera[variant]['full']['window_ADE'])<1e-4
        values={arm:{m['key']:np.mean([r['metrics']['ADE15'] for r in rows if r['variant']==variant and r['arm']==arm and r['key']==m['key']]) for m in meta} for arm in ARMS}
        paired[variant]={}
        for baseline in ['old_old','old_new']:
            delta=np.array([values['new_new'][m['key']]-values[baseline][m['key']] for m in meta])
            paired[variant][f'new_new_vs_{baseline}']=dict(improved=int(np.sum(delta<0)),windows=len(delta),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)))
        changes=[];oracleold=[];oraclenew=[]
        for file in OUT.glob(f'fold_*/seed_*/{variant}/predictions.npz'):
            z=np.load(file);truth=data['xy'][z['indices'],6:]
            changes.extend(np.linalg.norm(z['new_component']-z['old_component'],axis=-1).mean(1).tolist())
            oracleold.extend(np.linalg.norm(z['old_reference_camera']-truth,axis=-1).mean(1).tolist())
            oraclenew.extend(np.linalg.norm(z['new_reference_camera']-truth,axis=-1).mean(1).tolist())
        condition[variant]=dict(mean_component_displacement_px=float(np.mean(changes)),median_window_component_displacement_px=float(np.median(changes)),
                                old_condition_reference_camera_ADE=float(np.mean(oracleold)),new_condition_reference_camera_ADE=float(np.mean(oraclenew)))
    subsets={}
    for label,accept in [('nonoverlap',lambda r:r['nonoverlap']),('edge',lambda r:r['edge']),('no_edge',lambda r:not r['edge'])]:
        subsets[label]={v:{arm:reduce_rows([r for r in rows if r['variant']==v and r['arm']==arm and accept(r)]) for arm in ARMS} for v in VARIANTS}
    write(OUT/'summary.json',dict(main=full,paired=paired,condition_effect=condition,subsets=subsets,
                                hold_baseline=read(OLD/'summary.json')['reviewed']['hold']))
    print(json.dumps(dict(main={v:{a:dict(ADE=d['window_mean']['ADE15'],FDE=d['window_mean']['FDE15'],macro=d['recording_equal']['ADE15']) for a,d in b.items()} for v,b in full.items()},paired=paired,condition_effect=condition)))


if __name__=='__main__':main()
