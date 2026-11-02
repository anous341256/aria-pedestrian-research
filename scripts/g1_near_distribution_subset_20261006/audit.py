"""Independent selection/metric/geometry checks and portable per-fold data exports."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_near_distribution_subset_20261006'
BANK=ROOT/'outputs/transformer_matched97_20261006/new'
VARIANTS=['A_early_fusion','B_person_only','C_separated','D_bidirectional','E_oneway','F_interacting_geometry']
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selection_audit():
    for rel,h in read(OUT/'selection_hashes.json').items():assert sha(OUT/rel)==h,rel
    meta=read(BANK/'data/samples.json');data=dict(np.load(BANK/'data/dataset.npz'));original=read(BANK/'folds.json');locked=read(OUT/'selection_locked.json');design=read(OUT/'design_locked.json')
    survey=read(ROOT/'outputs/g1_population_survey_20261006/measured_features.json');bykey={r['key']:r for r in survey}
    desc=read(ROOT/'outputs/g1_edgc_training_distribution_20261006/window_distribution.json');thresholds=read(ROOT/'outputs/g1_population_survey_20261006/protocol_locked.json')['thresholds']
    checks=[]
    for arm in ['matched','random']:
        for fold,o,entry in zip(read(OUT/arm/'folds.json'),original,locked['folds']):
            ids=fold['train_indices'];n=len(ids);assert ids==sorted(set(ids));assert set(ids)<=set(o['train_indices'])&set(design['eligible'])
            assert fold['test_indices']==o['test_indices'] and fold['val_indices']==o['val_indices']
            assert n==int(.7*len(fold['eligible_train_indices']))
            rr=[bykey[k] for k in entry['target_keys']];assert all(r['near'] and r['recording'] not in [o['test'],o['validation']] for r in rr)
            expectedkeys=sorted(r['key'] for r in survey if r['near'] and r['recording'] not in [o['test'],o['validation']]);assert sorted(entry['target_keys'])==expectedkeys
            w=np.array([r['survey_weight'] for r in rr]);w/=sum(w);features=np.array([[desc[i]['features'][k] for k in 'EDGC'] for i in ids]);ref=np.array([[r['features'][k] for k in 'EDGC'] for r in rr])
            # Reconstruct target and selected moments without importing selection code.
            values=[];targets=[]
            for label in entry['diagnostics']['labels']:
                if '_cdf_' in label:
                    k,c=label.split('_cdf_');j='EDGC'.index(k);c=float(c);a=features[:,j]<=c;b=ref[:,j]<=c
                    # Text labels have rounded thresholds; use tight equality guard at boundaries.
                    a=(features[:,j]<c)|np.isclose(features[:,j],c,rtol=1e-11,atol=1e-13)
                    b=(ref[:,j]<c)|np.isclose(ref[:,j],c,rtol=1e-11,atol=1e-13)
                elif label.startswith('C_speed_bin_'):
                    category=int(label.rsplit('_',1)[1]);a=np.digitize(features[:,3],[.2,1,1.6])==category;b=np.digitize(ref[:,3],[.2,1,1.6])==category
                else:
                    bits=np.array(list(map(int,label[5:])));ts=np.array([thresholds[k] for k in 'EDGC']);a=np.all((features>=ts)==bits,axis=1);b=np.all((ref>=ts)==bits,axis=1)
                values.append(float(np.mean(a)));targets.append(float(w@b))
            assert np.max(np.abs(np.array(values)-entry['diagnostics'][arm]))<1e-9
            assert np.max(np.abs(np.array(targets)-entry['diagnostics']['target']))<1e-9
            export=OUT/arm/'subsets'/f'fold_{fold["fold"]}';export.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(export/'train.npz',**{k:v[ids] for k,v in data.items()})
            write(export/'samples.json',[dict(meta[i],original_index=i) for i in ids])
            write(export/'provenance.json',dict(source=str(BANK/'data/dataset.npz'),source_sha256=sha(BANK/'data/dataset.npz'),indices=ids,test_recording=o['test'],validation_recording=o['validation'],selection_sha256=sha(OUT/'selection_locked.json')))
            checks.append(dict(arm=arm,fold=fold['fold'],windows=n,target_windows=len(rr),recordings=len({meta[i]['recording'] for i in ids}),moments_replayed=True))
    write(OUT/'selection_audit.json',dict(passed=True,checks=checks,source_copy_equal=True,locked_hashes=True,target_recordings_excluded=True))
def final_audit():
    selection_audit();data=dict(np.load(BANK/'data/dataset.npz'));checks=[];maxgeo=0.;maxmetric=0.
    for arm in ['matched','random']:
        assert read(OUT/arm/'verification.json')['passed'];cfg=read(OUT/arm/'protocol.json');rows=read(OUT/arm/'metrics.json')
        bykey={(r['index'],r['seed'],r['mode'],r['method']):r for r in rows}
        for fold in read(OUT/arm/'folds.json'):
            f=fold['fold'];idx=np.array(fold['test_indices']);K=data['K'][idx]
            for seed in cfg['seeds']:
                rv=np.load(BANK/f'camera/fold_{f}/seed_{seed}/full/predictions.npz')['person_rotation'][idx]
                rotations=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(len(idx),15,3,3)
                H=K[:,None]@np.swapaxes(rotations,-1,-2)@np.linalg.inv(K)[:,None]
                for v in VARIANTS:
                    for mode in cfg['evaluation_inputs']:
                        z=np.load(OUT/arm/f'fold_{f}/seed_{seed}/{v}/predictions_{mode}.npz');assert np.array_equal(z['indices'],idx)
                        if v[0] in 'CF':
                            p=np.einsum('ntij,ntj->nti',H,np.concatenate([z['component'],np.ones((len(idx),15,1))],-1));xy=p[...,:2]/p[...,2:]
                            delta=float(np.max(np.abs(xy-z['pred'])));maxgeo=max(maxgeo,delta);assert delta<.002,(arm,f,seed,v,delta)
                        errors=np.linalg.norm(z['pred']-data['xy'][idx,6:],axis=-1);errors[(~np.isfinite(errors))|(z['depth']<=1e-6)]=32000
                        for j,i in enumerate(idx):
                            m=bykey[(int(i),seed,mode,v)]['metrics'];diff=max(abs(float(errors[j].mean())-m['ADE15']),abs(float(errors[j,-1])-m['FDE15']));maxmetric=max(maxmetric,diff);assert diff<1e-9
                    checks.append(dict(arm=arm,fold=f,seed=seed,variant=v))
    protected=read(OUT/'protected_previous_hashes.json');assert all(sha(ROOT/p)==h for p,h in protected.items())
    assert len(checks)==396
    write(OUT/'audit.json',dict(passed=True,models=396,independent_metric_max_px=maxmetric,independent_scipy_geometry_max_px=maxgeo,protected_files=len(protected),all_protected_unchanged=True,selection_locked_and_replayed=True))
    print('Independent audit passed',maxmetric,maxgeo,len(protected),flush=True)
if __name__=='__main__':
    import sys
    if '--selection-only' in sys.argv:selection_audit()
    else:final_audit()
