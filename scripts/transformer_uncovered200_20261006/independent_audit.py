"""Independent dataset/split/metric/geometry audit after the frozen experiment."""
from pathlib import Path
import json,hashlib,gzip
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/transformer_uncovered200_20261006'
OLD=ROOT/'outputs/transformer_matched97_20261006/new'
ANN=ROOT/'outputs/sam2_g1_uncovered200_20261006'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    arrays=dict(np.load(OUT/'data/dataset.npz'));old=dict(np.load(OLD/'data/dataset.npz'))
    meta=read(OUT/'data/samples.json');oldmeta=read(OLD/'data/samples.json');cfg=read(OUT/'protocol.json')
    assert meta[:296]==oldmeta
    for k in old:assert np.array_equal(arrays[k][:296],old[k]),k
    assert len(meta)==374 and len(set(m['key'] for m in meta))==374
    excluded={'010','012','015','016','017','019','037','038','058','069','070','074','083'}
    assert all(m['source_clip'] not in excluded for m in meta[296:])
    # Rebuild every added sample from source numeric boxes, rotations and exact timestamps.
    cache={};max_center=0.;max_H=0.;max_inverse=0.
    for i,m in enumerate(meta[296:],296):
        cid=m['source_clip']
        if cid not in cache:
            cal=dict(np.load(ANN/cid/'input/reference.npz'))
            numeric={(r['id'],r['frame']):r for r in read(ANN/cid/'review_v2/mask_metrics.json')}
            with gzip.open(ANN/cid/'review_v2/masks_rle.jsonl.gz','rt') as f:keys={(r['id'],r['frame']) for r in map(json.loads,f)}
            donors={(r['id'],r['frame']) for r in read(ANN/cid/'review_v2/audit.json')['donor_dependencies']}
            cache[cid]=(cal,numeric,keys,donors)
        cal,numeric,keys,donors=cache[cid];fs=m['frames']
        assert len(fs)==21 and fs==list(range(fs[0],fs[0]+21))
        assert all((m['id'],f) in keys and (m['id'],f) not in donors for f in fs)
        boxes=np.array([numeric[m['id'],f]['box'] for f in fs]);xy=(boxes[:,:2]+boxes[:,2:])/2
        assert np.array_equal(boxes,arrays['boxes'][i])
        max_center=max(max_center,float(np.max(np.abs(xy-arrays['xy'][i]))))
        assert not np.any(boxes[:,0]<=5) and not np.any(boxes[:,1]<=5) and not np.any(boxes[:,2]>=2555) and not np.any(boxes[:,3]>=1915)
        assert np.min(boxes[:,3]-boxes[:,1])>=.08*1920
        K=cal['K'];rot=cal['R'][fs];rel=np.array([rot[5].T@r for r in rot]);Hs=np.array([K@r@np.linalg.inv(K) for r in rel])
        max_H=max(max_H,float(np.max(np.abs(Hs.astype(np.float32)-arrays['H'][i]))))
        assert np.array_equal(arrays['seconds'][i],((cal['times_ns'][fs]-cal['times_ns'][fs[5]])/1e9).astype(np.float32))
        assert np.allclose(Rotation.from_rotvec(arrays['cam'][i]).as_matrix(),rel,atol=1e-6)
        assert np.array_equal(arrays['raw'][i],arrays['xy'][i,:6]) and np.all(arrays['raw_valid'][i]==1)
        # Reference stabilization followed by inverse rotation must recover original points.
        ph=np.c_[xy,np.ones(21)]
        stable=np.einsum('tij,tj->ti',Hs,ph);stable/=stable[:,2:]
        inv=np.array([K@r.T@np.linalg.inv(K) for r in rel])
        back=np.einsum('tij,tj->ti',inv,stable);back=back[:,:2]/back[:,2:]
        max_inverse=max(max_inverse,float(np.max(np.abs(back-xy))))
    # Independent matrix multiplication changes near-zero terms by rounding.
    assert max_center==0 and max_H<1e-9 and max_inverse<1e-8
    folds=read(OUT/'folds.json');oldfolds=read(OLD/'folds.json')
    for f,o in zip(folds,oldfolds):
        assert f['test_indices']==o['test_indices'] and f['val_indices']==o['val_indices']
        assert [i for i in f['train_indices'] if i<296]==o['train_indices']
        assert all(meta[i]['recording'] not in (f['test'],f['validation']) for i in f['train_indices'])
        assert all(i<199 for i in f['test_indices']+f['val_indices'])
        assert len(f['train_indices'])==len(set(f['train_indices']))
    assert sorted(i for f in folds for i in f['test_indices'])==list(range(199))
    table=read(OUT/'metrics.json');lookup={(r['method'],r['seed'],r['mode'],r['index']):r for r in table}
    assert len(lookup)==len(table)
    rows_checked=0;max_metric=0.;max_projection=0.;checkpoints=0
    for f in folds:
        indices=f['test_indices']
        for seed in cfg['seeds']:
            camera=OUT/f'camera/fold_{f["fold"]}/seed_{seed}/full'
            assert digest(camera/'camera.pt')==digest(OLD/f'camera/fold_{f["fold"]}/seed_{seed}/full/camera.pt')
            rotations=np.load(camera/'predictions.npz')['person_rotation']
            for v in cfg['variants']:
                folder=OUT/f'fold_{f["fold"]}/seed_{seed}/{v}';complete=read(folder/'complete.json');logs=read(folder/'training.json')
                assert len(logs)==12 and [r['epoch'] for r in logs]==list(range(10,121,10))
                selected=min(logs,key=lambda r:r['validation_image_ADE'])
                assert selected['epoch']==complete['epoch'] and selected['validation_image_ADE']==complete['validation_image_ADE']
                assert complete['protocol_sha256']==digest(OUT/'protocol.json');checkpoints+=1
                for mode in cfg['evaluation_inputs']:
                    z=np.load(folder/f'predictions_{mode}.npz');assert np.array_equal(z['indices'],indices)
                    d=np.linalg.norm(z['pred']-arrays['xy'][indices,6:],axis=-1);bad=(~np.isfinite(d))|(z['depth']<=1e-6);d[bad]=32000
                    for j,i in enumerate(indices):
                        expected=[d[j,:5].mean(),d[j,:10].mean(),d[j].mean(),d[j,-1]]
                        stored=lookup[v,seed,mode,i]['metrics']
                        max_metric=max(max_metric,max(abs(float(a)-stored[k]) for a,k in zip(expected,['ADE05','ADE10','ADE15','FDE15'])))
                        assert stored['invalid']==int(sum(bad[j]));rows_checked+=1
                        if v[0] in 'CF':
                            K=arrays['K'][i].astype(float);q=Rotation.from_rotvec(rotations[i].astype(float)).as_matrix()
                            H=np.array([K@r.T@np.linalg.inv(K) for r in q])
                            homogeneous=np.c_[z['component'][j],np.ones(15)]
                            p=np.einsum('tij,tj->ti',H,homogeneous);p=p[:,:2]/p[:,2:]
                            max_projection=max(max_projection,float(np.max(np.abs(p-z['pred'][j]))))
    assert max_metric<1e-5 and max_projection<.002,(max_metric,max_projection)
    locked=read(OUT/'locked_hashes.json')
    for p,h in locked.items():assert digest(ROOT/p)==h,p
    protected=read(OUT/'protected_previous_hashes.json')
    for p,h in protected.items():assert digest(ROOT/p)==h,p
    result=dict(passed=True,prepared_samples=374,added=78,user_exclusions_verified=sorted(excluded),
        exact_original_arrays=True,exact_original_train_lists=True,exact_validation_test=True,
        added_labels_absent_from_test=True,source_center_max_px=max_center,H_max=max_H,reference_roundtrip_max_px=max_inverse,
        checkpoints=checkpoints,metric_rows=rows_checked,max_metric_difference=max_metric,scipy_projection_max_px=max_projection,
        locked_files_verified=len(locked),protected_files_verified=len(protected))
    (OUT/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
    print(result,flush=True)

if __name__=='__main__':main()
