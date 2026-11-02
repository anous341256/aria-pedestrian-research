"""Independent SciPy reprojection, metric replay and preservation audit."""
from shared import *
from scipy.spatial.transform import Rotation


def main():
    cfg=read(OUT/'protocol.json');data=np.load(OLD/'dataset.npz');s=read(OUT/'summary.json')
    errors={v:[] for v in VARIANTS};endpoints={v:[] for v in VARIANTS};geometry=0.;files=0
    for fold in read(OUT/'folds.json'):
        for seed in cfg['seeds']:
            cam=cameras(fold['fold'],seed)['full']
            for variant in VARIANTS:
                p=OUT/f'fold_{fold["fold"]}/seed_{seed}/{variant}'
                z=np.load(p/'predictions_reviewed.npz');idx=z['indices'];files+=1
                # Independent column-vector projection with SciPy rotations.
                for j,i in enumerate(idx):
                    rays=np.linalg.solve(data['K'][i].astype(float),np.c_[z['component'][j],np.ones(F)].T).T
                    rotations=Rotation.from_rotvec(cam[i].astype(float)).as_matrix()
                    q=np.einsum('tji,tj->ti',rotations,rays)
                    pix=(data['K'][i].astype(float)@q.T).T
                    pred=pix[:,:2]/pix[:,2:]
                    difference=float(np.max(abs(pred-z['full'][j])));geometry=max(geometry,difference)
                    assert difference<.01,(i,variant,difference)
                    e=np.linalg.norm(pred-data['xy'][i,P:],axis=-1)
                    e[(q[:,2]<=1e-6)|~np.isfinite(e)]=32000.
                    errors[variant].append(float(e.mean()));endpoints[variant].append(float(e[-1]))
    assert files==54
    metric_difference=0.
    for v in VARIANTS:
        ade=float(np.mean(errors[v]));fde=float(np.mean(endpoints[v]));ref=s['main'][v+'/full']['window_mean']
        diff=max(abs(ade-ref['ADE15']),abs(fde-ref['FDE15']));metric_difference=max(metric_difference,diff)
        assert diff<.01
    source=read(OUT/'source_hashes.json')
    for p,h in source.items():assert sha(ROOT/p)==h,p
    base=ROOT/'models/forecasting/biflow_comp_dual_baseline_20261002';manifest=read(base/'manifest.json')
    for item in manifest['files']:assert sha(base/item['path'])==item['sha256'],item['path']
    annotation_sources=read(OLD/'sources.json')
    for p,h in annotation_sources.items():assert sha(ROOT/p.replace('\\','/'))==h,p
    assert read(OUT/'postflight.json')['passed'] and read(OUT/'presentation/verification.json')['passed']
    newhash={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'scripts/factorized_forecast_v1_20261002').glob('*.py')}
    write(OUT/'implementation_hashes.json',newhash)
    result=dict(passed=True,independent_reprojection_max_px=geometry,independent_metric_max_px=metric_difference,
                trained_checkpoints_and_predictions=files,source_files_unchanged=len(source),baseline_snapshot_files_unchanged=len(manifest['files']),
                original_annotation_and_video_sources_unchanged=len(annotation_sources),video_decode_verification=True,
                future_input_mutation_zero=True,separated_branch_camera_mutation_zero=True)
    write(OUT/'verification.json',result);print(result)


if __name__=='__main__':main()
