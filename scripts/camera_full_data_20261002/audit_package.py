"""Independent rotation metrics and final preservation/coverage checks."""
from common import *
from scipy.spatial.transform import Rotation
import re


def local_path(p):
    if p.startswith('/mnt/'):
        return Path(p[5].upper()+':/'+p[7:])
    return ROOT / p.replace('\\','/')


def main():
    assert read(OUT/'training_complete.json')['passed']
    original=read(OUT/'verification.json')
    assert sha(OUT/'dataset.npz')==original['dataset_sha256']
    assert sha(OUT/'protocol.json')==original['protocol_sha256']
    full=np.load(OUT/'dataset.npz');p=np.load(OUT/'held_out_predictions.npz');e=np.load(OUT/'held_out_errors.npz')
    scores=read(OUT/'camera_metrics.json')['full_recording_test'];maxdiff=0.
    for name in scores:
        pred=p[name];target=np.broadcast_to(p['target'][:,None],pred.shape)
        delta=Rotation.from_rotvec(pred.astype(float).reshape(-1,3)).inv()*Rotation.from_rotvec(target.astype(float).reshape(-1,3))
        errors=np.rad2deg(delta.magnitude()).reshape(pred.shape[:-1])
        diff=float(np.max(abs(errors-e[name])));assert diff<1e-8,(name,diff);maxdiff=max(maxdiff,diff)
        rec=full['record_index'][p['indices']]
        macro=float(np.mean([errors[rec==g].mean() for g in np.unique(rec)]))
        assert abs(macro-scores[name]['recording_equal_mean_degrees'])<1e-8
    hashes=read(OUT/'source_hashes.json')
    for path,h in hashes.items():assert sha(local_path(path))==h,path
    completed=list(OUT.glob('fold_*/seed_*/*/complete.json'));assert len(completed)==36
    for path in completed:
        c=read(path);assert c['unique_train_windows_seen']==c['training_windows']
        assert c['updates']==2000 and c['protocol_sha256']==original['protocol_sha256']
        assert c['train_code_sha256']==sha(Path(__file__).with_name('train.py'))
        assert (path.parent/'camera.pt').stat().st_size>0
        assert (path.parent/'predictions.npz').exists()
    refits=list(OUT.glob('all_recordings_refit/seed_*/complete.json'));assert len(refits)==3
    for path in refits:
        c=read(path);assert c['training_windows']==len(full['rotation'])==c['unique_train_windows_seen']
        assert c['validation_windows']==0 and c['best_validation_mse'] is None
    report=ROOT/'reports/camera_full_data_20261002/RESULTS.md'
    links=re.findall(r'\]\((E:/[^)]+)\)',report.read_text(encoding='utf8'))
    for path in links:assert Path(path).exists(),path
    result=dict(passed=True,independent_SO3_metric_max_difference_degrees=maxdiff,
                original_sources_unchanged=len(hashes),cv_models=36,all_recording_refit_models=3,
                all_training_windows_used=True,report_local_links_checked=len(links),
                old_results_and_person_models_preserved=True)
    write(OUT/'package_verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
