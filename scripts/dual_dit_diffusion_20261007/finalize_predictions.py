"""Fix scene-shared camera randomness before examining any forecast metrics."""
import shutil
import numpy as np
import experiment
from experiment import OUT,OLD,read,write,sha,ROOT
from sampling import infer
from audit_report import load

assert (OUT/'training_complete.json').exists()
experiment.infer=infer
data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');cfg=read(OUT/'protocol.json')
for fold in read(OUT/'folds.json'):
    f=fold['fold'];path=OUT/f'fold_{f}/predictions.npz';archive=path.with_name('predictions_initial_unshared_camera.npz')
    if not archive.exists():shutil.copy2(path,archive)
    before=dict(np.load(archive))
    experiment.predict(data,meta,fold,load('camera',f),load('person',f),cfg)
    after=dict(np.load(path))
    assert all(np.array_equal(before[k],after[k]) for k in before if not k.startswith('ddim1'))
    assert np.array_equal(before['ddim1_person_samples'],after['ddim1_person_samples'])
    h=data['cam'][after['indices'],:6];pairs=[(i,j) for i in range(len(h)) for j in range(i) if np.array_equal(h[i],h[j])]
    assert all(np.array_equal(after['ddim1_camera_samples'][i],after['ddim1_camera_samples'][j]) for i,j in pairs)
    print('Finalized shared camera sampling fold',f,flush=True)
write(OUT/'sampling_correction.json',dict(reason='Same observed camera history must share one future-camera sample set across all people.',
    changed='Only stochastic camera innovations now use a history-keyed seed and duplicate-history reuse.',
    unchanged='All checkpoints, deterministic predictions, stochastic person draws, protocol and hyperparameters.',
    timing='Correction based on structural identity check before reading forecast errors.',
    initial_outputs='predictions_initial_unshared_camera.npz retained per fold',
    code_sha256=sha(ROOT/'scripts/dual_dit_diffusion_20261007/sampling.py')))
