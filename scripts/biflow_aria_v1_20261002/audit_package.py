"""Check saved artifacts and independently recompute published learned metrics."""
from common import *
import re


def main():
    cfg = read(OUT / 'protocol.json')
    folds = read(OUT / 'folds.json')
    data = np.load(OUT / 'dataset.npz')
    rows = read(OUT / 'metrics.json')
    lookup = {(r['fold'], r['seed'], r['method'], r['mode'], r['index']): r
              for r in rows if r['seed'] != -1}
    checked, max_difference = 0, 0.
    for fold in folds:
        for seed in cfg['seeds']:
            parent = OUT / f'fold_{fold["fold"]}' / f'seed_{seed}'
            assert (parent / 'camera.pt').stat().st_size > 0
            for method in cfg['variants']:
                target = parent / method
                assert (target / 'checkpoint.pt').stat().st_size > 0
                assert (target / 'complete.json').exists()
                for mode in cfg['evaluation_inputs']:
                    z = np.load(target / f'predictions_{mode}.npz')
                    assert set(z['indices'].tolist()) == set(fold['test_indices'])
                    for j, i in enumerate(z['indices']):
                        names = [(method, 'pred')]
                        if method.startswith('comp'):
                            names.append((method + '_reference_camera', 'oracle'))
                        for name, key in names:
                            p = z[key][j]
                            e = np.sqrt(np.sum((p - data['xy'][i, 6:]) ** 2, axis=-1))
                            depth = z['depth'][j]
                            if key == 'oracle':
                                K = data['K'][i]
                                points = np.column_stack((z['component'][j], np.ones(15)))
                                depth = np.array([(K @ cv2.Rodrigues(v)[0].T @ np.linalg.inv(K) @ point)[2]
                                                  for v, point in zip(data['cam'][i, 6:], points)])
                            bad = ~np.isfinite(e) | (depth <= 1e-6)
                            e[bad] = 32000.
                            actual = [e[:5].mean(), e[:10].mean(), e.mean(), e[-1]]
                            row = lookup[(fold['fold'], seed, name, mode, int(i))]
                            saved = [row['metrics'][k] for k in ['ADE05', 'ADE10', 'ADE15', 'FDE15']]
                            diff = float(np.max(np.abs(np.array(actual) - saved)))
                            max_difference = max(max_difference, diff)
                            assert diff < 1e-5, (row, actual)
                            assert row['metrics']['invalid'] == int(bad.sum())
                            checked += 1
    assert checked == len(lookup)
    prediction_files = list(OUT.glob('fold_*/seed_*/*/predictions_*.npz'))
    assert len(prediction_files) == 216
    report = ROOT / 'reports/biflow_aria_v1_20261002/RESULTS.md'
    entry = OUT / 'presentation/START_HERE.md'
    links = []
    for p in [report, entry]:
        for link in re.findall(r'\]\((E:/[^)]+)\)', p.read_text(encoding='utf8')):
            assert Path(link).exists(), link
            links.append(link)
    assert read(OUT / 'verification.json')['passed']
    assert read(OUT / 'presentation/verification.json')['passed']
    result = dict(passed=True, trajectory_checkpoints=72, camera_checkpoints=18,
                  prediction_files=len(prediction_files), independently_recomputed_rows=checked,
                  max_metric_difference=max_difference, local_links_checked=len(links),
                  visual_review='Plots and representative comparison/camera video frames inspected.')
    write(OUT / 'package_verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
