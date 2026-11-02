"""Fix protocol and source provenance before training."""
from shared import *


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve fixed protocol'
    cfg = read(OLD / 'protocol.json')
    cfg.update(name='Independent camera and residual pedestrian forecast V1', variants=VARIANTS,
               coordinates='Rotation-compensated 2D residual; translation/parallax not separated. No validated dynamic-person depth in prepared data.',
               model='Independent GRU pedestrian encoder, history/future flow heads, fixed final reprojection. Camera is frozen full-data held-out GRU.',
               features='Stabilized relative xy (2), stabilized adjacent displacement (2), valid (1), time (1); origin (2). No raw box width/height.',
               camera='Reuse 18 matching full-data held-out camera models, not all-recording refits. Predictions enter only projection for separated variants.',
               validation='Checkpoint chosen by validation stabilized future pixel ADE, using reference rotation to define targets. Same criterion for all new variants; independent of predicted camera error.',
               capacity='separated context 66->95->64; coupled context 129->64->64; parameter difference 29, about 0.024 percent.',
               arms=['full', 'old_projection_diagnostic', 'frozen_projection_diagnostic', 'reference_camera_diagnostic'],
               comparison='Old frozen baseline retained. Matched retrained coupled control isolates direct camera conditioning at approximately equal capacity and identical inputs/budget/selection. Old-to-new also changes features and selection.',
               source_run=str(OLD.relative_to(ROOT)), camera_source_run=str(CAMERA.relative_to(ROOT)),
               primary='separated_dual with full-data held-out predicted camera; no best-test-method selection',
               visualization='Original nine fixed windows, seed 17. Additional gain/loss extrema explicitly post-score, with no parameter tuning.',
               training_approved='User explicitly approved implementing and evaluating the draft on 2026-10-02.')
    write(OUT / 'protocol.json', cfg)
    write(OUT / 'folds.json', read(OLD / 'folds.json'))
    paths = [OLD / f for f in ['dataset.npz', 'samples.json', 'folds.json', 'protocol.json', 'summary.json']]
    paths += [CAMERA / f for f in ['folds.json', 'protocol.json', 'verification.json']]
    paths += list(LEGACY_CODE.glob('*.py'))
    for fold in read(OUT / 'folds.json'):
        f = fold['fold']
        for seed in cfg['seeds']:
            paths += [CAMERA / f'fold_{f}/seed_{seed}/full/{name}' for name in ['camera.pt', 'predictions.npz']]
            paths += [OLD / f'fold_{f}/seed_{seed}/{name}' for name in ['camera.pt', 'camera_predictions.npy']]
            paths += [OLD / f'fold_{f}/seed_{seed}/comp_dual/{name}' for name in ['checkpoint.pt', 'predictions_reviewed.npz']]
    write(OUT / 'source_hashes.json', {p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    write(OUT / 'data_inventory.json', dict(prepared_arrays={k:list(v.shape) for k,v in dict(np.load(OLD/'dataset.npz')).items()},
                                            dynamic_person_depth_available=False, implementation='rotation-only branch of approved draft'))
    print('Protocol fixed; sources', len(paths), 'variants', VARIANTS)


if __name__ == '__main__': main()
