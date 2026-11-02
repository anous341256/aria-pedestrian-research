"""Replay both decoder heads, preserving all original weights and outputs."""
from shared import *
from network import PedestrianPredictor, torch
from train import noise_for, tensor, DEVICE

DEST = OUT / 'presentation/head_outputs_20261003'
CASE = '10_N01_020'
SEED = 17


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    meta = read(OLD / 'samples.json')
    index = next(i for i, m in enumerate(meta) if m['key'] == CASE)
    fold = next(f for f in read(OUT / 'folds.json') if index in f['test_indices'])
    folder = OUT / f'fold_{fold["fold"]}/seed_{SEED}/separated_dual'
    cp = folder / 'checkpoint.pt'
    source = folder / 'predictions_reviewed.npz'
    cp_hash, source_hash = sha(cp), sha(source)
    cfg = read(OUT / 'protocol.json')
    data = dict(np.load(OLD / 'dataset.npz'))
    saved = np.load(source)
    # Replay the complete held-out batch to retain the original CUDA GEMM shape.
    idx = saved['indices']
    local = list(idx).index(index)
    model = PedestrianPredictor(coupled=False).to(DEVICE)
    model.load_state_dict(torch.load(cp, map_location=DEVICE, weights_only=False)['model'])
    model.eval()
    h = input_history(data, idx, 'reviewed', np.random.default_rng(702))
    args = {k: tensor(h[k]) for k in ['seq', 'origin']}
    arrays = dict(input_sequence=h['seq'][local], origin=h['origin'][local])
    errors = {}
    with torch.no_grad():
        arrays['latent'] = model.encode(**args).cpu().numpy()[local]
        for name, length, saved_key in [('history', P, 'reconstruction_component'), ('future', F, 'component')]:
            noise = noise_for(meta, idx, name, cfg['inference_samples'])
            raw = model.sample(**args, noise=tensor(noise), kind=name,
                               steps=cfg['flow_steps']).cpu().numpy()
            raw = raw.reshape(len(idx), cfg['inference_samples'], length, 2)
            pixel = (raw.mean(1) + h['origin'][:, None]) * SCALE
            errors[name] = float(np.max(abs(pixel - saved[saved_key])))
            assert errors[name] < .001, errors
            arrays[name + '_noise'] = noise[local]
            arrays[name + '_samples_normalized'] = raw[local]
            arrays[name + '_mean_normalized'] = raw.mean(1)[local]
            arrays[name + '_pixel'] = pixel[local]
    camera_path = CAMERA / f'fold_{fold["fold"]}/seed_{SEED}/full/predictions.npz'
    camera = np.load(camera_path)['person_rotation']
    final, depth = project(arrays['future_pixel'][None], camera, data, [index])
    errors['composition'] = float(np.max(abs(final[0] - saved['full'][local])))
    assert errors['composition'] < .001 and (depth > 0).all()
    arrays.update(final_pixel=final[0], camera_prediction=camera[index],
                  xy=data['xy'][index], seconds=data['seconds'][index],
                  H_reference=data['H'][index], K=data['K'][index],
                  stable_reference=warp(data['xy'][index], data['H'][index])[0])
    np.savez_compressed(DEST / 'head_outputs.npz', **arrays)
    # These numerical exports are model outputs; reference arrays are separated.
    write(DEST / 'head_outputs.json', dict(case=CASE, seed=SEED, fold=fold['fold'],
          historical_head=dict(coordinates='rotation-normalized pixels, orientation at t',
                               times=arrays['seconds'][:P].tolist(), xy=arrays['history_pixel'].tolist()),
          future_head=dict(coordinates='rotation-normalized pixels, orientation at t',
                           times=arrays['seconds'][P:].tolist(), xy=arrays['future_pixel'].tolist()),
          camera_head=dict(coordinates='rotation vector, radians; future camera to t',
                           rotation_vectors=arrays['camera_prediction'].tolist()),
          final_composition=dict(coordinates='future image pixels at each corresponding time',
                                 xy=arrays['final_pixel'].tolist()),
          note='The history head is auxiliary. Only the future head and predicted camera rotation enter final composition.'))
    assert sha(cp) == cp_hash and sha(source) == source_hash
    write(DEST / 'replay_verification.json', dict(passed=True, device=DEVICE,
          max_error_vs_saved_px=errors, checkpoint_sha256=cp_hash,
          predictions_sha256=source_hash, camera_predictions_sha256=sha(camera_path),
          metadata=meta[index], history_shape=list(arrays['history_pixel'].shape),
          future_shape=list(arrays['future_pixel'].shape),
          per_head_sampling='Mean of the same five Gaussian starts, ten flow steps.',
          future_reference_used_for_inference=False))
    print(json.dumps(dict(destination=str(DEST), replay_max_error_px=errors)))


if __name__ == '__main__':
    import json
    main()
