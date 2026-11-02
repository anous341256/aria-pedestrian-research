"""Train separated and matched coupled controls without changing old assets."""
from shared import *
from network import PedestrianPredictor, torch
from train import noise_for, balanced, seed_all, tensor, DEVICE
import time, copy, argparse


def model_inputs(model, history, data, idx, cam):
    result = {k:tensor(history[k]) for k in ['seq', 'origin']}
    if model.coupled: result['camera'] = tensor(camera_condition(data, idx, cam))
    return result


def infer_component(model, data, meta, idx, cam, cfg, mode='reviewed', with_history=False):
    h = input_history(data, idx, mode, np.random.default_rng(702))
    args = model_inputs(model, h, data, idx, cam)
    samples = model.sample(**args, noise=tensor(noise_for(meta, idx, 'future', cfg['inference_samples'])), steps=cfg['flow_steps']).cpu().numpy()
    samples = samples.reshape(len(idx), cfg['inference_samples'], F, 2)
    points = (samples.mean(1) + h['origin'][:, None]) * SCALE
    result = dict(component=points, samples_normalized=samples, input_xy=h['input_xy'], input_valid=h['input_valid'])
    if with_history:
        rec = model.sample(**args, noise=tensor(noise_for(meta, idx, 'history', cfg['inference_samples'])), kind='history', steps=cfg['flow_steps']).cpu().numpy().reshape(len(idx), cfg['inference_samples'], P, 2).mean(1)
        stable = (rec + h['origin'][:, None]) * SCALE
        result['reconstruction_component'] = stable
        result['reconstruction'] = warp(stable, np.linalg.inv(data['H'][idx, :P]))[0]
    return result


def train_one(data, meta, fold, seed, variant, cams, cfg):
    dest = OUT / f'fold_{fold["fold"]}/seed_{seed}/{variant}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        record = read(dest / 'complete.json')
        assert record['run_sha256'] == sha(Path(__file__))
        assert record['network_sha256'] == sha(Path(__file__).with_name('network.py'))
        return
    start = time.monotonic(); seed_all(seed)
    model = PedestrianPredictor(coupled=variant.startswith('coupled')).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    weight = cfg['reconstruction_weight'] if variant.endswith('dual') else 0.
    best = float('inf'); beststate = None; logs = []
    for epoch in range(cfg['epochs']):
        model.train(); losses = []
        for update in range(cfg['updates_per_epoch']):
            rng = np.random.default_rng(seed * 100000 + epoch * 10 + update)
            idx = balanced(fold['train_indices'], meta, cfg['batch_size'], rng)
            h = input_history(data, idx, 'train', rng); target = targets(data, idx, h['origin'])
            args = model_inputs(model, h, data, idx, cams['full']); n = len(idx)
            loss, lf, lh = model.loss(**args, past=tensor(target['past']), future=tensor(target['future']),
                t1=tensor(rng.random(n)), t2=tensor(rng.random(n)),
                n1=tensor(rng.normal(0, cfg['noise_sigma'], (n, P*2))), n2=tensor(rng.normal(0, cfg['noise_sigma'], (n, F*2))), weight=weight)
            assert torch.isfinite(loss)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            losses.append([loss.item(), lf.item(), lh.item()])
        if (epoch + 1) % cfg['validation_every'] == 0:
            model.eval(); idx = np.array(fold['val_indices'])
            result = infer_component(model, data, meta, idx, cams['full'], cfg)
            truth = warp(data['xy'][idx, P:], data['H'][idx, P:])[0]
            val = float(np.linalg.norm(result['component'] - truth, axis=-1).mean())
            assert np.isfinite(val)
            logs.append(dict(epoch=epoch+1, total=float(np.mean(losses,0)[0]), future=float(np.mean(losses,0)[1]), history=float(np.mean(losses,0)[2]), validation_component_ADE=val))
            if val < best: best = val; beststate = copy.deepcopy(model.state_dict()); bestepoch = epoch + 1
    model.load_state_dict(beststate); model.eval()
    torch.save(dict(model=beststate, variant=variant, fold=fold, seed=seed, epoch=bestepoch, validation_component_ADE=best,
                    protocol_sha256=sha(OUT/'protocol.json')), dest/'checkpoint.pt')
    write(dest/'training.json', logs)
    idx = np.array(fold['test_indices'])
    for mode in cfg['evaluation_inputs']:
        result = infer_component(model, data, meta, idx, cams['full'], cfg, mode, with_history=variant.endswith('dual'))
        projections = dict(full=cams['full'], old=cams['old'], frozen=np.zeros_like(cams['full']), reference=data['cam'][:, P:])
        for arm, camera in projections.items():
            pred, depth = project(result['component'], camera, data, idx)
            result[arm] = pred; result[arm+'_depth'] = depth
        np.savez_compressed(dest/f'predictions_{mode}.npz', indices=idx, **result)
    record = dict(seconds=time.monotonic()-start, epoch=bestepoch, validation_component_ADE=best,
                  total_parameters=sum(p.numel() for p in model.parameters()), inference_parameters=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('history.')),
                  run_sha256=sha(Path(__file__)), network_sha256=sha(Path(__file__).with_name('network.py')), shared_sha256=sha(Path(__file__).with_name('shared.py')))
    write(dest/'complete.json', record)
    print(f'fold={fold["fold"]} seed={seed} {variant}: epoch={bestepoch}, validation residual ADE={best:.2f}, seconds={record["seconds"]:.1f}', flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--folds', nargs='+', type=int); args=parser.parse_args()
    cfg=read(OUT/'protocol.json'); data=dict(np.load(OLD/'dataset.npz')); meta=read(OLD/'samples.json')
    assert read(OUT/'preflight.json')['passed']
    print('Device', DEVICE, 'torch', torch.__version__, flush=True)
    for fold in read(OUT/'folds.json'):
        if args.folds is not None and fold['fold'] not in args.folds: continue
        for seed in cfg['seeds']:
            cams=cameras(fold['fold'],seed)
            for variant in VARIANTS: train_one(data,meta,fold,seed,variant,cams,cfg)
    if args.folds is None: write(OUT/'training_complete.json', dict(passed=True, pedestrian_models=54, reused_camera_models=18))


if __name__ == '__main__': main()
