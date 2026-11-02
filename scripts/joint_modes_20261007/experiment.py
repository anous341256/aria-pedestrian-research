"""Locked 117-window, six-fold, three-seed joint-mode experiment."""
from pathlib import Path
import sys, importlib.util, json, time, copy, argparse, hashlib
import numpy as np
import torch
from joint_model import JointModes, angle_error, joint_losses, group_average
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/factorized_forecast_v1_20261002'))
from shared import input_history, targets, P, F, SCALE, warp, read, write, sha
from model import CameraPredictor
from train import noise_for
spec = importlib.util.spec_from_file_location('original_person_network', ROOT/'scripts/factorized_forecast_v1_20261002/network.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
PedestrianPredictor = module.PedestrianPredictor
OUT = ROOT/'outputs/joint_modes_20261007'
OLD = ROOT/'outputs/biflow_aria_v1_20261002'
BASE = ROOT/'outputs/factorized_forecast_v1_20261002'
CAM = ROOT/'outputs/camera_full_data_20261002'
ATT = ROOT/'outputs/transformer_fusion_20261005'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)


def tensor(x): return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def groups_for(meta, indices):
    keys = [(meta[int(i)]['recording'], meta[int(i)]['timestamp_ns']) for i in indices]
    mapping = {key: j for j, key in enumerate(sorted(set(keys)))}
    return torch.tensor([mapping[k] for k in keys], dtype=torch.long, device=DEVICE)


def batch_indices(indices, meta, rng, count=48):
    # Recording-balanced sampling; include all eligible historical targets at a cutoff.
    rec = sorted({meta[i]['recording'] for i in indices})
    selected = []
    while len(selected) < count:
        r = rng.choice(rec); candidates = [i for i in indices if meta[i]['recording'] == r]
        i = int(rng.choice(candidates)); t = meta[i]['timestamp_ns']
        selected.extend(j for j in candidates if meta[j]['timestamp_ns'] == t)
    return np.asarray(selected, int)  # Complete final group, so batch can exceed 48.


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(version='joint_modes_20261007', seeds=[17,29,43], modes=[1,3],
        scope='Original117 exact recording-held-out folds, frozen existing GRU/flow backbones, small paired residual adapters.',
        camera_updates=2000, person_updates=240, camera_batch=256, person_min_batch=48,
        learning_rate=.001, weight_decay=.01, clip=1., validation_every_camera=100, validation_every_person=20,
        shared_mode_dimension=8, adapter_width=32, three_loss_weights=[1.,1.,1.], score_weight=.1,
        temperature=.5, camera_replay_weight=.25, camera_replay_score_weight=.1,
        scale_rule='Training-only frozen-base mean angular/residual/image errors; floors 1 degree,10px,10px. Camera pretraining uses full allowed training baseline angular error.',
        initialization='Training-only kmeans of camera residual trajectories; shared 8D mode codes. K1 identical pipeline/control.',
        selection='Validation selected-mode normalized three-component error; camera pretrain uses selected-mode angular error.',
        inference='Mean logits across eligible historical person tracks at identical recording/timestamp, argmax once, same paired mode for all. No average of final candidate trajectories.',
        population='Only already eligible dataset tracks; this is not a full-scene online detector/census.',
        training='Frozen old backbone predictions use original five fixed flow draws as deterministic base. Camera adapters pretrain on full permitted MPS; both adapters jointly train with all three losses plus full-camera replay. No new reconstruction loss; original dual-task backbone retained.',
        comparison='Old A-F and GRU are frozen system references. New adapters have additional training exposure, different objective/selection/capacity. K1 is matched extra-training control; K3 adds modes/selection, not cross-attention.',
        labels='Offline SAM references, MPS rotation; residual is not pure pedestrian motion.', no_test_tuning=True)
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json') == cfg
    else: write(OUT/'protocol.json', cfg)
    files = [OLD/'dataset.npz',OLD/'samples.json',OLD/'folds.json',CAM/'dataset.npz',CAM/'folds.json',ATT/'protocol.json']
    for f in range(6):
        for s in cfg['seeds']:
            files += list((BASE/f'fold_{f}/seed_{s}/separated_dual').glob('*'))
            files += list((CAM/f'fold_{f}/seed_{s}/full').glob('*'))
            files += list((ATT/f'fold_{f}/seed_{s}').glob('*/*.pt'))
            files += list((ATT/f'fold_{f}/seed_{s}').glob('*/predictions_reviewed.npz'))
    files += list((ROOT/'scripts/factorized_forecast_v1_20261002').glob('*.py'))
    files += list((ROOT/'scripts/transformer_fusion_20261005').glob('*.py'))
    if not (OUT/'source_hashes.json').exists(): write(OUT/'source_hashes.json', {str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()})
    write(OUT/'folds.json', read(OLD/'folds.json'))
    return cfg


def load_backbones(fold, seed):
    c = CameraPredictor().to(DEVICE)
    c.load_state_dict(torch.load(CAM/f'fold_{fold}/seed_{seed}/full/camera.pt',map_location=DEVICE,weights_only=False)['model'])
    p = PedestrianPredictor(False).to(DEVICE)
    p.load_state_dict(torch.load(BASE/f'fold_{fold}/seed_{seed}/separated_dual/checkpoint.pt',map_location=DEVICE,weights_only=False)['model'])
    for model in [c,p]:
        model.eval()
        for v in model.parameters(): v.requires_grad_(False)
    return c,p


@torch.no_grad()
def camera_features(camera, history):
    features, predictions = [], []
    for x in np.array_split(history, max(1,int(np.ceil(len(history)/2048)))):
        h = camera.encoder(tensor(x))[1][-1]
        features.append(h); predictions.append(camera.head(h).reshape(-1,F,3))
    return torch.cat(features), torch.cat(predictions)


@torch.no_grad()
def inputs(data, meta, idx, camera, person, mode='reviewed', rng=None):
    h = input_history(data,idx,mode,rng)
    cf,cb = camera_features(camera,data['cam'][idx,:P])
    seq, origin = tensor(h['seq']), tensor(h['origin'])
    pf = person.encode(seq,origin)
    pb = person.sample(seq,origin,tensor(noise_for(meta,idx,'future',5)),steps=10).reshape(len(idx),5,F,2).mean(1)
    return dict(camera_feature=cf, base_camera=cb, person_feature=pf, base_person=pb,
                origin=origin, K=tensor(data['K'][idx]), groups=groups_for(meta,idx))


def forward(model, x):
    return model(x['camera_feature'], x['base_camera'], x['person_feature'], x['base_person'])


def target_tensors(data, idx, origin):
    y = targets(data,idx,origin.detach().cpu().numpy())
    return dict(true_rotation=tensor(data['cam'][idx,P:]), true_residual=tensor(y['future'].reshape(-1,F,2)),true_image=tensor(data['xy'][idx,P:]))


def loss(model, x, y, scales):
    r,p,s = forward(model,x)
    return joint_losses(r,p,s,x['origin'],x['K'],groups=x['groups'],scales=scales,**y)


def camera_loss(model, cf, cb, truth, scale):
    rv,logits = model.camera_candidates(cf,cb)
    error = angle_error(rv,truth[:,None]).mean(-1)/scale
    target = torch.softmax(-error.detach()/.5,-1)
    return error.min(-1).values.mean() + .1 * (-(target*torch.log_softmax(logits,-1)).sum(-1).mean())


def initialize_model(k, cf, cb, truth, px, py):
    residual = (truth-cb).cpu().numpy().reshape(len(cb),45)
    std = np.maximum(residual.std(0), .01)
    if k == 1: anchors = residual.mean(0,keepdims=True)
    else:
        z = residual / std
        # Deterministic farthest point seeding, followed by Lloyd updates; training only.
        centers = [z[np.argmin(np.linalg.norm(z-z.mean(0),axis=1))]]
        while len(centers)<k:
            d = np.min(np.stack([np.sum((z-c)**2,axis=1) for c in centers]),axis=0)
            centers.append(z[d.argmax()])
        centers = np.asarray(centers)
        for _ in range(20):
            which = np.square(z[:,None]-centers).sum(-1).argmin(-1)
            centers = np.stack([z[which==j].mean(0) if np.any(which==j) else centers[j] for j in range(k)])
        anchors = centers*std
    ps = (py['true_residual']-px['base_person']).flatten(1).std(0,unbiased=False).clamp(min=.002).cpu().numpy()
    return JointModes(k, anchors, std, ps).to(DEVICE)


@torch.no_grad()
def infer(model, data, meta, idx, camera, person):
    model.eval(); x=inputs(data,meta,idx,camera,person)
    rv,p,logits=forward(model,x)
    from joint_model import project
    components=(p+x['origin'][:,None,None])*tensor(SCALE)
    images,depth=project(components,rv,x['K'])
    gl=group_average(logits,x['groups']); chosen=gl.argmax(-1)[x['groups']]
    rows=torch.arange(len(idx),device=DEVICE)
    return {key:value.detach().cpu().numpy() for key,value in dict(camera_candidates=rv,component_candidates=components,
        image_candidates=images,depth_candidates=depth,logits=logits,group_logits=gl,groups=x['groups'],selected=chosen,
        camera=rv[rows,chosen],component=components[rows,chosen],pred=images[rows,chosen],depth=depth[rows,chosen]).items()}


def fit(data,meta,fold,full,cfold,seed,k,cfg):
    dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/K{k}'; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists(): return
    start=time.monotonic();seed_all(seed);camera,person=load_backbones(fold['fold'],seed)
    tr=np.asarray(fold['train_indices']);va=np.asarray(fold['val_indices'])
    cx,cb=camera_features(camera,full[cfold['train_indices'],:P]);cy=tensor(full[cfold['train_indices'],P:])
    vx,vb=camera_features(camera,full[cfold['val_indices'],:P]);vy=tensor(full[cfold['val_indices'],P:])
    px=inputs(data,meta,tr,camera,person);py=target_tensors(data,tr,px['origin'])
    model=initialize_model(k,cx,cb,cy,px,py)
    base_er=angle_error(px['base_camera'],py['true_rotation']).mean().item()
    base_ep=torch.linalg.vector_norm((px['base_person']-py['true_residual'])*tensor(SCALE),dim=-1).mean().item()
    from joint_model import project
    base_image,_=project(((px['base_person']+px['origin'][:,None])*tensor(SCALE))[:,None],px['base_camera'][:,None],px['K'])
    base_eu=torch.linalg.vector_norm(base_image[:,0]-py['true_image'],dim=-1).mean().item()
    scales=[max(base_er,np.pi/180),max(base_ep,10.),max(base_eu,10.)]
    camera_scale=max(angle_error(cb,cy).mean().item(),np.pi/180)
    write(dest/'normalization.json',dict(scales=scales,camera_scale=camera_scale,training_person_indices=tr.tolist(),
        training_camera_count=len(cx),test=fold['test'],validation=fold['validation']))
    params=list(model.camera.parameters())+list(model.camera_score.parameters())+[model.mode]
    opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);logs=[];best=float('inf')
    for step in range(1,cfg['camera_updates']+1):
        ii=rng.integers(len(cx),size=cfg['camera_batch'])
        model.train();cl=camera_loss(model,cx[ii],cb[ii],cy[ii],camera_scale)
        assert torch.isfinite(cl)
        opt.zero_grad();cl.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        if step%cfg['validation_every_camera']==0:
            model.eval()
            with torch.no_grad():
                r,s=model.camera_candidates(vx,vb); selected=s.argmax(-1)
                val=angle_error(r[torch.arange(len(r),device=DEVICE),selected],vy).mean().item()
            logs.append(dict(step=step,loss=cl.item(),validation_angle=val))
            if val<best:best=val;state=copy.deepcopy(model.state_dict());beststep=step
    model.load_state_dict(state);write(dest/'camera_training.json',logs)
    torch.save(dict(model=state,selected_step=beststep),dest/'camera_pretrained.pt')
    vpx=inputs(data,meta,va,camera,person);vpy=target_tensors(data,va,vpx['origin'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    logs=[];best=float('inf')
    for step in range(1,cfg['person_updates']+1):
        rng=np.random.default_rng(seed*100000+(step-1)//2*10+(step-1)%2)
        idx=batch_indices(fold['train_indices'],meta,rng)
        x=inputs(data,meta,idx,camera,person,'train',rng);y=target_tensors(data,idx,x['origin'])
        model.train();jl,_=loss(model,x,y,scales)
        ii=rng.integers(len(cx),size=cfg['camera_batch']);cl=camera_loss(model,cx[ii],cb[ii],cy[ii],camera_scale)
        total=jl+.25*cl;assert torch.isfinite(total)
        opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%cfg['validation_every_person']==0:
            model.eval()
            with torch.no_grad():
                _,v=loss(model,vpx,vpy,scales);choice=v['group_logits'].argmax(-1)
                val=v['group_error'].gather(1,choice[:,None]).mean().item()
            logs.append(dict(step=step,loss=total.item(),validation_joint_error=val,batch=len(idx)))
            if val<best:best=val;state=copy.deepcopy(model.state_dict());beststep=step
    model.load_state_dict(state);model.eval()
    torch.save(dict(model=state,modes=k,fold=fold,seed=seed,scales=scales,selected_step=beststep,
        protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt')
    write(dest/'joint_training.json',logs)
    for name,idx in [('test',fold['test_indices']),('validation',fold['val_indices'])]:
        z=infer(model,data,meta,np.asarray(idx),camera,person)
        np.savez_compressed(dest/f'predictions_{name}.npz',indices=idx,**z)
    write(dest/'complete.json',dict(seconds=time.monotonic()-start,selected_step=beststep,validation_joint_error=best,
        adapter_parameters=sum(p.numel() for p in model.parameters()),frozen_backbone_parameters=sum(p.numel() for p in camera.parameters())+sum(p.numel() for p in person.parameters()),
        code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['experiment.py','joint_model.py']}))
    print(f'fold={fold["fold"]} seed={seed} K={k} complete {time.monotonic()-start:.1f}s; selected={beststep}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--fold',type=int);args=parser.parse_args()
    cfg=prepare()
    if args.prepare_only:return
    data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');full=np.load(CAM/'dataset.npz')['rotation']
    folds=read(OLD/'folds.json');cfolds=read(CAM/'folds.json')
    start=time.monotonic();print('Device',DEVICE,flush=True)
    for fold,cfold in zip(folds,cfolds):
        assert fold['test']==cfold['test'] and fold['validation']==cfold['validation']
        assert not set(fold['train']) & {fold['test'],fold['validation']}
        if args.fold is not None and fold['fold']!=args.fold:continue
        for seed in cfg['seeds']:
            for k in cfg['modes']:fit(data,meta,fold,full,cfold,seed,k,cfg)
    write(OUT/'training_complete.json',dict(seconds=time.monotonic()-start,models=36,complete_folds=[args.fold] if args.fold is not None else list(range(6))))


if __name__=='__main__':main()
