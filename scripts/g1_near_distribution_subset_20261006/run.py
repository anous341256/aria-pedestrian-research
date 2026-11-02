"""Locked paired training; exact unchanged old-arm training may reuse weights."""
from shared import *
from network import Predictor,torch
from model import CameraPredictor
from train import seed_all,tensor,DEVICE

def balanced(indices,metadata,n,rng):
    # Both arms use identical uniform-window sampling to preserve subset shares.
    return rng.choice(np.asarray(indices),n,replace=True)
import time,copy,shutil


def camera_features(data,fold,seed):
    path=CAMERA/f'fold_{fold}/seed_{seed}/full/camera.pt'
    camera=CameraPredictor().to(DEVICE)
    camera.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False)['model']); camera.eval()
    for p in camera.parameters():p.requires_grad_(False)
    with torch.no_grad():
        x=tensor(data['cam'][:,:P]); seq,last=camera.encoder(x)
        features=torch.cat([x,seq],-1).cpu().numpy()
        rotations=camera.head(last[-1]).reshape(-1,F,3).cpu().numpy()
    expected=np.load(path.with_name('predictions.npz'))['person_rotation']
    assert np.max(np.abs(rotations-expected))<1e-5
    return features,rotations


def forward(model,h,features,idx):
    kwargs=dict(seq=tensor(h['seq']),condition=tensor(h['condition']))
    if model.uses_camera_features:kwargs['camera']=tensor(features[idx])
    return model(**kwargs)


@torch.no_grad()
def infer(model,data,idx,features,rotations,mode='reviewed',diagnostic=False):
    h=history(data,idx,model.geometry,mode,np.random.default_rng(702))
    offsets=forward(model,h,features,idx).cpu().numpy().reshape(len(idx),F,2)
    points=(offsets+h['origin'][:,None])*SCALE
    pred,depth=project(points,rotations,data,idx) if model.geometry else (points,np.ones(points.shape[:2]))
    result=dict(pred=pred,depth=depth,component=points,input_xy=h['input_xy'],input_valid=h['input_valid'])
    if model.geometry and diagnostic:
        for name,rv in [('reference',data['cam'][:,P:]),('frozen',np.zeros_like(rotations))]:
            result[name],result[name+'_depth']=project(points,rv,data,idx)
    return result


def train_one(data,meta,fold,seed,variant,features,rotations,cfg):
    dest=OUT/f'fold_{fold["fold"]}/seed_{seed}/{variant}';dest.mkdir(parents=True,exist_ok=True)
    code_hashes={name:sha(Path(__file__).with_name(name)) for name in ['run.py','network.py','shared.py']}
    if (dest/'complete.json').exists():
        stored=read(dest/'complete.json')['code_hashes']
        assert stored==code_hashes
        return
    begin=time.monotonic();seed_all(seed)
    model=Predictor(variant,cfg['capacity'][variant]['ff_width']).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=float('inf'); logs=[]
    for epoch in range(cfg['epochs']):
        model.train(); losses=[]
        for update in range(cfg['updates_per_epoch']):
            rng=np.random.default_rng(seed*100000+epoch*10+update)
            idx=balanced(fold['train_indices'],meta,cfg['batch_size'],rng)
            h=history(data,idx,model.geometry,'train',rng)
            y=target(data,idx,h['origin'],model.geometry)
            loss=(forward(model,h,features,idx)-tensor(y)).square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            losses.append(loss.item())
        if (epoch+1)%cfg['validation_every']==0:
            model.eval(); idx=np.array(fold['val_indices']); z=infer(model,data,idx,features,rotations)
            val=float(np.mean([metrics(z['pred'][j],data['xy'][i,P:],z['depth'][j])['ADE15'] for j,i in enumerate(idx)]))
            logs.append(dict(epoch=epoch+1,training_mse=float(np.mean(losses)),validation_image_ADE=val))
            if val<best:best=val;state=copy.deepcopy(model.state_dict());bestepoch=epoch+1
    model.load_state_dict(state);model.eval()
    torch.save(dict(model=state,variant=variant,seed=seed,fold=fold,ff_width=cfg['capacity'][variant]['ff_width'],
                    epoch=bestepoch,validation_image_ADE=best,protocol_sha256=sha(OUT/'protocol.json')),dest/'checkpoint.pt')
    write(dest/'training.json',logs);idx=np.array(fold['test_indices'])
    for mode in cfg['evaluation_inputs']:
        z=infer(model,data,idx,features,rotations,mode,diagnostic=True)
        np.savez_compressed(dest/f'predictions_{mode}.npz',indices=idx,**z)
    record=dict(seconds=time.monotonic()-begin,epoch=bestepoch,validation_image_ADE=best,code_hashes=code_hashes,
                protocol_sha256=sha(OUT/'protocol.json'))
    write(dest/'complete.json',record)
    print(f'{fold["fold"]}/{seed}/{variant}: completed {record["seconds"]:.1f}s, selected epoch {bestepoch}',flush=True)


def main():
    cfg=read(OUT/'protocol.json'); assert read(OUT/'preflight.json')['passed']
    data=dict(np.load(DATA/'dataset.npz'));meta=read(DATA/'samples.json')
    print('Training',DEVICE,'torch',torch.__version__,flush=True)
    for fold in read(OUT/'folds.json'):
        for seed in cfg['seeds']:
            features,rotations=camera_features(data,fold['fold'],seed)
            for variant in VARIANTS:train_one(data,meta,fold,seed,variant,features,rotations,cfg)
    complete=list(OUT.glob('fold_*/seed_*/*/complete.json'));assert len(complete)==198
    write(OUT/'training_complete.json',dict(passed=True,models=198,reused_trajectory_models=sum(read(p).get('reused',False) for p in complete),reused_camera_models=33,new_camera_models=0))


if __name__=='__main__':main()
