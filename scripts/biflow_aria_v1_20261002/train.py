"""Recording-disjoint training with frozen camera forecasts and paired seeds."""
from common import *
from model import *
import time,random,copy,collections,argparse

torch.set_num_threads(2)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def tensors(feat):return {k:tensor(feat[k]) for k in ['seq','origin','camera','past','future']}
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def balanced(indices,metadata,n,rng):
    groups=sorted({metadata[i]['recording'] for i in indices});chosen=rng.choice(groups,n)
    return np.array([rng.choice([i for i in indices if metadata[i]['recording']==g]) for g in chosen])
def noise_for(meta,indices,kind,n=5):
    size=F*2 if kind=='future' else P*2
    return np.stack([np.random.default_rng(100000+int(hashlib.sha256((meta[i]['key']+kind).encode()).hexdigest()[:8],16)).normal(0,.15,(n,size)) for i in indices]).astype(np.float32)
def camera_fit(data,cm,fold,seed,cfg):
    folder=OUT/f'fold_{fold["fold"]}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/'camera.pt'
    seed_all(seed);model=CameraPredictor().to(DEVICE);logs=[]
    if path.exists():model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=False)['model'])
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);best=float('inf');beststate=None
        for epoch in range(cfg['epochs']):
            model.train();losses=[]
            for update in range(cfg['updates_per_epoch']):
                rng=np.random.default_rng(seed*100000+epoch*10+update);idx=balanced(fold['camera_train_indices'],cm,64,rng);x=tensor(data['camera'][idx,:P]);y=tensor(data['camera'][idx,P:]);loss=(model(x)-y).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(loss.item())
            if (epoch+1)%cfg['validation_every']==0:
                model.eval()
                with torch.no_grad():idx=fold['camera_val_indices'];vl=(model(tensor(data['camera'][idx,:P]))-tensor(data['camera'][idx,P:])).square().mean().item()
                logs.append(dict(epoch=epoch+1,train=float(np.mean(losses)),validation=vl))
                if vl<best:best=vl;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch+1
        model.load_state_dict(beststate);torch.save(dict(model=beststate,epoch=bestepoch,validation=best,fold=fold,seed=seed),path);write(folder/'camera_training.json',logs)
    model.eval()
    with torch.no_grad():pred=model(tensor(data['cam'][:,:P])).cpu().numpy()
    np.save(folder/'camera_predictions.npy',pred);return pred

def sample_result(model,data,meta,idx,comp,cam,mode,cfg,with_history=False):
    # Fixed corruption and flow starting noise make all four variants paired.
    feat=make_features(data,idx,comp,cam,mode,np.random.default_rng(702));t=tensors(feat)
    sampled=model.sample(t['seq'],t['origin'],t['camera'],tensor(noise_for(meta,idx,'future',cfg['inference_samples'])),steps=cfg['flow_steps']).cpu().numpy().reshape(len(idx),cfg['inference_samples'],F,2)
    points=(sampled.mean(1)+feat['origin'][:,None])*SCALE;image=[];oracle=[];depth=[];rec=[]
    for j,i in enumerate(idx):
        pred,z=project_future(points[j],cam[i],data['K'][i]) if comp else (points[j],np.ones(F));image.append(pred);depth.append(z)
        o=project_future(points[j],data['cam'][i,P:],data['K'][i])[0] if comp else pred;oracle.append(o)
    if with_history:
        rp=model.sample(t['seq'],t['origin'],t['camera'],tensor(noise_for(meta,idx,'history',cfg['inference_samples'])),kind='history',steps=cfg['flow_steps']).cpu().numpy().reshape(len(idx),cfg['inference_samples'],P,2).mean(1)
        rp=(rp+feat['origin'][:,None])*SCALE
        for j,i in enumerate(idx):rec.append(warp(rp[j],np.linalg.inv(data['H'][i,:P]))[0] if comp else rp[j])
    return dict(pred=np.array(image),depth=np.array(depth),oracle=np.array(oracle),component=points,samples_normalized=sampled,reconstruction=np.array(rec),input_xy=feat['input_xy'],input_valid=feat['valid'])

def run_model(data,meta,fold,seed,name,cam,cfg):
    folder=OUT/f'fold_{fold["fold"]}'/f'seed_{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():print('resume',fold['fold'],seed,name,flush=True);return
    begin=time.monotonic();comp=name.startswith('comp');dual=name.endswith('dual');seed_all(seed)
    model=DualFlow(cfg['hidden']).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);best=float('inf');beststate=None;training=[]
    for epoch in range(cfg['epochs']):
        model.train();losses=[]
        for update in range(cfg['updates_per_epoch']):
            rng=np.random.default_rng(seed*100000+epoch*10+update);idx=balanced(fold['train_indices'],meta,cfg['batch_size'],rng);feat=make_features(data,idx,comp,cam,'train',rng);t=tensors(feat);B=len(idx)
            t1=tensor(rng.random(B));t2=tensor(rng.random(B));n1=tensor(rng.normal(0,cfg['noise_sigma'],(B,P*2)));n2=tensor(rng.normal(0,cfg['noise_sigma'],(B,F*2)))
            loss,lf,lh=model.loss(**t,t1=t1,t2=t2,n1=n1,n2=n2,weight=cfg['reconstruction_weight'] if dual else 0.)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append([loss.item(),lf.item(),lh.item()])
        if (epoch+1)%cfg['validation_every']==0:
            model.eval();idx=fold['val_indices'];result=sample_result(model,data,meta,idx,comp,cam,'reviewed',cfg);vl=float(np.mean([metrics(result['pred'][j],data['xy'][i,P:],result['depth'][j])['ADE15'] for j,i in enumerate(idx)]))
            training.append(dict(epoch=epoch+1,total=float(np.mean(losses,0)[0]),future=float(np.mean(losses,0)[1]),history=float(np.mean(losses,0)[2]),validation_ADE=vl))
            if vl<best:best=vl;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch+1
    model.load_state_dict(beststate);model.eval();torch.save(dict(model=beststate,epoch=bestepoch,validation_ADE=best,fold=fold,seed=seed,variant=name,protocol_sha256=sha(OUT/'protocol.json')),folder/'checkpoint.pt');write(folder/'training.json',training)
    for mode in cfg['evaluation_inputs']:
        result=sample_result(model,data,meta,fold['test_indices'],comp,cam,mode,cfg,with_history=dual)
        np.savez_compressed(folder/f'predictions_{mode}.npz',indices=np.array(fold['test_indices']),**result)
    write(folder/'complete.json',dict(seconds=time.monotonic()-begin,epoch=bestepoch,val_ADE=best,total_parameters=sum(p.numel() for p in model.parameters()),inference_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('history.')),code_sha256=sha(Path(__file__)),model_code_sha256=sha(Path(__file__).with_name('model.py'))))
    print(f"fold={fold['fold']} seed={seed} {name} epoch={bestepoch} val={best:.1f} sec={time.monotonic()-begin:.1f}",flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--folds',nargs='+',type=int);p.add_argument('--seeds',nargs='+',type=int);a=p.parse_args();cfg=read(OUT/'protocol.json');data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');cm=read(OUT/'camera_samples.json')
    print('Device',DEVICE,'torch',torch.__version__,flush=True)
    for fold in read(OUT/'folds.json'):
        if a.folds is not None and fold['fold'] not in a.folds:continue
        for seed in a.seeds or cfg['seeds']:
            cam=camera_fit(data,cm,fold,seed,cfg)
            for name in cfg['variants']:run_model(data,meta,fold,seed,name,cam,cfg)
if __name__=='__main__':main()
