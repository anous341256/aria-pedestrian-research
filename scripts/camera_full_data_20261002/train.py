"""Matched camera data-size experiment and separately labelled full-data refits."""
from common import *
import torch
from torch import nn
import copy, time, argparse

torch.set_num_threads(2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CameraPredictor(nn.Module):
    # Identical architecture and state keys to the previous camera model.
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(3,32,batch_first=True)
        self.head = nn.Sequential(nn.Linear(32,64),nn.SiLU(),nn.Linear(64,F*3))
    def forward(self,x):
        return self.head(self.encoder(x)[1][-1]).reshape(-1,F,3)


def tensor(x):
    return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)


class Batches:
    def __init__(self,n,seed):
        self.rng=np.random.default_rng(seed);self.n=n;self.left=np.array([],dtype=int)
    def next(self,size):
        while len(self.left)<size:
            self.left=np.concatenate([self.left,self.rng.permutation(self.n)])
        batch,self.left=self.left[:size],self.left[size:]
        return batch


@torch.no_grad()
def infer(model,x):
    model.eval()
    return np.concatenate([model(tensor(a)).cpu().numpy() for a in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))])


def fit(folder, train, validation, seed, cfg, steps=None):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        record=read(folder/'complete.json')
        assert record['train_code_sha256']==sha(Path(__file__))
        model=CameraPredictor().to(DEVICE)
        model.load_state_dict(torch.load(folder/'camera.pt',map_location=DEVICE,weights_only=False)['model'])
        return model,record
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=CameraPredictor().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    batches=Batches(len(train),seed);tr=tensor(train);logs=[];best=float('inf');state=None;start=time.time();seen=np.zeros(len(train),bool)
    count=steps or cfg['steps']
    for step in range(1,count+1):
        model.train();idx=batches.next(cfg['batch_size']);seen[idx]=True
        batch=tr[tensor(idx).long()];pred=model(batch[:,:P]);loss=(pred-batch[:,P:]).square().mean()
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
        if step%cfg['validation_every']==0 or step==count:
            vl=float(np.mean((infer(model,validation[:,:P])-validation[:,P:])**2)) if validation is not None else None
            logs.append(dict(step=step,train_batch_mse=loss.item(),validation_mse=vl))
            if validation is None or vl<best:
                best=vl if vl is not None else 0.;state=copy.deepcopy(model.state_dict());beststep=step
            write(folder/'progress.json',dict(step=step,total=count,best_step=beststep,seconds=time.time()-start,unique_train_windows_seen=int(seen.sum())))
    model.load_state_dict(state)
    record=dict(seed=seed,training_windows=len(train),validation_windows=len(validation) if validation is not None else 0,
                best_step=beststep,best_validation_mse=best if validation is not None else None,
                updates=count,unique_train_windows_seen=int(seen.sum()),parameters=sum(p.numel() for p in model.parameters()),
                seconds=time.time()-start,train_code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'),
                scope='all-recording refit; no held-out score' if validation is None else 'recording-held-out experiment')
    torch.save(dict(model=state,metadata=record),folder/'camera.pt');write(folder/'training.json',logs);write(folder/'complete.json',record)
    print(str(folder.relative_to(OUT)), 'best_step',beststep,'seconds',round(record['seconds'],1),flush=True)
    return model,record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--refit-only',action='store_true');args=parser.parse_args()
    cfg=read(OUT/'protocol.json');full=np.load(OUT/'dataset.npz')['rotation'];old=np.load(OLD/'dataset.npz')
    folds=read(OUT/'folds.json');print('DEVICE',DEVICE,flush=True)
    if not args.refit_only:
        for fold in folds:
            val=full[fold['val_indices']]
            for seed in cfg['seeds']:
                for arm in ['small_matched','full']:
                    train=full[fold['train_indices']] if arm=='full' else old['camera'][fold['small_train_indices']]
                    folder=OUT/f'fold_{fold["fold"]}'/f'seed_{seed}'/arm
                    model,record=fit(folder,train,val,seed,cfg)
                    if not (folder/'predictions.npz').exists():
                        test=np.asarray(fold['test_indices'])
                        person=infer(model,old['cam'][:,:P])
                        np.savez_compressed(folder/'predictions.npz',indices=test,rotation=infer(model,full[test,:P]),person_rotation=person)
    # An all-recording model is useful for subsequent work, but never for the scores above.
    for seed in cfg['seeds']:
        selected=[read(OUT/f'fold_{f["fold"]}'/f'seed_{seed}'/'full/complete.json')['best_step'] for f in folds]
        steps=max(int(np.median(selected)),int(np.ceil(len(full)/cfg['batch_size'])))
        fit(OUT/'all_recordings_refit'/f'seed_{seed}',full,None,seed,cfg,steps=steps)
    write(OUT/'training_complete.json',dict(passed=True,cv_models=36,refit_models=3,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))


if __name__=='__main__': main()
