"""Use selected checkpoints to distinguish fit quality and held-out errors."""
from train import *

def main():
    cfg=read(OUT/'protocol.json');data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');rows=[]
    for fold in read(OUT/'folds.json'):
        for seed in cfg['seeds']:
            parent=OUT/f'fold_{fold["fold"]}'/f'seed_{seed}';cam=np.load(parent/'camera_predictions.npy')
            for name in ['raw_dual','comp_dual']:
                model=DualFlow().to(DEVICE);checkpoint=torch.load(parent/name/'checkpoint.pt',map_location=DEVICE,weights_only=False);model.load_state_dict(checkpoint['model']);model.eval()
                for split in ['train','val','test']:
                    idx=fold[split+'_indices'];r=sample_result(model,data,meta,idx,name.startswith('comp'),cam,'controlled_noise',cfg,True);err=np.linalg.norm(r['reconstruction']-data['xy'][idx,:P],axis=-1);visible=r['input_valid'].astype(bool)
                    rows.append(dict(fold=fold['fold'],seed=seed,method=name,split=split,selected_epoch=checkpoint['epoch'],history_visible_error=float(err[visible].mean()),history_missing_error=float(err[~visible].mean()) if (~visible).any() else None,future_ADE=float(np.mean([metrics(r['pred'][j],data['xy'][i,P:],r['depth'][j])['ADE15'] for j,i in enumerate(idx)]))))
    averages={name:{split:{k:float(np.mean([r[k] for r in rows if r['method']==name and r['split']==split and r[k] is not None])) for k in ['history_visible_error','history_missing_error','future_ADE']} for split in ['train','val','test']} for name in ['raw_dual','comp_dual']}
    angles=[]
    for i,m in enumerate(meta):
        a=np.linalg.norm(data['cam'][i,P:],axis=-1)*180/np.pi;angles.append(float(a.mean()))
    write(OUT/'diagnostics.json',dict(averages=averages,rows=rows,frozen_camera_mean_angle_deg=float(np.mean(angles)),note='Post-training diagnosis only; no model or hyperparameter changes after test scores. Train/val/test means average fold-seed diagnostics, not independent samples.'))
    print(averages);print('camera frozen mean angle',np.mean(angles))
if __name__=='__main__':main()
