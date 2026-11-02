"""Out-of-recording results; retain every seed/window, including invalid projections."""
from common import *
import collections

def reduce_rows(rows):
    metrics_names=['ADE05','ADE10','ADE15','FDE15']
    def avg(rs):return {k:float(np.mean([r['metrics'][k] for r in rs])) for k in metrics_names}
    recordings={g:avg([r for r in rows if r['recording']==g]) for g in sorted({r['recording'] for r in rows})}
    tracks={g:avg([r for r in rows if r['track']==g]) for g in sorted({r['track'] for r in rows})}
    seeds={str(g):avg([r for r in rows if r['seed']==g]) for g in sorted({r['seed'] for r in rows})}
    return dict(prediction_rows=len(rows),unique_windows=len({r['key'] for r in rows}),window_mean=avg(rows),equal_recording={k:float(np.mean([v[k] for v in recordings.values()])) for k in metrics_names},equal_track={k:float(np.mean([v[k] for v in tracks.values()])) for k in metrics_names},per_recording=recordings,per_seed=seeds,invalid_steps=sum(r['metrics']['invalid'] for r in rows))

def main():
    cfg=read(OUT/'protocol.json');data=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');folds=read(OUT/'folds.json');rows=[];recon=[];camera=[];inference={};modelcounts=[]
    for fold in folds:
        for seed in cfg['seeds']:
            parent=OUT/f'fold_{fold["fold"]}'/f'seed_{seed}';cam=np.load(parent/'camera_predictions.npy')
            for i in fold['test_indices']:
                a,b=rotmat(cam[i]),rotmat(data['cam'][i,P:]);deg=np.linalg.norm(rotvec(np.einsum('tji,tjk->tik',a,b)),axis=-1)*180/np.pi
                camera.append(dict(key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,mean_degrees=float(deg.mean()),final_degrees=float(deg[-1])))
            for name in cfg['variants']:
                path=parent/name;complete=read(path/'complete.json');modelcounts.append(complete['total_parameters']);assert complete['code_sha256']==sha(Path(__file__).with_name('train.py'));assert complete['model_code_sha256']==sha(Path(__file__).with_name('model.py'))
                for mode in cfg['evaluation_inputs']:
                    pred=np.load(path/f'predictions_{mode}.npz')
                    for j,i in enumerate(pred['indices']):
                        m=meta[i];base=dict(key=m['key'],clip=m['clip'],id=m['id'],track=f'{m["clip"]}_{m["id"]}',recording=m['recording'],seed=seed,mode=mode,edge=m['edge'],nonoverlap=m['nonoverlap'],fold=fold['fold'],index=int(i))
                        rows.append(dict(base,method=name,metrics=metrics(pred['pred'][j],data['xy'][i,P:],pred['depth'][j])))
                        if name.startswith('comp'):
                            _,z=project_future(pred['component'][j],data['cam'][i,P:],data['K'][i]);rows.append(dict(base,method=name+'_reference_camera',metrics=metrics(pred['oracle'][j],data['xy'][i,P:],z)))
                        if name.endswith('dual'):
                            valid=pred['input_valid'][j].astype(bool);before=np.linalg.norm(pred['input_xy'][j]-data['xy'][i,:P],axis=-1);after=np.linalg.norm(pred['reconstruction'][j]-data['xy'][i,:P],axis=-1)
                            recon.append(dict(base,method=name,visible_input_MAE=float(before[valid].mean()),reconstruction_visible_MAE=float(after[valid].mean()),reconstruction_all_MAE=float(after.mean()),missing_steps=int((~valid).sum()),reconstruction_missing_MAE=float(after[~valid].mean()) if (~valid).any() else None))
                    # For display use the predeclared seed 17, not the best training seed.
                    if seed==17 and mode=='reviewed':inference[f'{fold["fold"]}_{name}']=str(path/f'predictions_{mode}.npz')
    assert len(set(modelcounts))==1 and len(modelcounts)==72
    # Non-learned references on the same reviewed histories/windows.
    for i,m in enumerate(meta):
        x=data['xy'][i,:P];tt=data['seconds'][i,:P];future_t=data['seconds'][i,P:];v=np.sum(tt[:,None]*(x-x[-1]),axis=0)/np.sum(tt**2);cv=x[-1]+future_t[:,None]*v
        normalized,_=warp(x,data['H'][i,:P]);v2=np.sum(tt[:,None]*(normalized-normalized[-1]),axis=0)/np.sum(tt**2);component=normalized[-1]+future_t[:,None]*v2;oracle,z=project_future(component,data['cam'][i,P:],data['K'][i])
        pp={'hold':(np.repeat(x[-1,None],F,axis=0),None),'constant_velocity':(cv,None),'comp_cv_reference_camera':(oracle,z),'comp_cv_frozen_camera':(component,None)}
        for name,(p,z) in pp.items():rows.append(dict(key=m['key'],clip=m['clip'],id=m['id'],track=f'{m["clip"]}_{m["id"]}',recording=m['recording'],seed=-1,mode='reviewed',edge=m['edge'],nonoverlap=m['nonoverlap'],index=i,method=name,metrics=metrics(p,data['xy'][i,P:],z)))
    summary={}
    for mode in cfg['evaluation_inputs']:
        rr=[r for r in rows if r['mode']==mode];summary[mode]={name:reduce_rows([r for r in rr if r['method']==name]) for name in sorted({r['method'] for r in rr})}
    for label,accept in [('nonoverlap',lambda r:r['nonoverlap']),('no_edge',lambda r:not r['edge']),('edge',lambda r:r['edge'])]:
        rr=[r for r in rows if r['mode']=='reviewed' and accept(r)];summary[label]={name:reduce_rows([r for r in rr if r['method']==name]) for name in sorted({r['method'] for r in rr})}
    history={}
    for mode in cfg['evaluation_inputs']:
        history[mode]={}
        for name in ['raw_dual','comp_dual']:
            rr=[r for r in recon if r['mode']==mode and r['method']==name];history[mode][name]={k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in ['visible_input_MAE','reconstruction_visible_MAE','reconstruction_all_MAE','reconstruction_missing_MAE'] if any(r[k] is not None for r in rr)}
    # Paired improvements, averaged over the three training seeds for each window.
    wins={}
    for left,right in [('raw_future','raw_dual'),('comp_future','comp_dual'),('raw_dual','comp_dual')]:
        by={name:{m['key']:np.mean([r['metrics']['ADE15'] for r in rows if r['mode']=='reviewed' and r['method']==name and r['key']==m['key']]) for m in meta} for name in [left,right]}
        wins[f'{right}_vs_{left}']=dict(improved=int(sum(by[right][m['key']]<by[left][m['key']] for m in meta)),windows=len(meta))
    write(OUT/'metrics.json',rows);write(OUT/'summary.json',summary);write(OUT/'history_reconstruction.json',dict(summary=history,rows=recon));write(OUT/'camera_metrics.json',dict(mean_degrees=float(np.mean([c['mean_degrees'] for c in camera])),final_degrees=float(np.mean([c['final_degrees'] for c in camera])),rows=camera));write(OUT/'paired_wins.json',wins);write(OUT/'display_sources.json',inference)
    print('REVIEWED', {k:{'window':v['window_mean']['ADE15'],'recording':v['equal_recording']['ADE15']} for k,v in summary['reviewed'].items()});print('HISTORY',history);print('WINS',wins)
if __name__=='__main__':main()
