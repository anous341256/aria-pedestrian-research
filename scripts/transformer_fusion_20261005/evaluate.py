"""All held-out summaries, references, diagnostics; no model selection here."""
from shared import *


def reduce(rows):
    if not rows:return None
    keys=['ADE05','ADE10','ADE15','FDE15']
    def mean(rr):return {k:float(np.mean([r['metrics'][k] for r in rr])) for k in keys}
    def group(key):return {g:mean([r for r in rows if str(r[key])==g]) for g in sorted({str(r[key]) for r in rows})}
    rec=group('recording');track=group('track')
    return dict(windows=len({r['key'] for r in rows}),rows=len(rows),window_mean=mean(rows),
                recording_equal={k:float(np.mean([v[k] for v in rec.values()])) for k in keys},
                track_equal={k:float(np.mean([v[k] for v in track.values()])) for k in keys},
                per_recording=rec,per_seed=group('seed'),invalid_steps=sum(r['metrics']['invalid'] for r in rows))


def main():
    assert read(OUT/'training_complete.json')['passed']
    cfg=read(OUT/'protocol.json');data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json')
    rows=[];residual=[];horizons={};displacement=[]
    def add(pred,depth,idx,seed,mode,method):
        errors=np.linalg.norm(pred-data['xy'][idx,P:],axis=-1)
        errors[(~np.isfinite(errors))|(depth<=1e-6)]=32000.
        if mode=='reviewed':horizons.setdefault(method,[]).extend(errors.tolist())
        for j,i in enumerate(idx):
            m=meta[i];height=float((data['boxes'][i,P-1,3]-data['boxes'][i,P-1,1])/SCALE[1])
            turn=float(np.degrees(np.linalg.norm(data['cam'][i,-1])))
            rows.append(dict(index=int(i),key=m['key'],recording=m['recording'],clip=m['clip'],track=m['clip']+'_'+m['id'],
                             seed=seed,mode=mode,method=method,nonoverlap=m['nonoverlap'],edge=m['edge'],height=height,turn=turn,
                             metrics=metrics(pred[j],data['xy'][i,P:],depth[j])))
            if mode=='reviewed' and method in VARIANTS:
                displacement.append(dict(key=m['key'],method=method,seed=seed,
                                         predicted_final_distance=float(np.linalg.norm(pred[j,-1]-data['xy'][i,P-1])),
                                         reference_final_distance=float(np.linalg.norm(data['xy'][i,-1]-data['xy'][i,P-1]))))
    for fold in read(OUT/'folds.json'):
        f=fold['fold'];idx=np.array(fold['test_indices']); ones=np.ones((len(idx),F))
        stable=warp(data['xy'][idx,P:],data['H'][idx,P:])[0]
        for seed in cfg['seeds']:
            for mode in cfg['evaluation_inputs']:
                h=history(data,idx,False,mode,np.random.default_rng(702))
                add(np.repeat(h['input_xy'][:,-1,None],F,axis=1),ones,idx,seed,mode,'hold_last')
                if mode=='reviewed':
                    t=data['seconds'][idx,:P];t=t-t.mean(1,keepdims=True)
                    velocity=np.sum(t[...,None]*h['input_xy'],axis=1)/np.sum(t*t,axis=1)[:,None]
                    cv=h['input_xy'][:,-1,None]+data['seconds'][idx,P:,None]*velocity[:,None]
                    add(cv,ones,idx,seed,mode,'raw_cv')
                z=np.load(OLD/f'fold_{f}/seed_{seed}/comp_dual/predictions_{mode}.npz');assert np.array_equal(z['indices'],idx)
                add(z['pred'],z['depth'],idx,seed,mode,'old_comp_dual')
                z=np.load(PREVIOUS/f'fold_{f}/seed_{seed}/separated_dual/predictions_{mode}.npz');assert np.array_equal(z['indices'],idx)
                add(z['full'],z['full_depth'],idx,seed,mode,'previous_separated')
                for v in VARIANTS:
                    z=np.load(OUT/f'fold_{f}/seed_{seed}/{v}/predictions_{mode}.npz');assert np.array_equal(z['indices'],idx)
                    add(z['pred'],z['depth'],idx,seed,mode,v)
                    if v[0] in 'CF':
                        for diagnostic in ['reference','frozen']:
                            add(z[diagnostic],z[diagnostic+'_depth'],idx,seed,mode,v+'/'+diagnostic+'_diagnostic')
                        for j,i in enumerate(idx):
                            e=np.linalg.norm(z['component'][j]-stable[j],axis=-1)
                            residual.append(dict(key=meta[i]['key'],method=v,seed=seed,mode=mode,ADE=float(e.mean()),FDE=float(e[-1])))
    write(OUT/'metrics.json',rows);write(OUT/'residual_metrics.json',residual);write(OUT/'displacement.json',displacement)
    allmethods=sorted({r['method'] for r in rows});summary={}
    for mode in cfg['evaluation_inputs']:
        summary[mode]={m:reduce([r for r in rows if r['mode']==mode and r['method']==m]) for m in allmethods}
    reviewed=[r for r in rows if r['mode']=='reviewed']
    summary['sparse']={m:reduce([r for r in reviewed if r['method']==m and r['nonoverlap']]) for m in allmethods}
    summary['horizon']={m:np.mean(e,axis=0).tolist() for m,e in horizons.items()}
    summary['residual']={m:{mode:{key:float(np.mean([r[key] for r in residual if r['method']==m and r['mode']==mode]))
                                for key in ['ADE','FDE']} for mode in cfg['evaluation_inputs']} for m in [VARIANTS[2],VARIANTS[5]]}
    conditions={
        'turn_lt5':lambda r:r['turn']<5,'turn_5to15':lambda r:5<=r['turn']<15,'turn_ge15':lambda r:r['turn']>=15,
        'height_lt15pct':lambda r:r['height']<.15,'height_15to30pct':lambda r:.15<=r['height']<.3,'height_ge30pct':lambda r:r['height']>=.3,
        'border':lambda r:r['edge'],'not_border':lambda r:not r['edge']}
    summary['conditions']={name:{m:reduce([r for r in reviewed if r['method']==m and func(r)]) for m in allmethods} for name,func in conditions.items()}
    pairs=[('D_bidirectional','A_early_fusion'),('E_oneway','D_bidirectional'),('F_interacting_geometry','C_separated'),
           ('A_early_fusion','B_person_only'),('E_oneway','B_person_only'),('F_interacting_geometry','E_oneway')]
    paired={}
    for a,b in pairs:
        ar={(r['key'],r['seed']):r for r in reviewed if r['method']==a};br={(r['key'],r['seed']):r for r in reviewed if r['method']==b}
        assert ar.keys()==br.keys()
        differences=[dict(key=k[0],seed=k[1],recording=ar[k]['recording'],delta_ADE=ar[k]['metrics']['ADE15']-br[k]['metrics']['ADE15']) for k in ar]
        record={rec:float(np.mean([d['delta_ADE'] for d in differences if d['recording']==rec])) for rec in sorted({d['recording'] for d in differences})}
        per_window={key:float(np.mean([d['delta_ADE'] for d in differences if d['key']==key])) for key in sorted({d['key'] for d in differences})}
        paired[a+' minus '+b]=dict(window_mean_delta=float(np.mean(list(per_window.values()))),
                                  recording_equal_delta=float(np.mean(list(record.values()))),per_recording_delta=record,
                                  recording_wins=sum(d<0 for d in record.values()),window_wins=sum(d<0 for d in per_window.values()),
                                  per_seed_delta={str(s):float(np.mean([d['delta_ADE'] for d in differences if d['seed']==s])) for s in cfg['seeds']})
    summary['paired']=paired;write(OUT/'summary.json',summary)
    for v in VARIANTS+['hold_last','old_comp_dual','previous_separated']:
        z=summary['reviewed'][v];print(v,{k:round(val,2) for k,val in z['window_mean'].items()},'recording',round(z['recording_equal']['ADE15'],2))


if __name__=='__main__':main()
