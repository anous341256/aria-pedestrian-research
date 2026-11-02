"""Descriptive training diagnostics only; never selects data or model weights."""
from pathlib import Path
import json, hashlib, csv
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[2];V='tract_learning_map_20261029'
OUT=ROOT/'outputs'/V;SRC=ROOT/'outputs/horizon_15_15_g1_20261029';PRES=OUT/'presentation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
LABELS=['Easy','Trained','Hard','Confusing']
def classify(error,var,te,tv):return np.where(error>=te,np.where(var>=tv,'Confusing','Hard'),np.where(var>=tv,'Trained','Easy'))
def finite_corr(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)
def avg(v):return float(np.mean(v))
def aggregate(curve,ids,meta,equal_rec=False):
    if not equal_rec:return curve.mean(axis=1)
    rec=np.array([meta[i]['recording'] for i in ids]);return np.mean([curve[:,rec==r].mean(axis=1) for r in sorted(set(rec))],axis=0)
def main():
    verification=read(OUT/'verification.json');assert verification['passed']
    meta=read(SRC/'samples.json');folds=read(SRC/'folds.json');features=read(OUT/'edgc_features.json')
    PRES.mkdir(parents=True,exist_ok=True)
    fp=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc');matplotlib.font_manager.fontManager.addfont(fp.get_file())
    plt.rcParams.update({'font.family':fp.get_name(),'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    cache={};infos={};rows=[];thresholds=[];curvegroups=defaultdict(list);paired=[];corr=[];transitions=Counter();steps=np.arange(0,241,20)
    for f in folds:
      for seed in [17,29,43]:
        arrays={}
        for arm in ['original_long','expanded_long']:
          folder=OUT/arm/f'fold_{f["fold"]}'/f'seed_{seed}'
          z=dict(np.load(folder/'curves.npz'));info=read(folder/'complete.json')
          cache[(f['fold'],seed,arm)]=z;infos[(f['fold'],seed,arm)]=info;arrays[arm]=z
          assert np.array_equal(z['steps'],steps) and set(z['indices'])==set(f['expanded_train_indices'])
          assert not set(z['indices'])&(set(f['test_indices'])|set(f['val_indices']))
          assert np.isfinite(z['residual']).all() and np.isfinite(z['image']).all()
          assert np.all(z['exposure'].sum(axis=1)==steps*48)
          assert np.all(z['exposure'][:,list(set(range(len(meta)))-set(z['train_indices']))]==0)
          for group in ['old','new','validation']:
            ids=z['val_indices'] if group=='validation' else z['indices'][z['indices']<65 if group=='old' else z['indices']>=65]
            hit=None if group=='validation' else np.isin(z['indices'],ids)
            for metric in ['residual','image']:
              er=z['val_'+metric] if group=='validation' else z[metric][:,hit]
              for span,ss in [('full',slice(None)),('first05',slice(0,5)),('last05',slice(10,15))]:
                for weight in ['window','recording']:
                  values=aggregate(er[:,:,ss].mean(axis=2),ids,meta,weight=='recording')
                  curvegroups[(arm,group,metric,span,weight)].append(dict(fold=f['fold'],seed=seed,values=values.tolist(),selected_step=info['step']))
        expanded=arrays['expanded_long'];ref=expanded['residual'].mean(axis=2);e=ref[-1];v=ref[1:].var(axis=0)
        te=float(np.median(e));tv=float(np.median(v));t0=float(np.median(ref.var(axis=0)));tl=float(np.median(ref[6:].var(axis=0)))
        thresholds.append(dict(fold=f['fold'],seed=seed,error=te,variance=tv,variance_with0=t0,variance_late=tl))
        for arm,z in arrays.items():
          for j,i in enumerate(z['indices']):
            trained=int(i) in set(z['train_indices']);r=z['residual'][:,j].mean(axis=1);im=z['image'][:,j].mean(axis=1)
            var=float(r[1:].var());label=str(classify(r[-1],var,te,tv)) if trained else 'untrained_probe'
            label0=str(classify(r[-1],float(r.var()),te,t0)) if trained else 'untrained_probe'
            labell=str(classify(r[-1],float(r[6:].var()),te,tl)) if trained else 'untrained_probe'
            bestindex=int(infos[(f['fold'],seed,arm)]['step']//20)
            row=dict(fold=f['fold'],seed=seed,arm=arm,index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],source='old' if i<65 else 'new',trained=trained,
                final_residual_ADE=float(r[-1]),variance=var,group=label,group_with0=label0,group_late=labell,
                selected_residual_ADE=float(r[bestindex]),selected_image_ADE=float(im[bestindex]),final_image_ADE=float(im[-1]),
                early_residual_ADE=float(r[1]),late60_change=float(r[-1]-r[-4]),
                late60_slope=float(np.polyfit(steps[-4:],r[-4:],1)[0]),
                rebound_from_best=float(r[-1]-r[1:].min()),exposure240=int(z['exposure'][-1,i]),exposure_selected=int(z['exposure'][bestindex,i]),
                **features[i]['features'])
            rows.append(row)
          if arm=='expanded_long':
            for scope,hit in [('all',np.ones(len(z['indices']),bool)),('old',z['indices']<65),('new',z['indices']>=65)]:
              for feature in 'EDGC':
                x=np.array([features[i]['features'][feature] for i in z['indices']])[hit]
                corr.append(dict(fold=f['fold'],seed=seed,scope=scope,feature=feature,rho=finite_corr(x,e[hit])))
        old=arrays['original_long'];ids=old['indices'];hit=ids<65
        oldr=old['residual'].mean(axis=2);newr=expanded['residual'].mean(axis=2)
        oldlabels=classify(oldr[-1],oldr[1:].var(axis=0),te,tv);newlabels=classify(newr[-1],newr[1:].var(axis=0),te,tv)
        transitions.update(f'{a}->{b}' for a,b in zip(oldlabels[hit],newlabels[hit]))
        pair=dict(fold=f['fold'],seed=seed,old_train_final_delta=float((newr[-1,hit]-oldr[-1,hit]).mean()),
                  old_train_worse=int(np.sum(newr[-1,hit]>oldr[-1,hit])),old_train_count=int(hit.sum()),
                  new_probe_to_trained_final_delta=float((newr[-1,~hit]-oldr[-1,~hit]).mean()))
        paired.append(pair)
    write(OUT/'sample_dynamics.json',rows);write(OUT/'map_thresholds.json',thresholds)
    with (OUT/'sample_dynamics.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    serialcurves=[]
    for key,rr in curvegroups.items():
        byseed={str(s):np.mean([r['values'] for r in rr if r['seed']==s],axis=0).tolist() for s in [17,29,43]}
        serialcurves.append(dict(arm=key[0],group=key[1],metric=key[2],span=key[3],weight=key[4],
            mean=np.mean([r['values'] for r in rr],axis=0).tolist(),seed_means=byseed,
            selected_mean=float(np.mean([r['values'][r['selected_step']//20] for r in rr])),fits=rr))
    write(OUT/'learning_curves.json',dict(steps=steps.tolist(),curves=serialcurves))
    groups={}
    for source in ['old','new']:
        rr=[r for r in rows if r['arm']=='expanded_long' and r['source']==source]
        perfit=[]
        for f in folds:
          for seed in [17,29,43]:
            sub=[r for r in rr if r['fold']==f['fold'] and r['seed']==seed]
            perfit.append({g:sum(r['group']==g for r in sub)/len(sub) for g in LABELS})
        groups[source]=dict(fit_sample_instances=len(rr),unique_windows=len({r['index'] for r in rr}),
            mean_fit_proportions={g:avg([p[g] for p in perfit]) for g in LABELS},
            sensitivity_with0_disagreement=avg([r['group']!=r['group_with0'] for r in rr]),
            sensitivity_late_disagreement=avg([r['group']!=r['group_late'] for r in rr]))
    unique=[]
    for i,m in enumerate(meta):
        rr=[r for r in rows if r['arm']=='expanded_long' and r['index']==i]
        assert rr
        counts=Counter(r['group'] for r in rr)
        unique.append(dict(index=i,key=m['key'],recording=m['recording'],source=features[i]['source'],fits=len(rr),
            final_residual_ADE=avg([r['final_residual_ADE'] for r in rr]),variance=avg([r['variance'] for r in rr]),
            majority_group=counts.most_common(1)[0][0],majority_fraction=counts.most_common(1)[0][1]/len(rr),
            **features[i]['features']))
    write(OUT/'unique_window_map.json',unique)
    correlations={}
    for scope in ['all','old','new']:
        correlations[scope]={}
        for key in 'EDGC':
            values=[r['rho'] for r in corr if r['scope']==scope and r['feature']==key and r['rho'] is not None]
            correlations[scope][key]=dict(median=float(np.median(values)),range=[min(values),max(values)],positive_fits=sum(x>0 for x in values),n=len(values))
    exp=[r for r in rows if r['arm']=='expanded_long']
    summary=dict(groups=groups,paired=paired,old_training_quadrant_transitions=dict(transitions),correlations=correlations,
        old_training_worse_fits=sum(p['old_train_final_delta']>0 for p in paired),
        new_training_better_than_untrained_fits=sum(p['new_probe_to_trained_final_delta']<0 for p in paired),
        unique_window_majority_fraction_median=float(np.median([r['majority_fraction'] for r in unique])),
        expanded_late60_decreasing_fraction=avg([r['late60_slope']<0 for r in exp]),
        per_recording=[dict(recording=g,windows=len([r for r in unique if r['recording']==g]),final_residual_ADE=avg([r['final_residual_ADE'] for r in unique if r['recording']==g])) for g in sorted({r['recording'] for r in unique})],
        invalid_monitor_rays=sum(x['invalid_monitor_rays'] for x in verification['fits']),
        scope='Train-only fit-specific relative maps; repeated folds/seeds/windows not independent. No confidence/significance or label-quality causal claims.')
    write(OUT/'analysis_summary.json',summary)
    # Plot curve means by fit, seed envelope; original new curves are untrained probes.
    colors={'original_long':'#246A9B','expanded_long':'#E77732'}
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for col,(group,title) in enumerate([('old','两组共同的旧训练窗口'),('new','新增窗口：旧组未训练／扩充组训练'),('validation','独立验证录制')]):
      for row,(metric,label) in enumerate([('residual','行人残差 ADE / px'),('image','最终图像 ADE / px')]):
        ax=axes[row,col]
        for arm in colors:
            c=next(c for c in serialcurves if (c['arm'],c['group'],c['metric'],c['span'],c['weight'])==(arm,group,metric,'full','window'))
            a=np.array(list(c['seed_means'].values()));ax.plot(steps,c['mean'],color=colors[arm],label='原数据训练' if arm=='original_long' else '加入新 G1')
            ax.fill_between(steps,a.min(0),a.max(0),color=colors[arm],alpha=.12)
        ax.set_title(title);ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlabel('训练更新步数');ax.legend(fontsize=9)
    fig.suptitle('同一训练过程的逐窗口诊断｜各折等权平均，阴影为三种子的均值范围',fontsize=14)
    fig.tight_layout();fig.savefig(PRES/'learning_curves.png',dpi=170)
    for ax in axes.flat:ax.set_xlim(20,240)
    for col,group in enumerate(['old','new','validation']):
      for row,metric in enumerate(['residual','image']):
        vals=[]
        for c in serialcurves:
          if (c['group'],c['metric'],c['span'],c['weight'])==(group,metric,'full','window'):
            vals.extend(np.array(list(c['seed_means'].values()))[:,1:].ravel())
        lo,hi=min(vals),max(vals);axes[row,col].set_ylim(lo-(hi-lo)*.08,hi+(hi-lo)*.08)
    fig.suptitle('放大学习后段：步骤 20–240｜各折等权，阴影为三种子均值范围',fontsize=14)
    fig.savefig(PRES/'learning_curves_zoom.png',dpi=170);plt.close(fig)
    # One fold/seed map avoids conflating across-fit variances.
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True);t=thresholds[0]
    shown=[r for r in rows if r['fold']==0 and r['seed']==17 and r['trained']]
    xmin=min(r['variance'] for r in shown)*.7;xmax=max(r['variance'] for r in shown)*1.4
    ymin=min(r['final_residual_ADE'] for r in shown)*.7;ymax=max(r['final_residual_ADE'] for r in shown)*1.4
    for ax,arm in zip(axes,['original_long','expanded_long']):
        rr=[r for r in rows if r['fold']==0 and r['seed']==17 and r['arm']==arm and r['trained']]
        for source,color in [('old','#246A9B'),('new','#E77732')]:
            sub=[r for r in rr if r['source']==source]
            if sub:ax.scatter([r['variance'] for r in sub],[r['final_residual_ADE'] for r in sub],c=color,s=30,alpha=.7,label=f'{source} n={len(sub)}')
        ax.axvline(t['variance'],ls='--',c='gray');ax.axhline(t['error'],ls='--',c='gray')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
        ax.set_xlabel('步骤 20–240 的残差 ADE 方差 / px²');ax.set_ylabel('步骤 240 残差 ADE / px')
        for xx,yy,txt in [(.02,.94,'Hard'),(.77,.94,'Confusing'),(.02,.03,'Easy'),(.79,.03,'Trained')]:
            ax.text(xx,yy,txt,transform=ax.transAxes,fontsize=10,color='#555555',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        ax.set_title('原数据训练：仅训练中的旧窗口' if arm=='original_long' else '扩充训练：旧窗口＋新增窗口');ax.legend(loc='upper center');ax.grid(alpha=.2)
    fig.suptitle('数据地图示例：fold 0 / seed 17｜两图使用扩充训练池的同一中位数阈值',fontsize=13)
    fig.tight_layout();fig.savefig(PRES/'data_map.png',dpi=170);plt.close(fig)
    # Representative TRAINED samples in the expanded arm; selected posthoc for explanation.
    rr=[r for r in rows if r['fold']==0 and r['seed']==17 and r['arm']=='expanded_long']
    fig,axes=plt.subplots(2,2,figsize=(12,7));examples=[];z=cache[(0,17,'expanded_long')];zz=cache[(0,17,'original_long')]
    for ax,label in zip(axes.flat,LABELS):
        sub=[r for r in rr if r['group']==label]
        if not sub:ax.set_title(label+' (empty)');continue
        points=np.array([[np.log1p(r['variance']),np.log1p(r['final_residual_ADE'])] for r in sub]);center=np.median(points,axis=0)
        r=sub[int(np.argmin(((points-center)**2).sum(1)))];j=list(z['indices']).index(r['index'])
        ax.plot(steps,z['residual'][:,j].mean(1),'-o',color=colors['expanded_long'],label='扩充训练')
        ax.plot(steps,zz['residual'][:,j].mean(1),'--',color=colors['original_long'],label='原组训练' if r['source']=='old' else '原组未训练探针')
        ax.set_title(f'{label} | {r["source"]} | {r["key"]}',fontsize=9);ax.set_xlabel('更新步数');ax.set_ylabel('残差 ADE / px');ax.legend(fontsize=8);ax.grid(alpha=.2);examples.append(r)
    fig.suptitle('四类样本的实际学习曲线｜训练集内事后代表案例，不用于筛选数据')
    fig.tight_layout();fig.savefig(PRES/'example_curves.png',dpi=170);plt.close(fig);write(OUT/'example_windows.json',examples)
    fig,axes=plt.subplots(2,2,figsize=(12,8));flabels={'E':'E：log(框高) 增长率 / s','D':'D：框高 / 图像高','G':'G：补偿后纵向速度 / 图像高 / s','C':'C：相机平移速度 / m/s'}
    for ax,key in zip(axes.flat,'EDGC'):
        for source,color in [('old','#246A9B'),('new','#E77732')]:
            sub=[r for r in unique if r['source']==source];ax.scatter([r[key] for r in sub],[r['final_residual_ADE'] for r in sub],s=22,alpha=.6,color=color,label=source)
        ax.set_xlabel(flabels[key]);ax.set_ylabel('训练末残差 ADE / px');ax.set_yscale('log');ax.set_title(f'各训练折 Spearman 中位数 = {correlations["all"][key]["median"]:.2f}');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('EDGC 与学习难度｜每个唯一窗口一颗点，误差对训练它的折与种子取均值')
    fig.tight_layout();fig.savefig(PRES/'edgc_map.png',dpi=170);plt.close(fig)
    # Primary class proportions, equal fit weighting.
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(4)
    ax.bar(x-.18,[groups['old']['mean_fit_proportions'][g]*100 for g in LABELS],.36,label='旧窗口',color=colors['original_long'])
    ax.bar(x+.18,[groups['new']['mean_fit_proportions'][g]*100 for g in LABELS],.36,label='新窗口',color=colors['expanded_long'])
    ax.set_xticks(x,LABELS);ax.set_ylabel('每折／种子组内占比的平均值 %');ax.set_title('扩充训练中的相对难度分组（中位数分组，不是质量标签）');ax.legend();fig.tight_layout();fig.savefig(PRES/'group_proportions.png',dpi=170);plt.close(fig)
    write(OUT/'analysis_verification.json',dict(passed=True,fit_records=len(rows),unique_windows=len(unique),
        no_test_or_validation_in_maps=True,exposure_sums_exact=True,figures=6,
        source_files_unchanged=all(sha(ROOT/p)==h for p,h in read(OUT/'source_hashes.json').items())))
    print(json.dumps(dict(groups=groups,paired=paired,correlations=correlations),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
