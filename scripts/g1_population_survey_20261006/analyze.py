"""Probability-sampled, response-conditional G1 distribution; no forecast scores."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/g1_population_survey_20261006'
ORDER='EDGC'
LABELS=['E 人物尺度增长率','D 人物框相对高度','G 补偿后纵向速度','C 相机平移速度']
UNITS=['log(框高) / 秒','图像高度 %','图像高度 % / 秒，向下为正','米 / 秒']
SCALE=np.array([1.,100.,100.,1.])
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),'utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def features(boxes,cal):
    b=np.array(boxes,float)[:6];t=(cal['times_ns'][:6]-cal['times_ns'][0])/1e9
    tc=t-t.mean();R=cal['R'][:6];K=cal['K'];h=b[:,3]-b[:,1]
    xy=(b[:,:2]+b[:,2:])/2
    H=K@R[5].T@R@np.linalg.inv(K)
    p=np.einsum('nij,nj->ni',H,np.c_[xy,np.ones(6)]);p=p[:,:2]/p[:,2:]
    return dict(E=float(tc@np.log(h)/(tc@tc)),D=float(np.median(h)/1920),
                G=float(tc@p[:,1]/(tc@tc)/1920),
                C=float(np.linalg.norm(np.diff(cal['position'][:6],axis=0),axis=1).sum()/(t[-1]-t[0])))

def quantile(x,w):
    order=np.argsort(x);cdf=np.cumsum(w[order])/w.sum()
    return x[order[np.minimum(np.searchsorted(cdf,[.1,.5,.9]),len(order)-1)]].tolist()

def summarize(rows,weights,thresholds,field='features'):
    w=np.array(weights,float);w/=w.sum();x=np.array([[r[field][k] for k in ORDER] for r in rows])
    levels=x>=np.array([thresholds[k] for k in ORDER]);center=x-w@x
    cov=center.T@(center*w[:,None]);corr=cov/np.sqrt(np.outer(np.diag(cov),np.diag(cov)))
    return dict(n=len(rows),recordings=len({r['recording'] for r in rows}),
        features={k:dict(mean=float(w@x[:,j]),p10_p50_p90=quantile(x[:,j],w),
                        high_share=float(w@levels[:,j]),threshold=thresholds[k]) for j,k in enumerate(ORDER)},
        correlation=corr.tolist(),cells=[dict(code=''.join(map(str,bits)),n=int(np.all(levels==bits,axis=1).sum()),
             share=float(w@np.all(levels==bits,axis=1))) for bits in product([0,1],repeat=4)],
        below_camera_0p2=float(w@(x[:,3]<.2)),camera_1_to_1p6=float(w@((x[:,3]>=1)&(x[:,3]<=1.6))))

def main():
    design=read(OUT/'protocol_locked.json');samples=read(OUT/'sample_locked.json');strata=read(OUT/'strata.json')
    for f,h in read(OUT/'design_manifest.json').items():assert sha(OUT/f)==h,f
    rng=np.random.default_rng(design['seed'])
    for s in strata:
        assert sorted(rng.choice(s['keys'],size=s['n'],replace=False).tolist())==s['chosen']
        assert s['n']>0 and s['n']<=s['N'] and np.isclose(s['pi'],s['n']/s['N'])
    assert len(samples)==160 and len({s['key'] for s in samples})==160
    assert np.isclose(sum(s['survey_weight'] for s in samples),835)
    review=read(OUT/'visual_review.json');review_hash=sha(OUT/'visual_review.json')
    assert review['context_pages_reviewed']==list(range(1,33))
    reviewed=set(review['mask_and_temporal_reviewed_keys'])
    rows=[];ledger=[];windows=[];all_reasons=Counter();protected=read(ROOT/'outputs/g1_edgc_training_distribution_20261006/distribution.json')['audit']['source_hashes']
    for f,h in protected.items():assert sha(ROOT/f)==h,f
    for sample in samples:
        sid=sample['sample_id'];d=OUT/'samples'/sid;cal=dict(np.load(d/'reference.npz'))
        proposals=read(d/'proposals.json');seg=read(d/'segmentation.json')
        assert len(list(d.glob('f[0-9][0-9].jpg')))==21 and len(cal['R'])==21
        win=dict(sample_id=sid,recording=sample['recording'],weight=sample['survey_weight'],
                 proposals=len(proposals),automatic_eligible=sum(t['automatic_eligible'] for t in proposals),accepted=0,
                 global_exclusion=review['global_exclusions'].get(sid),visually_empty=sid in review['empty_contexts'])
        for tr in proposals:
            key=tr['key'];rec=dict(key=key,sample_id=sid,recording=sample['recording'],weight=sample['survey_weight'],
                automatic_eligible=tr['automatic_eligible'],automatic_reasons=tr['automatic_reasons'],min_detector_height=tr['min_height'])
            all_reasons.update(tr['automatic_reasons'])
            if win['global_exclusion']:rec.update(status='outside_visual_domain',reason=win['global_exclusion'])
            elif not tr['automatic_eligible']:rec.update(status='automatic_incomplete_or_outside_measurement',reason=';'.join(tr['automatic_reasons']))
            elif key in review['target_exclusions']:rec.update(status='visual_target_exclusion',reason=review['target_exclusions'][key])
            else:
                assert key in reviewed,key
                history=sorted(seg[key],key=lambda r:r['frame']);assert [x['frame'] for x in history]==list(range(6))
                if key in review['measurement_exclusions']:rec.update(status='mask_measurement_failure',reason=review['measurement_exclusions'][key])
                elif not all(x['quality']['valid'] for x in history):rec.update(status='mask_measurement_failure',reason='mask_guide_quality')
                else:
                    b=np.array([x['quality']['box'] for x in history]);height=(b[:,3]-b[:,1])/1920
                    if min(height)<.04:rec.update(status='mask_measurement_failure',reason='mask_height_below4percent')
                    else:
                        rec.update(status='measured_G1',reason='')
                        detector_b=np.array([x['box'] for x in sorted(tr['rows'],key=lambda r:r['frame'])])
                        f=features(b,cal);fd=features(detector_b,cal)
                        # Future suffix cannot change a historical descriptor.
                        mutation={k:v.copy() for k,v in cal.items()};mutation['R'][6:]=np.eye(3);mutation['position'][6:]+=100
                        assert features(b,mutation)==f
                        rows.append(dict(key=key,sample_id=sid,recording=sample['recording'],stratum=sample['stratum'],
                            start_s=sample['start_s'],survey_weight=sample['survey_weight'],inclusion_probability=sample['inclusion_probability'],
                            features=f,detector_features=fd,near=bool(tr['near_candidate'] and min(height)>=.08),
                            flags=[flag for flag,keys in review['sensitivity_flags'].items() if key in keys],
                            six_mask_boxes=b.tolist(),source=sample['video']))
                        win['accepted']+=1
            ledger.append(rec)
        windows.append(win)
    write(OUT/'response_ledger.json',ledger);write(OUT/'window_response.json',windows);write(OUT/'measured_features.json',rows)
    thresholds=design['thresholds'];w=np.array([r['survey_weight'] for r in rows]);counts=Counter(r['sample_id'] for r in rows)
    near=[r for r in rows if r['near']];train=[r for r in read(ROOT/'outputs/g1_edgc_training_distribution_20261006/window_distribution.json') if r['expected_sampling_probability']>0]
    assert len(train)==295
    strict=[r for r in train if r['prior_inventory_eligible']];clean=[r for r in rows if not r['flags']]
    groups={
        'survey_all':summarize(rows,w,thresholds),
        'survey_near':summarize(near,[r['survey_weight'] for r in near],thresholds),
        'survey_time_equal':summarize(rows,[r['survey_weight']/counts[r['sample_id']] for r in rows],thresholds),
        'survey_without_flagged_controls':summarize(clean,[r['survey_weight'] for r in clean],thresholds),
        'survey_unweighted':summarize(rows,np.ones(len(rows)),thresholds),
        'survey_detector_sensitivity':summarize(rows,w,thresholds,'detector_features'),
        'train_source':summarize(train,np.ones(len(train)),thresholds),
        'train_expected':summarize(train,[r['expected_sampling_probability'] for r in train],thresholds),
        'train_strict_G1_source':summarize(strict,np.ones(len(strict)),thresholds)}
    # Recording-cluster resampling: sensitivity intervals, not exact stratified-design CIs.
    recs=sorted({s['recording'] for s in samples});rng=np.random.default_rng(2026100618)
    boot={name:[] for name in ['survey_all','survey_near']}
    for _ in range(2000):
        draws=Counter(rng.choice(recs,len(recs),replace=True))
        for name,rr in [('survey_all',rows),('survey_near',near)]:
            bw=np.array([r['survey_weight']*draws[r['recording']] for r in rr])
            if bw.sum()==0:continue
            x=np.array([[r['features'][k] for k in ORDER] for r in rr]);bw=bw/bw.sum()
            boot[name].append([*(bw@x),*(bw@(x>=np.array([thresholds[k] for k in ORDER])))])
    for name,values in boot.items():
        interval=np.quantile(values,[.025,.975],axis=0)
        for j,k in enumerate(ORDER):
            groups[name]['features'][k]['recording_bootstrap_mean_interval']=interval[:,j].tolist()
            groups[name]['features'][k]['recording_bootstrap_high_share_interval']=interval[:,j+4].tolist()
    rec_stats=[]
    for recording in recs:
        sr=[r for r in rows if r['recording']==recording];ss=[s for s in samples if s['recording']==recording]
        rec_stats.append(dict(recording=recording,sampled_windows=len(ss),measured_person_windows=len(sr),
            weighted_person_share=sum(r['survey_weight'] for r in sr)/sum(w),frame_N=sum(s['N'] for s in strata if s['id']==recording.split('_')[-1])))
    summary=dict(design=design,feature_order=ORDER,groups=groups,
        response=dict(sampled_windows=len(samples),measured_windows=len(counts),measured_recordings=len({r['recording'] for r in rows}),
            measured_person_windows=len(rows),near_person_windows=len(near),cutoff_proposals=len(ledger),
            automatic_eligible=sum(r['automatic_eligible'] for r in ledger),mutually_exclusive_status=dict(Counter(r['status'] for r in ledger)),
            automatic_reasons_nonexclusive=dict(all_reasons),globally_excluded_windows=sum(bool(s['global_exclusion']) for s in windows),
            empty_contexts=sum(s['visually_empty'] for s in windows),zero_measured_windows=sum(s['accepted']==0 for s in windows)),
        recording_statistics=rec_stats,uncertainty_note='2000 recording-cluster resamples; conditional on measurable targets. Not an exact design CI, not correction for missed targets.',
        audit=dict(locked_hashes_intact=True,random_seed_replayed=True,all_strata_positive_probability=True,
            time_window_weight_sum=float(sum(s['survey_weight'] for s in samples)),historical_suffix_invariant=True,
            visual_review_sha256=review_hash,source_hashes_intact=True,source_hashes=protected))
    write(OUT/'distribution.json',summary)
    # ECDF retains the distribution shape; no bins are balanced by hand.
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']='Microsoft YaHei';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(12.4,8.1),layout='constrained')
    colors=['#2166ac','#1b9e77','#d97706']
    sets=[(rows,w,'G1 随机抽样 · 概率加权'),(near,np.array([r['survey_weight'] for r in near]),'其中近人 · 概率加权'),
          (train,np.array([r['expected_sampling_probability'] for r in train]),'当前训练 · 采样器期望')]
    for j,ax in enumerate(axes.ravel()):
        for (rr,ww,label),color in zip(sets,colors):
            v=np.array([r['features'][ORDER[j]] for r in rr])*SCALE[j];idx=np.argsort(v)
            ax.step(v[idx],np.cumsum(ww[idx])/ww.sum()*100,where='post',label=label,color=color,lw=2)
        ax.axvline(thresholds[ORDER[j]]*SCALE[j],color='#667085',ls=':',lw=1.2,label='既有高低阈值')
        ax.set_title(LABELS[j],loc='left',fontweight='bold');ax.set_xlabel(UNITS[j]);ax.set_ylabel('累计占比 %');ax.set_ylim(0,101);ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=9,loc='lower right')
    fig.suptitle(f'G1 分布抽样调查｜{len(samples)} 个随机时间窗 → {len(rows)} 个人物窗口（近人 {len(near)}）',fontsize=15)
    fig.savefig(OUT/'distribution_ecdf.png',dpi=170);fig.savefig(OUT/'distribution_ecdf.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.8,5.7),layout='constrained');pos=np.arange(4)
    for i,(name,label,color) in enumerate(zip(['survey_all','survey_near','train_expected'],['G1 抽样','G1 抽样中的近人','当前训练采样器期望'],colors)):
        vals=np.array([groups[name]['features'][k]['high_share']*100 for k in ORDER])
        bars=ax.bar(pos+(i-1)*.25,vals,.24,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in vals],padding=3,fontsize=10)
    ax.set_xticks(pos,LABELS);ax.set_ylim(0,100);ax.set_ylabel('超过既有阈值的占比 %');ax.legend(loc='upper left',ncol=3,fontsize=10)
    ax.set_title('沿用旧阈值比较分布，不重新切成一半一半',loc='left',fontsize=15);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(OUT/'high_level_comparison.png',dpi=170);fig.savefig(OUT/'high_level_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(response=summary['response'],groups={k:v for k,v in groups.items() if k in ['survey_all','survey_near','train_expected']}),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
