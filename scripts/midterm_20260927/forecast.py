"""Causal RGB-only pilot. Targets are assistant-prechecked silver, not gold.

Past-only functions accept exactly ten historical points/transforms. Future RGB
is ONLY a training-target transform or a separately named offline diagnostic.
"""
from pathlib import Path
import cv2
import numpy as np
from scipy.linalg import logm,expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,CONFIG,read_json,write_json,apply_h,encoder,finish_encoder
from review_pack import runs

HISTORY=10;FUTURE=20;N=np.diag([1/2560,1/1920,1.]);NI=np.linalg.inv(N)
METHODS=['stationary','raw_cv','rgb_factor_freeze','rgb_factor_cv','ridge_entangled','ridge_factorized','offline_future_rgb']

def cumulative(Hs):
    result=[];H=np.eye(3)
    for s in Hs:H=s@H;result.append(H.copy())
    return np.array(result)

def historical(p,Hs):
    assert p.shape==(HISTORY,2) and Hs.shape==(HISTORY,3,3)
    q=np.empty_like(p);C=np.eye(3);q[-1]=p[-1]
    for i in range(HISTORY-2,-1,-1):C=C@Hs[i+1];q[i]=apply_h([p[i]],C)[0]
    return q

def camera_trend(Hs):
    logs=[]
    for H in Hs[-3:]:
        z=logm(H)
        if np.max(np.abs(z.imag))>1e-6 or not np.isfinite(z).all():z=np.zeros((3,3))
        logs.append(z.real)
    rate=np.mean(logs,axis=0)
    return np.array([expm(rate*h) for h in range(1,FUTURE+1)])

def slope(p):
    t=np.arange(len(p),dtype=float);t-=t.mean()
    return t@(p-p.mean(axis=0))/(t@t)

def features(p,Hs,factor=False):
    q=historical(p,Hs) if factor else p
    # Identical information, dimension and ridge parameter count for both heads.
    return np.r_[(q-p[-1]).ravel(),p[-1],(Hs[1:]-np.eye(3)).reshape(9,9)[:,:8].ravel()]

def safe_project(p,H):
    q=apply_h([p],H)[0]
    # Undefined projective forecast gets a fixed large error, never silently removed.
    return q if np.isfinite(q).all() else np.array([10.,10.])

def predict_past_only(p,Hs,models=None):
    assert len(p)==HISTORY and len(Hs)==HISTORY
    horizons=np.arange(1,FUTURE+1)[:,None];q=historical(p,Hs);futureH=camera_trend(Hs)
    ped=p[-1]+horizons*slope(q)
    result={'stationary':np.repeat(p[-1][None],FUTURE,axis=0),'raw_cv':p[-1]+horizons*slope(p),
            'rgb_factor_freeze':ped,'rgb_factor_cv':np.array([safe_project(x,H) for x,H in zip(ped,futureH)])}
    if models:
        for key,factor in [('ridge_entangled',False),('ridge_factorized',True)]:
            m=models[key];x=features(p,Hs,factor);y=((x-m['mean'])/m['std'])@m['coef']+m['bias'];pred=p[-1]+y.reshape(FUTURE,2)
            result[key]=np.array([safe_project(x,H) for x,H in zip(pred,futureH)]) if factor else pred
    return result

def windows(clip,tracks,stride):
    geo=np.load(clip/'geometry/transforms.npz');hs=geo['rgb_homography'];hs=N@hs@NI
    for t in tracks:
        for rs in runs(t['boxes']):
            for k in range(HISTORY-1,len(rs)-FUTURE,stride):
                chunk=rs[k-HISTORY+1:k+FUTURE+1];fs=[r['frame_id'] for r in chunk];f=rs[k]['frame_id']
                b=np.array([r['bbox_xyxy'] for r in chunk]);points=(b[:,:2]+b[:,2:])/2/np.array([2560,1920])
                hh=hs[fs[:HISTORY]].copy();hh[0]=np.eye(3)
                hist=points[:HISTORY];target=points[HISTORY:]
                if not np.isfinite(historical(hist,hh)).all():continue
                yield {'track':t.get('track_id',t.get('id')),'frame':f,'history':hist,'Hs':hh,'target':target,
                       'futureH':cumulative(hs[fs[HISTORY:]]),'history_invalid_pairs':int(np.all(np.isclose(hs[fs[1:HISTORY]],np.eye(3)),axis=(1,2)).sum()),
                       'future_invalid_pairs':int(np.all(np.isclose(hs[fs[HISTORY:]],np.eye(3)),axis=(1,2)).sum()),
                       'edge_truncated':bool(((b[:,0]<5)|(b[:,1]<5)|(b[:,2]>2555)|(b[:,3]>1915)).any())}

def fit_model(X,Y):
    X=np.array(X);Y=np.array(Y);mean=X.mean(0);std=X.std(0);std[std<1e-5]=1
    Z=(X-mean)/std;bias=Y.mean(0);alpha=10.
    coef=np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]),Z.T@(Y-bias))
    return dict(mean=mean,std=std,coef=coef,bias=bias,alpha=alpha)

def test_contract(models):
    rng=np.random.default_rng(9);p=np.cumsum(rng.normal(size=(10,2))*.005,axis=0)+.5;hs=np.repeat(np.eye(3)[None],10,0)
    result=predict_past_only(p,hs,models)
    identity=float(np.max(np.abs(result['raw_cv']-result['rgb_factor_cv'])));assert identity<1e-10
    # Wrapper always slices at cutoff; arbitrary future suffix cannot affect headline models.
    allp=np.r_[p,rng.normal(size=(20,2))];allh=np.r_[hs,np.repeat(np.eye(3)[None],20,0)]
    a=predict_past_only(allp[:10],allh[:10],models);allp[10:]=1e9;allh[10:]=1e9
    b=predict_past_only(allp[:10],allh[:10],models);leak=max(float(np.max(abs(a[m]-b[m]))) for m in a);assert leak==0
    return {'identity_compensation_equals_raw_cv':identity,'future_suffix_mutation_max_change':leak,
            'api':'all headline methods receive only 10 historic centers and RGB transforms; no MPS arrays are loaded by forecasting code'}

def main():
    specs=read_json(CONFIG)['clips'];out=OUT/'results/forecast';out.mkdir(exist_ok=True)
    X={m:[] for m in ['ridge_entangled','ridge_factorized']};Y={m:[] for m in X};trainrows=[]
    for spec in specs:
        if spec['split']!='development':continue
        clip=OUT/'clips'/spec['id'];tracks=read_json(clip/'yolo/tracks.json')['tracks']
        for row in windows(clip,tracks,1):
            p,H=row['history'],row['Hs'];C=row['futureH'];target=row['target']
            transformed=np.array([safe_project(q,np.linalg.inv(h)) for q,h in zip(target,C)])
            for m,factor in [('ridge_entangled',False),('ridge_factorized',True)]:
                X[m].append(features(p,H,factor));Y[m].append(((transformed if factor else target)-p[-1]).ravel())
            trainrows.append({'clip':spec['id'],'track':row['track'],'frame':row['frame']})
    assert trainrows,'No training examples';models={m:fit_model(X[m],Y[m]) for m in X}
    np.savez_compressed(out/'models.npz',**{m+'_'+k:v for m,d in models.items() for k,v in d.items()})
    write_json(out/'training_windows.json',trainrows);tests=test_contract(models)
    rows=[];examples=[];coverage=[]
    for spec in specs:
        clip=OUT/'clips'/spec['id'];tracks=read_json(clip/'annotation_v1/tracks.json')['tracks'];count=0
        for r in windows(clip,tracks,5):
            pred=predict_past_only(r['history'],r['Hs'],models)
            ped=pred['rgb_factor_freeze'];pred['offline_future_rgb']=np.array([safe_project(q,h) for q,h in zip(ped,r['futureH'])])
            entry={'clip':spec['id'],'recording':spec['recording'],'split':spec['split'],'track':r['track'],'frame':r['frame'],
                   'history_invalid_pairs':r['history_invalid_pairs'],'future_invalid_pairs':r['future_invalid_pairs'],'edge_truncated':r['edge_truncated']}
            for m,pts in pred.items():
                err=np.linalg.norm((pts-r['target'])*np.array([2560,1920]),axis=1)
                for h in [5,10,20]:entry[f'{m}_ADE{h}']=float(err[:h].mean());entry[f'{m}_FDE{h}']=float(err[h-1])
            rows.append(entry);count+=1
            examples.append(dict(entry,history=(r['history']*[2560,1920]).tolist(),target=(r['target']*[2560,1920]).tolist(),predictions={m:(p*[2560,1920]).tolist() for m,p in pred.items()}))
        coverage.append({'clip':spec['id'],'windows':count,'review_tracks':len(tracks)})
    write_json(out/'windows.json',rows);write_json(out/'predictions.json',examples)
    summaries=[]
    for spec in specs:
        subset=[r for r in rows if r['clip']==spec['id']]
        if not subset:continue
        summaries.append({'spec':spec,'windows':len(subset),'methods':{m:{f'{metric}{h}':float(np.mean([r[f'{m}_{metric}{h}'] for r in subset])) for metric in ['ADE','FDE'] for h in [5,10,20]} for m in METHODS}})
    summary={'training_windows':len(trainrows),'training_recordings':sorted({s['recording'] for s in specs if s['split']=='development'}),
      'train_labels':'unreviewed automatic YOLO proposals, weak/noisy supervision','evaluation_labels':'assistant visually prechecked partial labels; NOT user-confirmed gold',
      'evaluation_stride':5,'history_frames':10,'future_frames':20,'ridge_alpha_fixed':10,'parameters_per_ridge':int(models['ridge_entangled']['coef'].size+40),
      'feature_dimensions':len(X['ridge_entangled'][0]),'coverage':coverage,'clips':summaries,'tests':tests,
      'limitations':['very small noisy training set; not a competitive learned model','selected long surviving objects bias evaluation toward easy trajectories','forecast errors include automatic box noise','invalid background estimates use identity and are counted, not dropped','offline_future_rgb consumes future images and is excluded from RGB-only headline claims','simple constant log-homography extrapolation can fail during head turns','no final generalization or improvement claim until user review and more recordings']}
    write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');test=[s for s in summaries if s['spec']['split']=='test'];xx=np.arange(len(test))
    for ax,h in zip(axes,[10,20]):
        for m in METHODS:ax.plot(xx,[s['methods'][m][f'ADE{h}'] for s in test],marker='o',label=m)
        ax.set_xticks(xx,[s['spec']['id'] for s in test],rotation=30,ha='right');ax.set_title(f'{h/10:g}s center ADE, partial silver');ax.set_ylabel('pixels at 2560x1920');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(out/'forecast_comparison.png',dpi=170);plt.close(fig)
    # Qualitative success AND failure, chosen post-hoc and explicitly labeled.
    evals=[e for e in examples if e['split']=='test'];rank=sorted(evals,key=lambda e:e['raw_cv_ADE20']-e['rgb_factor_cv_ADE20'])
    picks=[('failure',rank[0]),('supporting',rank[-1])] if rank else []
    for tag,e in picks:
        cap=cv2.VideoCapture(str(OUT/'clips'/e['clip']/'rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame']);ok,im=cap.read();assert ok;cap.release();im=cv2.resize(im,(1280,960))
        colors={'reference':(60,255,70),'raw_cv':(50,90,255),'rgb_factor_cv':(255,220,20),'offline_future_rgb':(220,90,240)}
        curves={'reference':e['target'],**{k:e['predictions'][k] for k in colors if k!='reference'}}
        for m,pts in curves.items():
            pp=np.array(pts)/2;pp=np.clip(pp,-10000,10000).astype(np.int32);cv2.polylines(im,[pp.reshape(-1,1,2)],False,colors[m],3)
        cv2.rectangle(im,(0,0),(1280,95),(0,0,0),-1)
        cv2.putText(im,f'POST-HOC {tag}: {e["clip"]} P{e["track"]} f={e["frame"]}',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(im,'2s future image coordinates on CURRENT frame (not world paths)',(10,46),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.putText(im,'Green=silver ref | red=raw CV | cyan=RGB factor | purple=FUTURE RGB diagnostic',(10,70),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        cv2.putText(im,'Selection illustrates mechanism; not an unbiased improvement estimate.',(10,91),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
        cv2.imwrite(str(out/f'case_{tag}.jpg'),im);write_json(out/f'case_{tag}.json',e)
    print('FORECAST',len(trainrows),'train',len(rows),'eval',summary['coverage'],flush=True)
    for s in summaries:print(s['spec']['id'],{m:round(v['ADE20'],2) for m,v in s['methods'].items()},flush=True)

if __name__=='__main__':main()
