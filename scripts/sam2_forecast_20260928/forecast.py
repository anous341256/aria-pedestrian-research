"""Frozen PIE B0 checkpoint transferred to selected SAM2 tracks, past-only.

No model training, future RGB, Aria pose, IMU, gaze, or vehicle speed is used.
The official pinned Mamba class runs with its official pure-Torch scan reference.
"""
import ast, hashlib, math, sys
from pathlib import Path
from typing import Optional
sys.path.insert(0,str(Path(__file__).with_name('vendor')/'pythonlib'))
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from prepare import ROOT, SRC, OUT, read, write

SCALE=np.array([2560.,1920.,2560.,1920.],dtype=np.float32)
VENDOR=Path(__file__).with_name('vendor')
CHECKPOINT=ROOT/'models/controlled_baselines/b0_zeroego_seed1/best.pt'
METHODS=['stationary','linear','cvcs','mamba_b0']

def load_official_mamba():
    env=dict(torch=torch,nn=nn,F=F,Tensor=Tensor,Optional=Optional,math=math,
        rearrange=rearrange,repeat=repeat,causal_conv1d_fn=None,
        causal_conv1d_update=None,selective_state_update=None)
    # Do not import CUDA extensions; execute the exact official AST definitions.
    for filename,name in [('selective_scan_interface.py','selective_scan_ref'),('mamba_simple.py','Mamba')]:
        path=VENDOR/filename
        manifest=read(VENDOR/'manifest.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][filename]['sha256']
        tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name]
        assert len(nodes)==1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        if name=='selective_scan_ref':env['selective_scan_fn']=env[name]
    return env['Mamba']

def load_model():
    Mamba=load_official_mamba()
    sys.path.insert(0,str(ROOT/'TrajMamba-R0'))
    import trajmamba.model as module
    def factory(backend,d_model,num_layers=1,**kwargs):
        assert backend=='mamba' and num_layers==1
        return Mamba(d_model=d_model,use_fast_path=False,**kwargs)
    module.make_sequence_model=factory
    ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert ckpt['config']['data']['ego_input_mode']=='zero'
    cfg=module.TrajMambaConfig(obs_len=15,pred_len=45,**ckpt['config']['model'])
    model=module.TrajMambaR0(cfg).eval()
    model.load_state_dict(ckpt['model'],strict=True)
    return model,ckpt

def resample_history(boxes):
    """Six real 10Hz observations t-0.5,...,t -> 15 past-only 30Hz tokens."""
    assert boxes.shape==(6,4)
    xywh=np.c_[(boxes[:,:2]+boxes[:,2:])/2,boxes[:,2:]-boxes[:,:2]]/SCALE
    raw_t=np.arange(-5,1)/10
    token_t=np.arange(-14,1)/30
    return np.stack([np.interp(token_t,raw_t,xywh[:,j]) for j in range(4)],axis=-1).astype(np.float32)

@torch.inference_mode()
def predict(model,past):
    """Only accepts historical boxes, no target argument or global annotation access."""
    b=torch.from_numpy(np.stack([resample_history(x) for x in past]))
    out=model(b,torch.zeros(len(b),15,1))
    times=np.arange(15,dtype=np.float32);times-=times.mean()
    slopes=np.einsum('t,btd->bd',times,b.numpy())/(times@times)
    tau=np.arange(1,46,dtype=np.float32)[None,:,None]
    last=b.numpy()[:,-1:, :]
    result={'stationary':np.repeat(last,45,axis=1),
            'linear':last+tau*slopes[:,None,:],
            'cvcs':out['reference'].numpy(),
            'mamba_b0':out['pred_boxes'].numpy()}
    # Score only original 10Hz timestamps, not invented future interpolation.
    return {m:values[:,2::3,:]*SCALE for m,values in result.items()}

@torch.inference_mode()
def verify_backend(model):
    torch.manual_seed(928)
    errs={}
    # Independent full-sequence scan vs token-by-token official recurrence.
    for name in ['pedestrian_encoder','ego_encoder','ego_guided_decoder']:
        m=getattr(model,name);x=torch.randn(2,60,256)*.1
        a=m(x);conv,ssm=m.allocate_inference_cache(2,60)
        ys=[]
        for t in range(60):
            y,conv,ssm=m.step(x[:,t:t+1],conv,ssm);ys.append(y)
        err=float((a-torch.cat(ys,dim=1)).abs().max())
        assert err<1e-5,(name,err)
        errs[name]=err
    return errs

def main():
    torch.set_num_threads(4)
    specs=read(OUT/'candidates.json')
    # Frozen before model inference; visual review does not alter raw labels.
    protocol=dict(history_source_frames=6,history_source_span_seconds=.5,
        network_history_tokens=15,network_history_span_seconds=14/30,
        prediction_seconds=1.5,prediction_scored_frames=15,source_fps=10,
        network_fps=30,metric_stride_frames=5,nonoverlap_stride_frames=20,
        selection='Four continuous candidate tracks selected by length, scene diversity, and identity sampling BEFORE prediction errors.',
        reference='Raw visible SAM2 mask envelopes, assistant sampled identity review; no user-confirmed gold.',
        edge_definition='Any source history or target box within 5px of an image boundary.',
        adaptation='Normalize xywh by 2560x1920. Past-only linear interpolation 10Hz->30Hz; retain predicted steps 3,6,...,45. No future target interpolation.',
        methods=METHODS,training='None in this run; frozen B0 checkpoint pretrained on PIE.',
        same_information='All methods receive the same 15 past bbox tokens. No images enter these predictors.',
        user_authorization='2026-09-28: use several long SAM annotations for a prediction demonstration.',
        representative_frames={s['key']:s['start']+min(50,(s['end']-s['start'])//2) for s in specs})
    write(OUT/'protocol.json',protocol)
    model,ckpt=load_model();backend=verify_backend(model)
    allrows=[]
    for spec in specs:
        p=SRC/spec['recording']/'sam2_raw_v1/tracks.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['source_annotation_sha256']
        tr=next(t for t in read(p)['tracks'] if t['id']==spec['track'])
        byframe={r['frame']:r['bbox_xyxy'] for r in tr['frames'] if r.get('bbox_xyxy')}
        frames=list(range(spec['start']+5,spec['end']-15+1))
        histories=np.array([[byframe[g] for g in range(f-5,f+1)] for f in frames],dtype=np.float32)
        predictions={m:[] for m in METHODS}
        for i in range(0,len(histories),32):
            pred=predict(model,histories[i:i+32])
            for m in METHODS:predictions[m].extend(pred[m])
        rows=[]
        for i,f in enumerate(frames):
            target=np.array([byframe[g] for g in range(f+1,f+16)],dtype=np.float32)
            targetxy=np.c_[(target[:,:2]+target[:,2:])/2,target[:,2:]-target[:,:2]]
            bb=np.r_[histories[i],target]
            edge=bool(np.any((bb[:,0]<=5)|(bb[:,1]<=5)|(bb[:,2]>=2555)|(bb[:,3]>=1915)))
            row=dict(key=spec['key'],recording=spec['recording'],track=spec['track'],frame=f,
                evaluated=(f-frames[0])%5==0,nonoverlap=(f-frames[0])%20==0,
                edge_truncated=edge,history_xyxy=histories[i].tolist(),target_xywh=targetxy.tolist(),
                predictions={},metrics={})
            for m in METHODS:
                v=np.array(predictions[m][i]);assert v.shape==(15,4) and np.isfinite(v).all()
                err=np.linalg.norm(v[:,:2]-targetxy[:,:2],axis=1)
                row['predictions'][m]=v.tolist()
                row['metrics'][m]={f'{metric}_{h/10:.1f}s':float(err[:h].mean() if metric=='ADE' else err[h-1])
                    for metric in ['ADE','FDE'] for h in [5,10,15]}
                row['metrics'][m]['invalid_size_steps']=int((v[:,2:]<=0).any(axis=1).sum())
                row['metrics'][m]['offscreen_center_steps']=int(((v[:,0]<0)|(v[:,0]>2560)|(v[:,1]<0)|(v[:,1]>1920)).sum())
            rows.append(row)
        write(OUT/f'{spec["key"]}_predictions.json',rows)
        allrows.extend(rows)
        print(spec['key'],len(rows),'rolling predictions',sum(r['evaluated'] for r in rows),'scored windows',flush=True)
    def summarize(rows):
        if not rows:return {'windows':0,'methods':{}}
        return {'windows':len(rows),'methods':{m:{metric:float(np.mean([r['metrics'][m][metric] for r in rows]))
            for metric in rows[0]['metrics'][m]} for m in METHODS}}
    evalrows=[r for r in allrows if r['evaluated']]
    summary={'all':summarize(evalrows),'nonedge':summarize([r for r in evalrows if not r['edge_truncated']]),
        'nonoverlap':summarize([r for r in allrows if r['nonoverlap']]),
        'by_track':{s['key']:summarize([r for r in evalrows if r['key']==s['key']]) for s in specs},
        'by_recording':{sid:summarize([r for r in evalrows if r['recording']==sid]) for sid in sorted({s['recording'] for s in specs})}}
    write(OUT/'summary.json',summary)
    # End-to-end prefix-only contract: mutate all future boxes then recompute adapter+model.
    sample=allrows[0];prefix=np.array(sample['history_xyxy'],dtype=np.float32)
    full=np.r_[prefix,np.ones((15,4),dtype=np.float32)]
    a=predict(model,full[:6][None]);full[6:]=1e8;b=predict(model,full[:6][None])
    leak=max(float(abs(a[m]-b[m]).max()) for m in METHODS);assert leak==0
    # Known linear motion gives analytic centers at each forecast timestamp.
    ts=np.arange(-5,1)/10;centers=np.c_[500+20*ts,700-10*ts]
    linearbox=np.c_[centers-[40,100],centers+[40,100]].astype(np.float32)
    test=predict(model,linearbox[None])
    analytic=np.c_[500+20*np.arange(1,16)/10,700-10*np.arange(1,16)/10]
    line_error=float(abs(test['linear'][0,:,:2]-analytic).max());assert line_error<.002
    write(OUT/'verification.json',dict(strict_checkpoint_load=True,checkpoint_epoch=ckpt['epoch'],
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        source_hashes_unchanged=True,future_suffix_mutation_max_change=leak,
        known_linear_motion_max_error_pixels=line_error,official_scan_vs_step_max_errors=backend,
        backend='Official v2.3.2.post1 Mamba class + selective_scan_ref, CPU float32. No CUDA fused-kernel parity claim.',
        torch_version=torch.__version__,rolling_windows=len(allrows),scored_windows=len(evalrows),
        human_gold=False,training_performed=False))
    print('SUMMARY',summary['all'],flush=True)

if __name__=='__main__':main()
