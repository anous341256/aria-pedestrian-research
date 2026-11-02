import sys,time,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'external/aria_stereo_20261008'))
import export_depth_from_stereo as official

def paired(model,left,right,cfg):
    # Official preprocessing; left-right passes are independent batch elements.
    l=np.stack([left,right[:,::-1].copy()]);r=np.stack([right,left[:,::-1].copy()])
    l=torch.from_numpy(l.copy()).float().cuda()[:,None].repeat(1,3,1,1)
    r=torch.from_numpy(r.copy()).float().cuda()[:,None].repeat(1,3,1,1)
    with torch.no_grad(),torch.amp.autocast('cuda'):
        d=model.forward(l,r,iters=cfg.valid_iters,test_mode=True).float().cpu().numpy().reshape(2,512,512)
    disp=d[0];back=d[1,:,::-1].copy();y,x=np.indices(disp.shape);xx=x-disp;xi=np.clip(np.nan_to_num(xx),0,511).astype(int)
    lr=(np.abs(disp-back[y,xi])<1)&(xx>=0)&(xx<=511)
    return disp,lr

if __name__=='__main__':
    out=ROOT/'outputs/depth_forecast_20261008';out.mkdir(parents=True,exist_ok=True)
    m,c=official.load_foundation_stereo(str(ROOT/'models/foundation_stereo_20261008/23-51-11/model_best_bp2.pth'))
    results=[]
    for cid in ['03','07']:
        a=np.load(ROOT/f'outputs/aria_stereo_validation_20261008/{cid}/010/pair.npz')
        torch.cuda.synchronize();t=time.perf_counter();d,v=official.run_lr_consistency(m,a['left'],a['right'],c);t1=time.perf_counter()-t
        torch.cuda.synchronize();t=time.perf_counter();b,w=paired(m,a['left'],a['right'],c);t2=time.perf_counter()-t
        results.append(dict(clip=cid,sequential_seconds=t1,batched_seconds=t2,disparity_max_diff=float(np.max(np.abs(d-b))),disparity_mean_diff=float(np.mean(np.abs(d-b))),lr_mask_agreement=float(np.mean(v==w)),peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    (out/'batch_validation.json').write_text(json.dumps(results,indent=2));print(results)
