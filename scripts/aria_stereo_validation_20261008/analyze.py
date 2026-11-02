"""RGB registration, mask-interior distance diagnostics and review media."""
import sys,json,gzip,collections,subprocess
from pathlib import Path
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate import ROOT,OUT,dump,sha
sys.path.insert(0,str(ROOT/'scripts/midterm_20260927'))
from common import encoder,finish_encoder

def decode(rle):
    mask=np.zeros((1920,2560),np.uint8)
    counts=rle[:-4];x1,y1,x2,y2=map(int,rle[-4:])
    mask[y1:y2+1,x1:x2+1]=np.repeat(np.arange(len(counts))%2,np.array(counts,dtype=np.int64)).reshape(y2-y1+1,x2-x1+1)
    return mask

def project(pair,depth,support,valid):
    y,x=np.where(support&np.isfinite(depth)&(depth>0))
    z=depth[y,x];fx,fy,cx,cy=pair['intrinsics']
    points=np.stack([(x-cx)*z/fx,(y-cy)*z/fy,z],1)
    t=pair['T_rgb_cv'];pr=points@t[:3,:3].T+t[:3,3]
    uvh=pr@pair['K_rgb'].T; uv=uvh[:,:2]/uvh[:,2:]
    inside=(pr[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<2559.5)&(uv[:,1]>=0)&(uv[:,1]<1919.5)
    x=x[inside];y=y[inside];pr=pr[inside];uv=np.rint(uv[inside]).astype(int)
    # Exact RGB pixel z-buffer. No morphological depth fill is used for measurement.
    order=np.argsort(pr[:,2],kind='stable');_,idx=np.unique(uv[order,1]*2560+uv[order,0],return_index=True);keep=order[idx]
    x=x[keep];y=y[keep];pr=pr[keep];uv=uv[keep]
    return dict(u=uv[:,0],v=uv[:,1],z_rgb=pr[:,2],range_m=np.linalg.norm(pr,axis=1),valid=valid[y,x],x_cv=x,y_cv=y)

def color_depth(z):
    # Fixed near-red / far-blue inverse Turbo scale, log depth 1--15 m.
    scaled=np.clip(np.log(np.maximum(z,1))/np.log(15),0,1)
    return cv2.applyColorMap(np.uint8((1-scaled)*255),cv2.COLORMAP_TURBO)

def text(im,s,x,y,scale=.6,c=(235,235,235)):
    cv2.putText(im,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,c,1,cv2.LINE_AA)

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());measurements=[];frames=[];masks={};summaries=[]
    media=OUT/'presentation';media.mkdir(exist_ok=True)
    for cid in protocol['clips']:
        with gzip.open(ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz','rt') as f:
            masks[cid]=collections.defaultdict(list)
            for line in f:
                r=json.loads(line)
                if r['frame']<=20:masks[cid][r['frame']].append(r)
    for e in protocol['entries']:
        cid=e['clip'];f=e['frame'];folder=OUT/cid/f'{f:03d}'
        pair=np.load(folder/'pair.npz');depth=np.load(folder/'depth.npz');z=depth['depth_m'];disp=depth['disparity_px']
        y,x=np.indices(z.shape);xr=x-disp;xi=np.clip(np.nan_to_num(xr),0,511).astype(int)
        support=pair['support_left']&pair['support_right'][y,xi]&(xr>=0)&(xr<=511)
        valid=support&depth['lr_consistent']&np.isfinite(z)&(z>0)&np.isfinite(disp)&(disp>0)
        np.savez_compressed(folder/'checked_depth.npz',depth_m=z,valid=valid,support=support)
        points=project(pair,z,support,valid);np.savez_compressed(folder/'rgb_points.npz',**points)
        meta=json.loads((folder/'metadata.json').read_text());inf=json.loads((folder/'inference.json').read_text())
        frames.append(dict(**e,valid_fraction=float(valid.mean()),lr_supported=float(valid.sum()/support.sum()),
            support_fraction=float(support.mean()),stereo_skew_ms=meta['stereo_skew_ms'],rgb_skew_ms=meta['rgb_skew_ms'],
            epipolar_matches=meta['epipolar_matches'],epipolar_median_px=float(np.median(meta['epipolar_abs_y_errors_px'])),
            epipolar_p90_px=float(np.percentile(meta['epipolar_abs_y_errors_px'],90)),inference_seconds=inf['seconds']))
        for r in masks[cid][f]:
            mask=decode(r['rle']);interior=cv2.erode(mask,np.ones((15,15),np.uint8))>0
            member=interior[points['v'],points['u']];good=member&points['valid'];n=int(good.sum());a=points['z_rgb'][good];dist=points['range_m'][good]
            measurements.append(dict(clip=cid,frame=f,id=r['id'],segment=r['segment'],points=n,supported_points=int(member.sum()),
                lr_fraction=float(n/max(1,member.sum())),z_median_m=float(np.median(a)) if n>=30 else None,
                range_median_m=float(np.median(dist)) if n>=30 else None,
                z_iqr_m=float(np.percentile(a,75)-np.percentile(a,25)) if n>=30 else None,
                mask_pixels=int(mask.sum()),interior_pixels=int(interior.sum())))
        print('MEASURED',cid,f,flush=True)
    for cid,id in sorted({(r['clip'],r['id']) for r in measurements}):
        rr=sorted([r for r in measurements if r['clip']==cid and r['id']==id],key=lambda r:r['frame']);good=[r for r in rr if r['z_median_m'] is not None]
        steps=[abs(b['z_median_m']-a['z_median_m']) for a,b in zip(good[:-1],good[1:]) if b['frame']==a['frame']+1]
        summaries.append(dict(clip=cid,id=id,mask_frames=len(rr),measured_frames=len(good),
            depth_min_m=min((r['z_median_m'] for r in good),default=None),depth_max_m=max((r['z_median_m'] for r in good),default=None),
            median_points=float(np.median([r['points'] for r in rr])),median_lr_fraction=float(np.median([r['lr_fraction'] for r in rr])),
            median_abs_step_m=float(np.median(steps)) if steps else None,max_abs_step_m=max(steps,default=None),
            median_iqr_m=float(np.median([r['z_iqr_m'] for r in good])) if good else None))
    dump(OUT/'measurements.json',measurements);dump(OUT/'frame_metrics.json',frames);dump(OUT/'track_summary.json',summaries)
    # Shared 0--12 m y axis makes clip plots directly comparable; no temporal smoothing.
    colors=['#4caf50','#edb84b','#dd65cc','#37bacb','#eb7171','#ababff','#fafafa']
    charts={}
    for cid in protocol['clips']:
        fig,ax=plt.subplots(figsize=(15.36,2.3),dpi=100);fig.patch.set_facecolor('#101820');ax.set_facecolor('#101820')
        for j,id in enumerate(sorted({r['id'] for r in measurements if r['clip']==cid})):
            rr=[r for r in measurements if r['clip']==cid and r['id']==id and r['z_median_m'] is not None]
            if rr:
                values={r['frame']:r['z_median_m'] for r in rr}
                ax.plot(np.arange(21)/10,[values.get(f,np.nan) for f in range(21)],'-o',ms=3,label=id,color=colors[j%len(colors)])
        ax.set_xlim(0,2);ax.set_ylim(0,12);ax.set_xlabel('Recorded time (s)',color='white');ax.set_ylabel('RGB-axis depth (m)',color='white')
        ax.tick_params(colors='white');ax.grid(alpha=.2)
        if ax.lines:ax.legend(ncol=6,loc='upper right')
        else:ax.text(.5,.5,'No existing SAM masks in this selected interval',transform=ax.transAxes,ha='center',color='white',fontsize=14)
        fig.tight_layout()
        fig.savefig(media/f'{cid}_depth_trace.png',facecolor=fig.get_facecolor());plt.close(fig);charts[cid]=cv2.imread(str(media/f'{cid}_depth_trace.png'))
    enc,log=encoder(media/'stereo_depth_review.mp4',1536,850,5,20)
    snapshots=[]
    try:
        for cid in protocol['clips']:
            ce,cl=encoder(media/f'{cid}_stereo_depth.mp4',1536,850,5,20)
            try:
                ids=sorted({r['id'] for r in measurements if r['clip']==cid})
                for f in range(21):
                    folder=OUT/cid/f'{f:03d}';pair=np.load(folder/'pair.npz');dep=np.load(folder/'checked_depth.npz')
                    with np.load(folder/'rgb_points.npz') as packed:points={k:packed[k] for k in packed.files}
                    canvas=np.full((850,1536,3),(32,24,16),np.uint8)
                    text(canvas,f'Aria Gen 2 | clip {cid} | t={f/10:.1f}s | pretrained stereo + existing SAM masks',16,25,.7)
                    text(canvas,'Rectified front-left camera',16,58);text(canvas,'Depth: warm=near, cool=far; black=rejected',524,58,.53);text(canvas,'Depth projected into RGB + SAM contours',1036,58,.55)
                    canvas[70:582,:512]=cv2.cvtColor(pair['left'],cv2.COLOR_GRAY2BGR)
                    dc=color_depth(dep['depth_m']);dc[~dep['valid']]=0;canvas[70:582,512:1024]=dc
                    rgb=cv2.resize(cv2.imread(str(folder/'rgb.jpg')),(512,384));overlay=rgb.copy()
                    ii=np.where(points['valid'])[0];col=color_depth(points['z_rgb'].reshape(-1,1)).reshape(-1,3)
                    # Visual-only point splats; no filled/splatted values are used in measurements.
                    for k in ii[::2]:cv2.circle(overlay,(int(points['u'][k]/5),int(points['v'][k]/5)),1,tuple(map(int,col[k])),-1)
                    rgb=cv2.addWeighted(rgb,.6,overlay,.4,0)
                    for r in masks[cid][f]:
                        mask=cv2.resize(decode(r['rle']),(512,384),interpolation=cv2.INTER_NEAREST)
                        contour,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        h=colors[ids.index(r['id'])%len(colors)].lstrip('#');c=tuple(int(h[k:k+2],16) for k in (4,2,0));cv2.drawContours(rgb,contour,-1,c,1)
                        m=next(a for a in measurements if a['clip']==cid and a['frame']==f and a['id']==r['id'])
                        b=r['box'];xx=min(400,max(0,int(b[0]/5)));yy=max(16,min(365,int(b[1]/5)-5))
                        label=f"{r['id']} {m['z_median_m']:.2f} m" if m['z_median_m'] is not None else f"{r['id']} insufficient"
                        j=ids.index(r['id']);text(canvas,label,1030+250*(j//4),83+14*(j%4),.43,c)
                        cv2.putText(rgb,r['id'],(xx,yy),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_AA);text(rgb,r['id'],xx,yy,.4,c)
                    canvas[134:518,1024:1536]=rgb
                    ramp=color_depth(np.geomspace(1,15,280).reshape(1,-1));canvas[591:603,608:888]=np.repeat(ramp,12,axis=0)
                    text(canvas,'1 m',555,603,.48);text(canvas,'15 m+',900,603,.48)
                    text(canvas,'Numbers: median interior surface depth, not body-center GT',1030,548,.42)
                    canvas[620:850]=charts[cid]
                    cursor=int(95+(1500-95)*f/20);cv2.line(canvas,(cursor,635),(cursor,797),(180,180,180),1)
                    text(canvas,'Playback 0.5x | No measured distance GT',16,606,.5)
                    ce.stdin.write(canvas.tobytes());enc.stdin.write(canvas.tobytes())
                    if f in [0,10,20]:cv2.imwrite(str(media/f'{cid}_{f:03d}_review.jpg'),canvas)
                    if f==10:snapshots.append(canvas[:620])
            finally:finish_encoder(ce,cl)
    finally:finish_encoder(enc,log)
    cv2.imwrite(str(media/'overview.jpg'),np.concatenate(snapshots,axis=0))
    # Geometric inverse check and source immutability.
    errs=[]
    for e in protocol['entries']:
        p=np.load(OUT/e['clip']/f"{e['frame']:03d}"/'pair.npz');t=p['T_rgb_cv'];errs.append(float(np.max(np.abs(np.linalg.inv(t)@t-np.eye(4)))))
    unchanged={p:sha(ROOT/p)==v for p,v in protocol['protected'].items()}
    assert all(unchanged.values())
    checks=[]
    for path in media.glob('*.mp4'):
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(850,1536,3);count+=1
        cap.release();expected=63 if path.name=='stereo_depth_review.mp4' else 21;assert count==expected
        checks.append(dict(file=path.name,decoded_frames=count))
    dump(OUT/'checks.json',dict(protected_sources=unchanged,inverse_transform_max_error=max(errs),videos=checks,
        support_correction='Rectified constant-255 support accepts >=254; exact ==255 incorrectly discarded interpolation-rounding pixels. checked_depth.npz recomputes all validity consistently; raw depth outputs retained.',
        execution='Initial inference paused by a concurrently refreshed prepared zip; resumed after preparation completed. No selected time replaced.'))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
