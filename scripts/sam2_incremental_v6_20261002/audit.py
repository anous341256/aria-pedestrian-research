"""Verify donors/time/memory; census unresolved discovery and cross-ID mask overlap."""
import sys,json,collections
from pathlib import Path
import cv2,numpy as np
from render import ROOT,OUT,SRC,load,decode,read_json,write_json
def iou(a,b):
    if a is None or b is None:return 0.
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))
def audit(cid):
    dest=OUT/cid;inv=read_json(dest/'inventory.json');maskrows=load(dest/'masks_rle.jsonl.gz');decisions=load(dest/'decisions.jsonl.gz');byframe=collections.defaultdict(list);lookup={(r['frame'],r['id']):r for r in maskrows};dl={(r['frame'],r['id']):r for r in decisions};batches={};seedfails=[];checks=0;candchecks=0;negcalls=0;conflicts=[];pointcache={}
    for path in sorted(dest.glob('batch_*/input.json')):
        info=read_json(path)
        for e in info['entries']:batches[e['id']]=info['batch']
        for c in load(path.parent/'candidates.jsonl.gz'):
            candchecks+=1;negcalls+=bool(c['negative_points'])
            for p in c['negative_points']:
                f=c['frame'];donor=p['donor_id'];assert p['frame']==f and donor!=c['owner'];assert batches.get(donor,999)<info['batch'];assert dl[(f,donor)]['memory_written']
                key=(f,donor)
                if key not in pointcache:pointcache[key]=decode(lookup[key]['rle'])
                x,y=p['xy'];assert pointcache[key][y,x];checks+=1
            if len(pointcache)>300:pointcache.clear()
    for d in decisions:
        if d['status']=='SEED_FAILED':seedfails.append(dict(id=d['id'],frame=d['frame']))
        if d.get('memory_after'):
            vals=d['memory_after']['references']+d['memory_after']['recent'];assert max(vals)<=d['frame']
            if not d['memory_written']:assert d['memory_after']==d['memory_before']
        if d.get('prior'):assert d['prior']['last_observation_frame']<d['frame']
    for r in maskrows:byframe[r['frame']].append(r)
    nonempty=0
    for f,rows in byframe.items():
        ms=[]
        for r in rows:
            m=cv2.resize(decode(r['rle']),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.any():nonempty+=1;ms.append((r['id'],m,int(m.sum())))
        for j,(identity,a,aa) in enumerate(ms):
            for other,b,bb in ms[:j]:
                inter=int(np.count_nonzero(a&b));contain=inter/max(1,min(aa,bb));union=aa+bb-inter
                if contain>.65 and min(aa,bb)/max(aa,bb)>.15:conflicts.append(dict(frame=f,ids=[identity,other],containment=contain,iou=inter/max(1,union)))
    tracks=read_json(dest/'proposal_tracklets.json');weak=[]
    for t in tracks:
        if t['eligible'] or inv['proposal_disposition'][str(t['proposal_id'])]['status']=='rejected_visual_review':continue
        hits=[]
        for r in t['rows']:
            best=max(((iou(r['box'],d.get('observation_box')),d['id']) for d in decisions if d['frame']==r['frame']),default=(0,None))
            if best[0]>.35:hits.append(best[1])
        if not hits and t['max_conf']>=.3:weak.append(t)
    summary=dict(clip=cid,batches=inv['batches'],provisional_ids=len(inv['entries']),seed_failures=seedfails,mask_rows=len(maskrows),nonempty_masks=nonempty,candidate_calls=candchecks,negative_candidate_calls=negcalls,verified_negative_points=checks,high_overlap_pairs_frames=len(conflicts),high_overlap_unique_pairs=len({tuple(sorted(c['ids'])) for c in conflicts}),weak_unassociated_proposals=len(weak),verification='all donor earlier-batch/current-frame/interior checks; no rejected memory write; no future pose in propagation',caveat='Counts are automatic diagnostics, not GT accuracy or unique people.')
    write_json(dest/'audit.json',summary);write_json(dest/'overlap_flags.json',conflicts);write_json(dest/'weak_review.json',weak)
    if weak:
        cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));cells=[]
        for t in weak:
            s=max(t['rows'],key=lambda r:r['confidence']);cap.set(cv2.CAP_PROP_POS_FRAMES,s['frame']);ok,im=cap.read();assert ok;x1,y1,x2,y2=map(int,s['box']);pad=20;crop=im[max(0,y1-pad):min(1920,y2+pad),max(0,x1-pad):min(2560,x2+pad)];h,w=crop.shape[:2];scale=min(210/w,220/h);small=cv2.resize(crop,(max(1,int(w*scale)),max(1,int(h*scale))));tile=np.zeros((260,220,3),np.uint8);tile[35:35+small.shape[0],:small.shape[1]]=small
            cv2.putText(tile,f"T{t['proposal_id']} f{s['frame']} n{len(t['rows'])} p{t['max_conf']:.2f}",(3,19),0,.41,(255,255,255),1);cells.append(tile)
        cap.release()
        for p in range((len(cells)+29)//30):
            sheet=np.zeros((5*260,6*220,3),np.uint8)
            for j,cell in enumerate(cells[p*30:(p+1)*30]):sheet[j//6*260:(j//6+1)*260,j%6*220:(j%6+1)*220]=cell
            cv2.imwrite(str(dest/f'weak_review_{p+1}.jpg'),sheet)
    print(summary,flush=True)
if __name__=='__main__':
    for i in range(1,11):
        cid=f'{i:02d}'
        if (OUT/cid/'inventory.json').exists() and not (OUT/cid/'audit.json').exists():audit(cid)
