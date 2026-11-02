"""Independent visual/motion screening. No tracker scores are used."""
from pathlib import Path
import json,csv,sys
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('E:/New_research')
OUT=ROOT/'outputs/g1_street_candidates_20261001'
OUT.mkdir(parents=True,exist_ok=True)
INV={r['id']:r for r in json.loads((ROOT/'outputs/dataset_threeway_20261001/motion_inventory.json').read_text(encoding='utf8'))}
RANGES={'155313':(165,251),'164535':(0,16),'164557':(0,75),'192702':(100,280),'193311':(20,81),'193524':(0,37),'193709':(30,96),'193956':(0,78),'201129':(0,40)}
def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def signals(sid):
    rec=INV[sid]
    rows=list(csv.DictReader(open(rec['timestamp_map'],encoding='utf-8-sig')))
    stamps=np.array([float(r['vrs_device_time_ns'])/1e9 for r in rows]);rel=np.array([float(r['mp4_time_ns'])/1e9 for r in rows])
    path=Path(rec['mps_source']);header=path.open().readline().strip().split(',')
    cols=['tracking_timestamp_us','qx_world_device','qy_world_device','qz_world_device','qw_world_device','quality_score','tx_world_device','ty_world_device','tz_world_device']
    a=np.loadtxt(path,delimiter=',',skiprows=1,usecols=[header.index(k) for k in cols])
    pt=a[:,0]/1e6;j=np.searchsorted(pt,stamps).clip(1,len(pt)-1)
    j=np.where(abs(pt[j]-stamps)<abs(pt[j-1]-stamps),j,j-1)
    q=a[j,1:5];q/=np.linalg.norm(q,axis=1,keepdims=True)
    valid=(abs(pt[j]-stamps)<=.03)&(a[j,5]>.5)
    return rel,stamps,q,valid,a[j,6:9]
def metrics(sig,start,end):
    rel,st,q,valid,pos=sig;m=(rel>=start)&(rel<end);inds=np.where(m)[0]
    if len(inds)<2:return None
    i=inds[1:];dt=st[i]-st[i-1]
    ok=valid[i]&valid[i-1]&(dt>.02)&(dt<.2)
    v=np.degrees(2*np.arccos(np.clip(abs((q[i]*q[i-1]).sum(1)),0,1)))/np.maximum(dt,1e-9)
    i=inds[10:];dt1=st[i]-st[i-10]
    ok1=valid[i]&valid[i-10]&(dt1>.8)&(dt1<1.2)
    o=np.degrees(2*np.arccos(np.clip(abs((q[i]*q[i-10]).sum(1)),0,1)))
    good=inds[valid[inds]]
    span=float(np.linalg.norm(pos[good[-1]]-pos[good[0]])) if len(good)>1 else None
    return {'pose_coverage':float(valid[m].mean()),'omega_p95_deg_s':float(np.percentile(v[ok],95)) if ok.any() else None,'max_1s_rotation_deg':float(max(o[ok1])) if ok1.any() else None,'device_endpoint_displacement_m':span}
def sheet(spec):
    sid,start,end=spec;cap=cv2.VideoCapture(INV[sid]['video'])
    frames=np.unique(np.linspace(int(start*10),int(end*10)-1,12).astype(int))
    im=Image.new('RGB',(1600,990),(19,24,30));d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
    for k,f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,bgr=cap.read();assert ok,(sid,f)
        x=k%4*400;y=k//4*330
        im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(bgr,(400,300)),cv2.COLOR_BGR2RGB)),(x,y+30))
        d.text((x+5,y+4),f'{sid} | {f/10:.1f}s',font=font,fill='white')
    cap.release();p=OUT/'screen_sheets'/f'{sid}_{start:03d}_{end:03d}.jpg';p.parent.mkdir(exist_ok=True);im.save(p,quality=90)
    return str(p)
def main():
    allrows=[];chosen=[]
    for sid,(lo,hi) in RANGES.items():
        sig=signals(sid);rows=[]
        for start in range(lo,hi-4):
            met=metrics(sig,start,start+5)
            if met['max_1s_rotation_deg'] is None:continue
            row={'id':sid,'start_s':start,'end_s':start+5,**met}
            row['rank_score']=met['max_1s_rotation_deg']+.22*met['omega_p95_deg_s']+max(0,.9-met['pose_coverage'])*200
            rows.append(row)
        allrows.extend(rows);keep=[]
        for row in sorted(rows,key=lambda x:x['rank_score']):
            if any(abs(row['start_s']-x['start_s'])<7 for x in keep):continue
            keep.append(row)
            if len(keep)>= (12 if sid=='192702' else 6 if sid in ['155313','164557','193709'] else 4):break
        chosen.extend(sorted(keep,key=lambda x:x['start_s']))
        np.savez_compressed(OUT/f'pose_{sid}.npz',rel=sig[0],stamp=sig[1],q=sig[2],valid=sig[3],pos=sig[4])
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row,path in zip(chosen,pool.map(sheet,[(r['id'],r['start_s'],r['end_s']) for r in chosen])):row['sheet']=path
    save(OUT/'all_sliding_motion_windows.json',allrows);save(OUT/'screen_candidates.json',chosen)
    for r in chosen:print(r['id'],r['start_s'],r['end_s'],'1sec',round(r['max_1s_rotation_deg'],1),'omega',round(r['omega_p95_deg_s'],1),'cov',round(r['pose_coverage'],2),flush=True)
if __name__=='__main__':main()
