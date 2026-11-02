"""Read-only recording audit; offline MPS motion is descriptive, not tracking input."""
from pathlib import Path
import csv, json
import numpy as np

ROOT=Path('E:/New_research')
OUT=ROOT/'outputs/dataset_threeway_20261001'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=read(ROOT/'outputs/sam2_scene_trial_20260927/scene_catalog.json')
    samples=list(csv.DictReader((ROOT/'reports/Aria_pedestrian_screening/sampled_frame_screen.csv').open(encoding='utf-8-sig')))
    result=[]
    for scene in scenes:
        rec=scene['recording']; sid=scene['id']
        parsed=Path('E:/Aria/parsed')/rec
        video=parsed/'logs/214-1/video.mp4'
        timepath=parsed/'logs/214-1/mp4_to_vrs_time_ns.csv'
        rows=list(csv.DictReader(timepath.open(encoding='utf-8-sig')))
        stamp=np.array([float(r['vrs_device_time_ns'])/1e9 for r in rows])
        rel=np.array([float(r['mp4_time_ns'])/1e9 for r in rows])
        posepath=ROOT/f'data/Aria/raw_vrs/mps_{rec}_vrs/slam/closed_loop_trajectory.csv'
        item={**scene,'video':str(video),'video_exists':video.exists(),'timestamp_map':str(timepath),'mps_source':str(posepath) if posepath.exists() else None,'windows':[],'sampled_frames':[]}
        if posepath.exists():
            cols=['tracking_timestamp_us','qx_world_device','qy_world_device','qz_world_device','qw_world_device','quality_score']
            header=posepath.open().readline().strip().split(',')
            a=np.loadtxt(posepath,delimiter=',',skiprows=1,usecols=[header.index(k) for k in cols])
            graphs=np.loadtxt(posepath,delimiter=',',skiprows=1,usecols=0,dtype=str)
            pt=a[:,0]/1e6
            idx=np.searchsorted(pt,stamp).clip(1,len(pt)-1)
            idx=np.where(np.abs(pt[idx]-stamp)<np.abs(pt[idx-1]-stamp),idx,idx-1)
            dt=np.abs(pt[idx]-stamp)
            q=a[idx,1:5]; q/=np.linalg.norm(q,axis=1,keepdims=True)
            graph=graphs[idx]
            valid=(dt<=.03)&(a[idx,5]>.5)
            delta=np.diff(stamp)
            angle=np.degrees(2*np.arccos(np.clip(np.abs((q[1:]*q[:-1]).sum(axis=1)),0,1)))
            velocity=np.full(len(stamp),np.nan)
            ok=valid[1:]&valid[:-1]&(delta>.02)&(delta<.2)&(graph[1:]==graph[:-1])
            velocity[1:]=np.where(ok,angle/np.maximum(delta,1e-9),np.nan)
            one=np.full(len(stamp),np.nan)
            d1=np.degrees(2*np.arccos(np.clip(np.abs((q[10:]*q[:-10]).sum(axis=1)),0,1)))
            ok1=valid[10:]&valid[:-10]&(stamp[10:]-stamp[:-10]>.8)&(stamp[10:]-stamp[:-10]<1.2)&(graph[10:]==graph[:-10])
            one[10:]=np.where(ok1,d1,np.nan)
            item['pose_coverage']=float(valid.mean())
            item['pose_graph_count']=len(set(graphs.tolist()))
            item['angular_speed_p50_p95_deg_s']=[float(np.nanpercentile(velocity,p)) for p in [50,95]]
        else:
            valid=np.zeros(len(stamp),bool); velocity=one=np.full(len(stamp),np.nan)
            item['pose_coverage']=0
        for start in np.arange(0,len(rows)/10,5):
            end=min(start+5,len(rows)/10)
            m=(rel>=start)&(rel<end)
            v=velocity[m]; o=one[m&(rel>=start+1)]
            cov=float(valid[m].mean()) if m.any() else 0
            p95=float(np.nanpercentile(v,95)) if np.isfinite(v).any() else None
            maximum=float(np.nanmax(o)) if np.isfinite(o).any() else None
            motion='unknown' if cov<.8 or maximum is None else ('strong' if maximum>=25 or p95>=60 else ('low' if maximum<=10 and p95<=25 else 'moderate'))
            item['windows'].append({'start_s':float(start),'end_s':float(end),'pose_coverage':cov,'omega_p95_deg_s':p95,'max_1s_rotation_deg':maximum,'motion_band':motion})
        for sample in samples:
            if sample['recording']!=rec:continue
            f=int(sample['frame']); t=f/10
            window=next((w for w in item['windows'] if w['start_s']<=t<w['end_s']),None)
            item['sampled_frames'].append({'frame':f,'time_s':t,'detector_people':int(sample['person_count']),'detector_large_people':int(sample['large_person_count']),'window':window})
        item['motion_band_seconds']={band:sum(w['end_s']-w['start_s'] for w in item['windows'] if w['motion_band']==band) for band in ['low','moderate','strong','unknown']}
        result.append(item)
        print(sid,'omega',item.get('angular_speed_p50_p95_deg_s'),'bands',item['motion_band_seconds'],flush=True)
    save(OUT/'motion_inventory.json',result)
    save(OUT/'protocol.json',{'date':'2026-10-01','scope':'All 29 local Aria recordings; retrospective descriptive classification, not a benchmark split or method evaluation','motion_source':'Offline MPS closed-loop device orientation; geodesic rotation magnitude is invariant to fixed device-camera extrinsics','alignment':'Nearest pose to original export VRS timestamp, tolerance 30 ms, quality > .5. Duplicate timestamps excluded from derivatives. Missing/low coverage retained as unknown.','window_seconds':5,'motion_bands':{'low':'max 1 s rotation <= 10 deg AND p95 adjacent-frame angular speed <= 25 deg/s','strong':'max 1 s rotation >= 25 deg OR p95 speed >= 60 deg/s','moderate':'between low and strong','unknown':'pose coverage <80% or no valid 1 s pair'},'threshold_status':'Exploratory screening priors; not thresholds from HEADS-UP and not calibrated against tracking outcomes. Motion bands alone do not assign the three scene classes.','people_counts':'Historical YOLO samples, 12 per recording; noisy detections including posters/reflections. Not GT, not temporal density estimates.','limitations':['No camera translation or parallax decomposition','Sparse visual review cannot certify absence of short crossings or occlusion','Retrospective labels must not be interpreted as controlled acquisition conditions']})

if __name__=='__main__':main()
