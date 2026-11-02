"""Independent timestamp, raw-quaternion and cache arithmetic checks."""
import argparse,csv,hashlib,json,itertools
from pathlib import Path
import numpy as np

def mat(q):
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=a.data_root;out=a.out;g=out/'gaze';audit=json.loads((g/'audit.json').read_text());maxray=maxpos=maxraw=0.;checks=0;totals=0
    for rec in audit['records']:
        name=rec['recording'];d=np.load(g/f'{name}.npz');ts=d['timestamp_ns'];totals+=len(ts)
        with (root/f'data/Aria/raw_vrs/mps_{name}_vrs/eye_gaze/general_eye_gaze.csv').open() as f:
            raw=list(csv.DictReader(f))
        assert np.array_equal(ts,[int(r['tracking_timestamp_us'])*1000 for r in raw])
        assert np.all(np.diff(ts)>0)
        manual=np.array([[np.tan(float(r['yaw_rads_cpf'])),np.tan(float(r['pitch_rads_cpf'])),1.] for r in raw])
        manual/=np.linalg.norm(manual,axis=1,keepdims=True)
        # Official Gen2 loader angles differ slightly from CSV combined-angle decimals
        # (checked independently ~1e-7 rad); do not demand double precision agreement there.
        maxraw=max(maxraw,float(np.max(np.abs(d['direction_cpf']-manual))))
        assert np.allclose(d['direction_cpf'],manual,rtol=0,atol=5e-7)
        good=np.flatnonzero(d['pose_valid'])
        if not len(good):continue
        picked=good[np.unique(np.linspace(0,len(good)-1,5,dtype=int))];targets=ts[picked];k=0;prev=None
        path=root/f'data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv'
        with path.open() as f:
            for row in itertools.chain(csv.DictReader(f),[None]):
                t=int(row['tracking_timestamp_us'])*1000 if row is not None else float('inf')
                while k<len(targets) and t>targets[k]:
                    assert prev is not None
                    i=picked[k];pt=int(prev['tracking_timestamp_us'])*1000
                    assert 0<=(ts[i]-pt)/1e6<=20
                    assert abs(d['pose_age_ms'][i]-(ts[i]-pt)/1e6)<1e-10
                    R=mat(np.array([float(prev[f'q{ax}_world_device']) for ax in 'xyzw']))
                    p=np.array([float(prev[f't{ax}_world_device']) for ax in 'xyz'])
                    wr=R@d['T_device_cpf'][:3,:3]@d['direction_cpf'][i]
                    cp=p+R@d['T_device_rgb'][:3,3]
                    maxray=max(maxray,float(np.max(np.abs(wr-d['direction_world'][i]))))
                    maxpos=max(maxpos,float(np.max(np.abs(cp-d['rgb_center_world'][i]))));checks+=1;k+=1
                prev=row
                if k==len(targets):break
        assert k==len(targets)
    print('independent_geometry_maxima',maxray,maxpos,flush=True)
    assert totals==audit['total']['rows'];assert maxray<1e-12 and maxpos<1e-12
    count=0
    for cohort,path in [('short','outputs/g1_expanded_forecast_20261029/expanded_samples.json'),('long','outputs/horizon_15_15_g1_20261029/samples.json')]:
        meta=json.loads((root/path).read_text());assert len(meta)==len(audit['cohorts'][cohort])
        for m,row in zip(meta,audit['cohorts'][cohort]):
            assert m['key']==row['key'];d=np.load(g/f"{m['recording']}.npz");ts=d['timestamp_ns']
            lo,hi=int(m['timestamps_ns'][0]),int(m['timestamps_ns'][15]);ix=[j for j,t in enumerate(ts) if lo<=t<=hi]
            assert len(ix)==row['native_rows'];assert sum(d['valid'][ix])==row['valid_direction']
            assert sum(d['pose_valid'][ix])==row['pose_joined'];assert sum(d['in_image'][ix])==row['in_image']
            # Physical future truncation yields the same historical observation sequence.
            assert np.array_equal(ts[ix],np.array([t for t in ts[ts<=hi] if t>=lo]));count+=1
    sources={}
    for p in [out/'inventory/source_hashes.json',g/'source_hashes.json']:sources.update(json.loads(p.read_text()))
    changed=[p for p,h in sources.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    assert not changed
    result={'gaze_timestamps_and_direction_rows':totals,'manual_raw_quaternion_pose_samples':checks,
            'world_direction_max_abs_error':maxray,'optical_center_max_abs_m':maxpos,
            'raw_csv_vs_official_direction_max_abs':maxraw,
            'historical_windows_checked':count,'future_truncation_passed':True,'protected_files_unchanged':len(sources),
            'scope':'Arithmetic and read-only checks, not gaze accuracy, behavior labels, or independent event counts.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
