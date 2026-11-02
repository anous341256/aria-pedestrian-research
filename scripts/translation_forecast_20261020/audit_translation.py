"""Independently replay raw MPS positions, timestamps and camera lever arm."""
from prepare import *
import pandas as pd

def quaternion_matrices(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=q.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)

def main():
    c=np.load(OUT/"camera_dataset.npz");d=np.load(OUT/"dataset.npz")
    meta=read(OUT/"samples.json");names=read(CAM/"inventory.json")["record_names"]
    calibrations=read(CAM/"calibrations.json")
    maximum=0.;personmax=0.;sharedmax=0.;prefix=0;boundary=0
    for rec,name in enumerate(names):
        df=pd.read_csv(ROOT/f"data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv",
            usecols=["tracking_timestamp_us"]+[f"t{a}_world_device" for a in "xyz"]+[f"q{a}_world_device" for a in "xyzw"])
        ns=df.tracking_timestamp_us.to_numpy(dtype=np.int64)*1000
        pos=df[[f"t{a}_world_device" for a in "xyz"]].to_numpy()
        q=df[[f"q{a}_world_device" for a in "xyzw"]].to_numpy()
        R=quaternion_matrices(q)
        ext=np.asarray(calibrations[name]["T_device_camera"])
        xyz=pos+R@ext[:3,3]
        frames=np.load(CAM/f"poses/{name}.npz")
        ii=np.flatnonzero(c["record_index"]==rec)
        sampled=c["translation_mps_ns"][ii];asked=c["translation_rgb_ns"][ii]
        j=np.searchsorted(ns,sampled)
        assert np.array_equal(ns[j],sampled)
        assert (sampled<=asked[:,-1,None]).all()
        expected=(np.searchsorted(ns,asked,side="right")-1).clip(0)
        assert np.array_equal(j,expected)
        boundary+=int((sampled>asked).sum())
        cutoff=c["frame_index"][ii]
        # Row-vector formulation independently equivalent to R_cutoff.T @ displacement.
        local=(xyz[j]-xyz[j[:,-1,None]])@frames["R"][cutoff]
        maximum=max(maximum,float(np.max(abs(local-c["translation"][ii]))))
        for k,m in enumerate(meta):
            if m["recording"]!=name:continue
            jt=np.searchsorted(ns,d["translation_mps_ns"][k])
            assert np.array_equal(ns[jt],d["translation_mps_ns"][k])
            assert (ns[jt]<=m["timestamp_ns"]).all()
            pp=xyz[jt]
            calc=(pp-pp[-1])@d["R"][k,15]
            personmax=max(personmax,float(np.max(abs(calc-d["translation"][k]))))
            assert np.max(abs(pp-d["translation_world_position"][k]))<1e-9
            match=ii[c["cutoff_ns"][ii]==m["timestamp_ns"]]
            assert len(match)==1
            sharedmax=max(sharedmax,float(np.max(abs(d["translation"][k]-c["translation"][match[0]]))))
            # Remove every raw MPS sample after this forecast origin.
            cutns=ns[ns<=m["timestamp_ns"]]
            indices=(np.searchsorted(cutns,np.array(m["timestamps_ns"][:16]),side="right")-1).clip(0)
            assert np.array_equal(indices,jt)
            prefix+=1
    assert maximum<1e-6 and personmax<1e-6 and sharedmax==0
    source=read(OUT/"source_hashes.json")
    assert all(sha(ROOT/k)==v for k,v in source.items())
    result=dict(passed=True,all_camera_history_positions_replayed=len(c["translation"])*16,
        person_raw_source_prefix_replays=prefix,manual_quaternion_full_camera_max_m=maximum,
        manual_quaternion_person_max_m=personmax,person_full_camera_translation_difference=sharedmax,
        historical_start_boundary_occurrences=boundary,all_position_samples_at_or_before_forecast=True,
        protected_files_unchanged=len(source))
    write(OUT/"translation_audit.json",result);print(result,flush=True)
if __name__=="__main__":main()
