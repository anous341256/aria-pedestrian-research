"""Paired historical-translation ablation; preserve all prior experiments."""
from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREV = ROOT / "outputs/horizon_swap_20261017"
OUT = ROOT / "outputs/translation_forecast_20261020"
CAM = ROOT / "outputs/camera_full_data_20261002"
OLD = ROOT / "outputs/biflow_aria_v1_20261002"
BASE = ROOT / "outputs/factorized_forecast_v1_20261002"
spec = importlib.util.spec_from_file_location("horizon_geometry", ROOT/"scripts/horizon_swap_20261017/prepare.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
read, write, sha, rm, rv, warp = [getattr(geometry,k) for k in ["read","write","sha","rm","rv","warp"]]
ARMS = ["no_translation", "with_translation"]

def main():
    import pandas as pd
    from scipy.spatial.transform import Rotation
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = dict(version="translation_forecast_20261020", arms=ARMS, seeds=[17,29,43],
        cohort="Exact previous 117 person windows, six held-out recording folds and 27185 full-camera windows",
        horizon="16 historical points spanning1.5s ->5 future points spanning0.5s; same cutoff/targets as previous long_history",
        inputs="Append3 historical RGB-camera displacement coordinates in meters to BOTH camera and pedestrian GRU inputs. R_cutoff.T @ (p_rgb_history-p_rgb_cutoff). No future translation is extracted or supplied.",
        position="Closed-loop offline MPS device position and quaternion; factory T_device_camera lever arm. Select last MPS pose at/before each historical RGB time. At recording start only, if none exists, allow first pose within30ms of that historical frame, provided it is still at/before the forecast cutoff. No future-cutoff pose; quality>=.5. Original rotations unchanged.",
        control="Identical enlarged architectures/initial states/batches/augmentations. Zero all3 translation channels in no_translation. Extra GRU input columns initialized zero in BOTH arms.",
        camera_model="Original GRU32, MLP64,5 future rotation vectors. Input6 instead of3. 2000 updates, batch256, AdamW lr.001 wd.01 clip1. Validation rotvec MSE every100.",
        person_model="Original GRU64/context95-64/future-flow/history-flow. Input9 instead of6. 240 updates,48 recording-balanced samples, same40/20/40 augmentation, history loss .5,5 flow draws/10 steps. Validation residual ADE every20.",
        geometry="Rotation-only history compensation and future reprojection unchanged. No depth, no explicit translation compensation, no predicted future translation, no attention or gaze.",
        scaling="Translation in meters divided by fixed1m; no data-derived normalization or future statistics.",
        evaluation="Final ADE/FDE, residual errors, camera angle, GT-branch replacements and cross-arm branch swaps; seed/recording/sparse aggregates; fixed original nine videos seed17.",
        selection="No test tuning, no source labels/model/PPT changes, no replacement of selected baseline. Prior saved long_history is secondary reference; new matched no_translation is primary.",
        limitation="Offline MPS and reviewed SAM references; no online pose or independent ground-truth accuracy claim.")
    if (OUT/"preflight.json").exists():
        assert read(OUT/"protocol.json")==cfg
        return
    if (OUT/"protocol.json").exists() and read(OUT/"protocol.json")!=cfg:
        write(OUT/"protocol_before_start_boundary_fix.json",read(OUT/"protocol.json"))
    write(OUT/"protocol.json",cfg)
    plan=ROOT/"plans/TRANSLATION_FORECAST_20261020.txt"
    plan.parent.mkdir(exist_ok=True)
    plan.write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
    protected={str(p.relative_to(ROOT)):sha(p) for p in PREV.rglob("*") if p.is_file() and p.suffix in [".json",".npz",".pt"]}
    for folder in ["horizon_swap_20261017","biflow_aria_v1_20261002","camera_full_data_20261002"]:
        protected.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"scripts"/folder).glob("*.py")})
    calibrations=read(CAM/"calibrations.json")
    protected[str((CAM/"calibrations.json").relative_to(ROOT))]=sha(CAM/"calibrations.json")
    names=read(CAM/"inventory.json")["record_names"]
    camera=dict(np.load(PREV/"camera_dataset.npz"))
    data=dict(np.load(PREV/"dataset.npz"))
    meta=read(PREV/"samples.json")
    ctr=np.empty((len(camera["rotation"]),16,3),np.float32)
    ctimes=np.empty((len(ctr),16),np.int64)
    cmps=np.empty_like(ctimes)
    ptr=np.empty((len(meta),16,3),np.float32)
    pmps=np.empty((len(meta),16),np.int64)
    pworld=np.empty((len(meta),16,3),np.float64)
    audit=[]
    for rec,name in enumerate(names):
        source=ROOT/f"data/Aria/raw_vrs/mps_{name}_vrs/slam/closed_loop_trajectory.csv"
        protected[str(source.relative_to(ROOT))]=sha(source)
        framepath=CAM/f"poses/{name}.npz"
        protected[str(framepath.relative_to(ROOT))]=sha(framepath)
        frames=np.load(framepath)
        requested=frames["times_ns"]
        cols=["tracking_timestamp_us","graph_uid","quality_score"]+[f"t{a}_world_device" for a in "xyz"]+[f"q{a}_world_device" for a in "xyzw"]
        df=pd.read_csv(source,usecols=cols)
        ns=df.tracking_timestamp_us.to_numpy(np.int64)*1000
        assert (np.diff(ns)>0).all()
        ii=np.searchsorted(ns,requested,side="right")-1
        safe=ii.clip(0)
        age=requested-ns[safe]
        quality=df.quality_score.to_numpy()[safe]
        graphs=df.graph_uid.to_numpy()[safe]
        q=df[[f"q{a}_world_device" for a in "xyzw"]].to_numpy()[safe]
        p=df[[f"t{a}_world_device" for a in "xyz"]].to_numpy()[safe]
        ext=np.array(calibrations[name]["T_device_camera"])
        rd=Rotation.from_quat(q).as_matrix()
        # T_world_rgb = T_world_device @ T_device_rgb
        rgb=p+np.einsum("nij,j->ni",rd,ext[:3,3])
        valid=(abs(age)<=30000000)&(quality>=.5)&np.isfinite(rgb).all(1)
        ix=np.flatnonzero(camera["record_index"]==rec)
        fs=camera["frame_index"][ix,None]+np.arange(-15,1)
        assert valid[fs].all(),(name,"invalid historical MPS")
        assert (ns[safe[fs]]<=requested[fs[:,-1,None]]).all()
        assert (graphs[fs]==graphs[fs[:,-1,None]]).all()
        assert np.array_equal(requested[fs[:,-1]],camera["cutoff_ns"][ix])
        disp=rgb[fs]-rgb[fs[:,-1,None]]
        local=np.einsum("nji,ntj->nti",frames["R"][fs[:,-1]],disp)
        ctr[ix]=local
        ctimes[ix]=requested[fs]
        cmps[ix]=ns[safe[fs]]
        for j,m in enumerate(meta):
            if m["recording"]!=name: continue
            ts=np.array(m["timestamps_ns"][:16])
            ff=np.searchsorted(requested,ts)
            assert np.array_equal(requested[ff],ts)
            assert valid[ff].all()
            assert (ns[safe[ff]]<=ts[-1]).all()
            assert (graphs[ff]==graphs[ff[-1]]).all()
            assert np.max(abs(frames["R"][ff]-data["R"][j,:16]))<1e-8
            ptr[j]=np.einsum("ji,tj->ti",data["R"][j,15],rgb[ff]-rgb[ff[-1]])
            pmps[j]=ns[safe[ff]]
            pworld[j]=rgb[ff]
        # A physically truncated source yields identical selection for a complete history.
        ff=fs[0];cut=requested[ff[-1]]
        truncated=ns[ns<=cut]
        assert np.array_equal((np.searchsorted(truncated,requested[ff],side="right")-1).clip(0),safe[ff])
        audit.append(dict(recording=name,windows=len(ix),max_selected_age_ms=float(age[fs].max()/1e6),
            start_boundary_frame_occurrences=int((age[fs]<0).sum()),
            max_after_historical_frame_ms=float(max(0,-age[fs].min())/1e6),
            min_quality=float(quality[fs].min()),source_prefix_selection_exact=True))
        print(name,len(ix),flush=True)
    assert np.isfinite(ctr).all() and np.isfinite(ptr).all()
    assert (ctr[:,-1]==0).all() and (ptr[:,-1]==0).all()
    assert (cmps<=ctimes[:,-1,None]).all()
    data.update(translation=ptr,translation_mps_ns=pmps,translation_world_position=pworld)
    camera.update(translation=ctr,translation_rgb_ns=ctimes,translation_mps_ns=cmps)
    np.savez_compressed(OUT/"dataset.npz",**data)
    np.savez_compressed(OUT/"camera_dataset.npz",**camera)
    for filename in ["samples.json","folds.json","camera_folds.json"]:
        write(OUT/filename,read(PREV/filename))
    write(OUT/"source_hashes.json",protected)
    write(OUT/"translation_extraction.json",audit)
    write(OUT/"preflight.json",dict(passed=True,person_windows=len(meta),camera_windows=len(ctr),recordings=len(names),history_translation_shape=list(ptr.shape),
        no_future_translation=True,source_prefix_selection_exact=True,max_selected_age_ms=max(x["max_selected_age_ms"] for x in audit),protected_files=len(protected)))
    print(read(OUT/"preflight.json"),flush=True)

if __name__=="__main__":main()
