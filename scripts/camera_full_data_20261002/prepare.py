"""Read all available recordings; no video or pedestrian labels required."""
from common import *
import pandas as pd
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider


def main():
    assert not (OUT / 'dataset.npz').exists(), 'Preserve existing prepared data'
    raw = ROOT / 'data/Aria/raw_vrs'
    parsed = Path('/mnt/e/Aria/parsed')
    entries, values, records, cutoffs, starts, nonoverlap = [], [], [], [], [], []
    hashes, calibrations, compatibility = {}, {}, []
    for vrs in sorted(raw.glob('*.vrs')):
        name = vrs.stem
        traj = raw / f'mps_{name}_vrs/slam/closed_loop_trajectory.csv'
        fmap = parsed / name / 'logs/214-1/mp4_to_vrs_time_ns.csv'
        entry = dict(recording=name, vrs=str(vrs), trajectory=str(traj), frame_map=str(fmap))
        if not traj.exists() or not fmap.exists():
            entry.update(used=False, reason='Missing closed-loop MPS or RGB timestamp map')
            entries.append(entry); continue
        provider = data_provider.create_vrs_data_provider(str(vrs))
        cal = provider.get_device_calibration().get_camera_calib('camera-rgb')
        ext = cal.get_transform_device_camera().to_matrix()
        calibrations[name] = dict(T_device_camera=ext.tolist(), source=str(vrs), vrs_bytes=vrs.stat().st_size)
        cols = ['graph_uid','tracking_timestamp_us','quality_score'] + [f'q{x}_world_device' for x in 'xyzw']
        df = pd.read_csv(traj, usecols=cols)
        ns = np.rint(df.tracking_timestamp_us.to_numpy()*1000).astype(np.int64)
        assert np.all(np.diff(ns) > 0), name
        table = pd.read_csv(fmap)
        times = np.rint(table.vrs_device_time_ns.to_numpy()).astype(np.int64)
        assert np.all(np.diff(times) >= 0), name
        duplicate_times = int(np.sum(np.diff(times)==0))
        right = np.clip(np.searchsorted(ns, times), 1, len(ns)-1)
        idx = np.where(abs(ns[right]-times)<abs(ns[right-1]-times),right,right-1)
        q = df[[f'q{x}_world_device' for x in 'xyzw']].to_numpy()[idx]
        gap = abs(ns[idx]-times)/1e6
        quality = df.quality_score.to_numpy()[idx]
        norm = np.linalg.norm(q, axis=1)
        good = (gap<=30)&(quality>=.5)&np.isfinite(q).all(1)&(norm>.9)&(norm<1.1)
        safeq = q.copy(); safeq[~good] = [0,0,0,1]
        R = Rotation.from_quat(safeq).as_matrix() @ ext[:3,:3]
        graphs = df.graph_uid.to_numpy()[idx]
        span = (times[-1]-times[0])/1e9
        valid_intervals = good[1:]&good[:-1]&(np.diff(times)>0)&(np.diff(times)<120_000_000)&(graphs[1:]==graphs[:-1])
        valid_seconds = float(np.diff(times)[valid_intervals].sum()/1e9)
        indices = np.arange(5,len(times)-15)
        fs = indices[:,None] + np.arange(-5,16)
        timing = abs((times[fs]-times[indices,None])/1e9 - np.arange(-5,16)*.1).max(1) <= .015
        keep = good[fs].all(1)&(graphs[fs]==graphs[indices,None]).all(1)&timing&(np.diff(times[fs],axis=1)>0).all(1)
        fs, indices = fs[keep], indices[keep]
        rel = np.einsum('nji,ntjk->ntik',R[indices],R[fs])
        rv = Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(-1,21,3)
        rindex = len(records)
        values.append(rv.astype(np.float32)); records.append(name)
        cutoffs.append(times[indices]); starts.append(indices)
        # Greedy spacing of complete two-second intervals for a sparse sensitivity check.
        sparse = np.zeros(len(indices),bool); end = -np.inf
        for j, t in enumerate(times[indices]/1e9):
            if t-.5 > end:
                sparse[j]=True; end=t+1.5
        nonoverlap.append(sparse)
        np.savez_compressed(OUT / f'poses/{name}.npz',R=R,times_ns=times,valid=good,quality=quality,graph=graphs.astype(str),frame_index=np.arange(len(times)))
        entry.update(used=True,record_index=rindex,frames=len(times),duplicate_rgb_times=duplicate_times,pose_rows=len(df),span_seconds=float(span),valid_seconds=valid_seconds,valid_frames=int(good.sum()),windows=len(indices),nonoverlap_windows=int(sparse.sum()),rejected_windows=int(len(keep)-keep.sum()),max_accepted_gap_ms=float(gap[good].max()))
        entries.append(entry)
        hashes[str(traj.relative_to(ROOT))]=sha(traj)
        hashes[str(fmap)]=sha(fmap)
        for cid in range(1,11):
            prior = ROOT / f'outputs/sam2_g1_v6_v8_20261001/{cid:02d}/input'
            meta = read(prior/'metadata.json')
            if meta['source']['recording_group'] != name: continue
            old = np.load(prior/'reference.npz')
            ii = np.searchsorted(times, old['times_ns'])
            assert np.array_equal(times[ii], old['times_ns'])
            diff=float(np.max(np.abs(R[ii]-old['R'])))
            assert diff < 1e-8, (name,diff)
            compatibility.append(dict(clip=f'{cid:02d}',max_R_difference=diff))
        print(name, 'seconds',round(span,1),'windows',len(indices), flush=True)
        del provider,df
    OUT.mkdir(exist_ok=True)
    x=np.concatenate(values); rec=np.concatenate([np.full(len(v),i,np.int16) for i,v in enumerate(values)])
    np.savez_compressed(OUT/'dataset.npz',rotation=x,record_index=rec,cutoff_ns=np.concatenate(cutoffs),frame_index=np.concatenate(starts),nonoverlap=np.concatenate(nonoverlap))
    oldfolds=read(OLD/'folds.json'); folds=[]
    for old in oldfolds:
        test, val=old['test'],old['validation']; train=[r for r in records if r not in [test,val]]
        folds.append(dict(fold=old['fold'],test=test,validation=val,train=train,
                          train_indices=np.flatnonzero(np.isin(rec,[records.index(r) for r in train])).tolist(),
                          val_indices=np.flatnonzero(rec==records.index(val)).tolist(),test_indices=np.flatnonzero(rec==records.index(test)).tolist(),
                          small_train_indices=old['camera_train_indices']))
    write(OUT/'inventory.json',dict(recordings=entries,record_names=records,total_windows=len(x),valid_seconds=sum(e.get('valid_seconds',0) for e in entries),span_seconds=sum(e.get('span_seconds',0) for e in entries),compatibility=compatibility))
    write(OUT/'calibrations.json',calibrations);write(OUT/'source_hashes.json',hashes);write(OUT/'folds.json',folds)
    protocol=dict(date='2026-10-02',task='Historical rotation 6 points at 10 Hz to future 15 points, rotation only',
                  sources='All locally available Aria recordings with closed-loop MPS, factory camera extrinsics and RGB timestamps. No person annotation required.',
                  window_stride_frames=1,quality_min=.5,max_pose_gap_ms=30,max_window_timestamp_deviation_ms=15,
                  model='Exactly the previous CameraPredictor: GRU(3,32), Linear(32,64), SiLU, Linear(64,45)',
                  inputs='Historical relative camera rotation only; last observed orientation defines reference. Offline MPS histories; not RGB-only or proven online poses.',
                  steps=2000,batch_size=256,learning_rate=.001,weight_decay=.01,gradient_clip=1.,validation_every=100,seeds=[17,29,43],
                  comparison=['small_matched','full'],loss='rotation-vector MSE',selection='Minimum full validation recording rotation-vector MSE; same validation set for both sizes',
                  sampling='Repeated random permutations of all training windows; equal optimizer update count and batch size across data-size arms',
                  split='Six existing test recordings individually held out; next original recording is validation; all other available recordings train. No frame/window split.',
                  test='Full held-out recordings and the original 117 person windows. Primary recording-equal mean SO(3) angular error; also sparse windows, horizon, motion bins.',
                  references=['frozen orientation','constant angular velocity least squares over 6 history points','old saved small-data camera'],
                  full_refit='After CV freeze, refit each seed on all available windows for max(median best full-model step of its six folds, one complete data pass). Refit checkpoints carry no held-out score.',
                  person_models='Remain frozen; optional replacement at inference is a compatibility diagnostic, not a newly trained trajectory method.',
                  no_test_tuning=True)
    write(OUT/'protocol.json',protocol)
    print('TOTAL',len(records),'recordings',len(x),'windows',round(sum(e.get('valid_seconds',0) for e in entries)/60,2),'valid minutes',flush=True)


if __name__=='__main__':
    (OUT/'poses').mkdir(parents=True,exist_ok=True)
    main()
