"""Export native, strictly historical gaze rays. Run with /opt/aria-vrs."""
from pathlib import Path
import json, csv, hashlib, shutil
import numpy as np
from projectaria_tools.core import data_provider, calibration, mps
from projectaria_tools.core.mps.utils import get_eyegaze_point_at_depth

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/gaze_attention_20261008'
OLD=ROOT/'outputs/biflow_aria_v1_20261002'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Do not replace frozen protocol'
    meta=read(OLD/'samples.json'); data=dict(np.load(OLD/'dataset.npz'))
    protected=[OLD/'dataset.npz',OLD/'samples.json',ROOT/'outputs/transformer_fusion_20261005/folds.json']
    cache={}; audit=[]; windows=[]; refs={}; max_projection_error=0.
    for rec in sorted({m['recording'] for m in meta}):
        vrs=ROOT/'data/Aria/raw_vrs'/f'{rec}.vrs'
        csvpath=vrs.parent/f'mps_{rec}_vrs/eye_gaze/general_eye_gaze.csv'
        protected.append(csvpath)
        provider=data_provider.create_vrs_data_provider(str(vrs))
        dev=provider.get_device_calibration(); src=dev.get_camera_calib('camera-rgb')
        w,h=map(int,src.get_image_size())
        dst=calibration.get_linear_camera_calibration(w,h,float(src.get_focal_lengths()[0]),'pinhole',src.get_transform_device_camera())
        K=np.eye(3); K[0,0],K[1,1]=dst.get_focal_lengths();K[:2,2]=dst.get_principal_point()
        R=(dst.get_transform_device_camera().inverse()@dev.get_transform_device_cpf()).rotation().to_matrix()
        with csvpath.open() as f: rows=list(csv.DictReader(f))
        ts=np.array([int(r['tracking_timestamp_us'])*1000 for r in rows],dtype=np.int64)
        gaze=mps.read_eyegaze(str(csvpath));assert len(gaze)==len(ts) and (np.diff(ts)>0).all()
        xy=np.zeros((len(ts),2));valid=np.zeros(len(ts),bool)
        for i,g in enumerate(gaze):
            assert abs(int(g.tracking_timestamp.total_seconds()*1e9)-int(ts[i]))<=1000
            if not g.combined_gaze_valid or not np.isfinite([g.yaw,g.pitch]).all(): continue
            ray=R@np.asarray(get_eyegaze_point_at_depth(g.yaw,g.pitch,1.0))
            pixel=dst.project(ray)
            if pixel is None or ray[2]<=0: continue
            direct=K@ray;direct=direct[:2]/direct[2]
            max_projection_error=max(max_projection_error,float(np.max(np.abs(direct-pixel))))
            xy[i]=pixel/np.array([w,h]);valid[i]=True
        cache[rec]=(ts,xy,valid,K)
        audit.append(dict(recording=rec,samples=len(ts),median_interval_ms=float(np.median(np.diff(ts))/1e6),
                          valid_direction_in_image=int(valid.sum()),image_size=[w,h]))
        print(rec,len(ts),int(valid.sum()),flush=True)
    for i,m in enumerate(meta):
        clip=m['clip'];p=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{clip}/input/reference.npz'
        if clip not in refs:refs[clip]=dict(np.load(p));protected.append(p)
        alltimes=refs[clip]['times_ns']; frames=np.asarray(m['frames'][:6],dtype=int)
        start,end=map(int,alltimes[frames[[0,-1]]]);ts,xy,valid,K=cache[m['recording']]
        assert np.allclose(K,data['K'][i],atol=.001)
        chosen=np.flatnonzero((ts>=start)&(ts<=end))
        # Repeating selection after physically removing future samples is identical.
        trunc=ts[ts<=end];assert np.array_equal(chosen,np.flatnonzero((trunc>=start)&(trunc<=end)))
        windows.append(dict(indices=chosen,xy=xy[chosen],valid=valid[chosen],time=(ts[chosen]-end)/1e9,
                            timestamp_ns=ts[chosen],start_ns=start,cutoff_ns=end))
    slots=max(len(w['indices']) for w in windows);n=len(windows)
    values=np.zeros((n,slots,3),np.float32);mask=np.zeros((n,slots),bool)
    timestamps=np.zeros((n,slots),np.int64)
    for i,w in enumerate(windows):
        count=len(w['indices']);values[i,:count,:2]=w['xy'];values[i,:count,2]=w['time']/.5
        mask[i,:count]=w['valid'];values[i,~mask[i]]=0;timestamps[i,:count]=w['timestamp_ns']
    np.savez_compressed(OUT/'gaze.npz',values=values,valid=mask,timestamps_ns=timestamps,
                        start_ns=[w['start_ns'] for w in windows],cutoff_ns=[w['cutoff_ns'] for w in windows])
    write(OUT/'gaze_audit.json',dict(recordings=audit,windows=n,slots=slots,
        raw_counts=[len(w['indices']) for w in windows],valid_counts=mask.sum(1).tolist(),
        zero_valid_windows=int((mask.sum(1)==0).sum()),projection_max_px=max_projection_error,
        future_truncation_selection_passed=True,
        semantics='General MPS gaze direction projected into rectified RGB; parallel-ray approximation, not depth-calibrated fixation. No interpolation; inclusive historical frame-time interval only.'))
    shutil.copyfile(ROOT/'outputs/transformer_fusion_20261005/folds.json',OUT/'folds.json')
    protected += list((ROOT/'scripts/transformer_fusion_20261005').glob('*.py'))
    write(OUT/'protected_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in protected})
    cfg=dict(version='gaze_attention_20261008',arms=['without_gaze','with_gaze'],windows=n,
        seeds=[17,29,43],folds=6,epochs=120,updates_per_epoch=2,batch_size=48,validation_every=10,
        learning_rate=.001,weight_decay=.01,gradient_clip=1.,hidden=64,heads=4,ff=128,
        gaze_slots=slots,dropout=0.,history_frames=6,future_frames=15,
        inputs='Raw historical image centers, historical relative camera rotation vectors, native-timestamp historical gaze image direction.',
        architecture='Two self-attention blocks per stream; one simultaneous cross-attention block per stream reading both other streams. Three learned summary tokens concatenate with last position/calibration context and decode 15 future image offsets. No separate motion output heads or reprojection.',
        ablation='Identical registered architecture/initialization. Without gaze all gaze data tokens are zeroed and masked, including timestamps/validity; the learned summary token remains as a content-free fusion query. Same folds, batch draws, augmentation, optimizer, update count, selection criterion.',
        camera='Raw historical rotation only; no pretrained camera network or future camera forecasts.',
        selection='Minimum validation image ADE every 10 epochs; test recording excluded from training and validation.',
        dataset='Exact original117 to match the recent architecture experiments; reviewed offline SAM references, not independent human GT.',
        reporting='Two trained arms only; ADE/FDE, per-seed, recording-equal, sparse diagnostics; hold-last contextual reference. No tuning based on test scores.')
    write(OUT/'protocol.json',cfg)
    print('READY',n,'slots',slots,'valid counts',np.unique(mask.sum(1),return_counts=True),flush=True)

if __name__=='__main__': main()
