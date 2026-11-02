"""Frozen held-out camera evaluation; no parameter selection from test scores."""
from common import *
from train import CameraPredictor, infer, torch, DEVICE
import cv2


def constant_velocity(rotation):
    t=np.arange(-5,1)*.1
    omega=np.einsum('t,ntd->nd',t,rotation[:,:6])/np.sum(t*t)
    return omega[:,None,:]*(np.arange(1,16)*.1)[None,:,None]


def summarize(error, record_index, sparse, target):
    # N x seeds x 15. Seeds averaged only after calculating each prediction's error.
    value=error.mean(axis=1)
    def stats(e, ids):
        if len(e)==0:return None
        by={str(int(r)):e[ids==r].mean(axis=0).tolist() for r in np.unique(ids)}
        equal=np.mean(list(by.values()),axis=0)
        return dict(windows=len(e),recordings=len(by),window_mean_degrees=float(e.mean()),
                    recording_equal_mean_degrees=float(equal.mean()),
                    recording_equal_final_degrees=float(equal[-1]),curve_degrees=equal.tolist(),
                    per_recording={r:dict(mean_degrees=float(np.mean(v)),final_degrees=float(v[-1])) for r,v in by.items()})
    result=stats(value,record_index)
    result['per_seed_recording_equal_mean']=[stats(error[:,s],record_index)['recording_equal_mean_degrees'] for s in range(error.shape[1])]
    result['sparse']=stats(value[sparse],record_index[sparse])
    turn=np.rad2deg(np.linalg.norm(target[:,-1],axis=-1))
    result['motion_bins']={label:stats(value[mask],record_index[mask]) for label,mask in [
        ('final_rotation_under_5deg',turn<5),('final_rotation_5_to_15deg',(turn>=5)&(turn<15)),('final_rotation_at_least_15deg',turn>=15)]}
    return result


def project(points,rv,K):
    rotations=np.stack([cv2.Rodrigues(v.astype(np.float64))[0] for v in rv])
    H=K@rotations.transpose(0,2,1)@np.linalg.inv(K)
    p=np.einsum('tij,tj->ti',H,np.column_stack([points,np.ones(len(points))]))
    return p[:,:2]/p[:,2:],p[:,2]


def main():
    cfg=read(OUT/'protocol.json');d=np.load(OUT/'dataset.npz');old=np.load(OLD/'dataset.npz')
    folds=read(OUT/'folds.json');oldfolds=read(OLD/'folds.json');names=read(OUT/'inventory.json')['record_names']
    meta=read(OLD/'samples.json')
    cameras={name:[] for name in ['old_small','small_matched','full']}
    person_cameras={name:[] for name in cameras};testids=[];personids=[];downstream=[]
    for fold in folds:
        idx=np.array(fold['test_indices']);testids.extend(idx.tolist())
        ii=np.array(oldfolds[fold['fold']]['test_indices']);personids.extend(ii.tolist())
        for name in cameras:
            a=[];b=[]
            for seed in cfg['seeds']:
                if name=='old_small':
                    folder=OLD/f'fold_{fold["fold"]}'/f'seed_{seed}'
                    model=CameraPredictor().to(DEVICE)
                    model.load_state_dict(torch.load(folder/'camera.pt',map_location=DEVICE,weights_only=False)['model'])
                    p=infer(model,d['rotation'][idx,:6]);person=np.load(folder/'camera_predictions.npy')[ii]
                else:
                    folder=OUT/f'fold_{fold["fold"]}'/f'seed_{seed}'/name
                    c=read(folder/'complete.json');assert c['protocol_sha256']==sha(OUT/'protocol.json')
                    assert c['unique_train_windows_seen']==c['training_windows']
                    z=np.load(folder/'predictions.npz');assert np.array_equal(z['indices'],idx)
                    p=z['rotation'];person=z['person_rotation'][ii]
                a.append(p);b.append(person)
                for variant in ['comp_future','comp_dual']:
                    frozen=np.load(OLD/f'fold_{fold["fold"]}'/f'seed_{seed}'/variant/'predictions_reviewed.npz')
                    assert np.array_equal(frozen['indices'],ii)
                    for j,i in enumerate(ii):
                        xy,depth=project(frozen['component'][j],person[j],old['K'][i])
                        err=np.linalg.norm(xy-old['xy'][i,6:],axis=-1)
                        bad=~np.isfinite(err)|(depth<=1e-6);err[bad]=32000
                        downstream.append(dict(key=meta[i]['key'],recording=meta[i]['recording'],seed=seed,
                                               variant=variant,camera=name,ADE=float(err.mean()),FDE=float(err[-1]),invalid=int(bad.sum())))
            cameras[name].append(np.stack(a,axis=1));person_cameras[name].append(np.stack(b,axis=1))
    idx=np.array(testids);ii=np.array(personids);records=d['record_index'][idx];sparse=d['nonoverlap'][idx]
    y=d['rotation'][idx,6:];persony=old['cam'][ii,6:]
    for dic,rot in [(cameras,d['rotation'][idx]),(person_cameras,old['cam'][ii])]:
        for name in list(dic):dic[name]=np.concatenate(dic[name])
        dic['frozen_orientation']=np.zeros((len(rot),1,15,3),np.float32)
        dic['constant_angular_velocity']=constant_velocity(rot)[:,None]
    summary={};errors={}
    for name,p in cameras.items():
        err=degrees(p,y[:,None]);assert np.isfinite(err).all();errors[name]=err
        summary[name]=summarize(err,records,sparse,y)
    personsummary={};perrec=np.array([names.index(meta[i]['recording']) for i in ii]);pernon=np.array([meta[i]['nonoverlap'] for i in ii])
    for name,p in person_cameras.items():
        err=degrees(p,persony[:,None]);personsummary[name]=summarize(err,perrec,pernon,persony)
    result={}
    for variant in ['comp_future','comp_dual']:
        result[variant]={}
        for name in ['old_small','small_matched','full']:
            rr=[r for r in downstream if r['variant']==variant and r['camera']==name]
            by={rec:float(np.mean([r['ADE'] for r in rr if r['recording']==rec])) for rec in {r['recording'] for r in rr}}
            result[variant][name]=dict(window_ADE=float(np.mean([r['ADE'] for r in rr])),recording_equal_ADE=float(np.mean(list(by.values()))),per_recording=by,
                                       invalid_steps=sum(r['invalid'] for r in rr))
    write(OUT/'camera_metrics.json',dict(full_recording_test=summary,original_person_windows=personsummary,record_names=names,seeds=cfg['seeds']))
    write(OUT/'fixed_person_reprojection.json',dict(note='Only replace final projection rotations. Person component prediction and its conditioning stay frozen from V1; not a retrained or fully reconditioned forecasting model.',summary=result,rows=downstream))
    np.savez_compressed(OUT/'held_out_predictions.npz',indices=idx,target=y,**cameras)
    np.savez_compressed(OUT/'held_out_errors.npz',indices=idx,**errors)
    np.savez_compressed(OUT/'person_camera_predictions.npz',indices=ii,target=persony,**person_cameras)
    print('FULL RECORDING TEST',json.dumps({k:dict(mean=v['recording_equal_mean_degrees'],final=v['recording_equal_final_degrees'],sparse=v['sparse']['recording_equal_mean_degrees']) for k,v in summary.items()}))
    print('ORIGINAL WINDOWS',json.dumps({k:dict(mean=v['window_mean_degrees'],recording=v['recording_equal_mean_degrees']) for k,v in personsummary.items()}))
    print('FIXED PERSON REPROJECTION',json.dumps(result))


if __name__=='__main__':main()
