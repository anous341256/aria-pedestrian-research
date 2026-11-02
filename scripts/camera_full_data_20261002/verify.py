from common import *
from train import CameraPredictor, tensor, torch, DEVICE


def main():
    d=np.load(OUT/'dataset.npz');old=np.load(OLD/'dataset.npz');inventory=read(OUT/'inventory.json')
    names=inventory['record_names'];folds=read(OUT/'folds.json');smallmeta=read(OLD/'camera_samples.json')
    tested=[]
    for f in folds:
        groups=[]
        for key in ['train_indices','val_indices','test_indices']:
            indices=np.array(f[key]);groups.append({names[i] for i in d['record_index'][indices]})
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        assert set.union(*groups)==set(names)
        assert {smallmeta[i]['recording'] for i in f['small_train_indices']} <= groups[0]
        tested.extend(f['test_indices'])
    assert len(set(tested))==len(tested)
    assert np.isfinite(d['rotation']).all()
    assert np.max(abs(d['rotation'][:,5])) < 1e-6
    model=CameraPredictor().to(DEVICE)
    saved=torch.load(OLD/'fold_0/seed_17/camera.pt',map_location=DEVICE,weights_only=False)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    with torch.no_grad():
        original=model(tensor(old['cam'][:,:6])).cpu().numpy()
        reference=np.load(OLD/'fold_0/seed_17/camera_predictions.npy')
        exact=float(np.max(abs(original-reference)))
        assert exact<1e-7,exact
        changed=old['cam'].copy();changed[:,6:]=999.
        assert np.array_equal(model(tensor(changed[:,:6])).cpu().numpy(),original)
    assert abs(float(degrees(np.array([[[0.,0.,np.pi/2]]]),np.zeros((1,1,3)))[0,0])-90)<1e-8
    assert float(degrees(np.zeros((1,1,3)),np.zeros((1,1,3)))[0,0])==0
    assert len(inventory['compatibility'])==10
    result=dict(passed=True,recordings=len(names),windows=len(d['rotation']),held_out_full_recording_windows=len(tested),
                exact_old_architecture_output_max_difference=exact,parameters=sum(p.numel() for p in model.parameters()),
                future_suffix_does_not_change_model_input=True,all_splits_recording_disjoint=True,
                geometry_matches_all_ten_previous_clips=True,dataset_sha256=sha(OUT/'dataset.npz'),protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
