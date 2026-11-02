"""Geometry, branch gradients, paired selection and frozen-base reproduction."""
from experiment import *
from joint_model import rotation_matrix, project, joint_losses
from scipy.spatial.transform import Rotation


def main():
    cfg=prepare();seed_all(17)
    data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');fold=read(OLD/'folds.json')[0]
    rv=torch.randn(4,3,15,3,device=DEVICE)*.2;rv[0,0]=0
    expected=Rotation.from_rotvec(rv.cpu().numpy().reshape(-1,3)).as_matrix().reshape(4,3,15,3,3)
    rotation_delta=float(np.max(np.abs(rotation_matrix(rv).cpu().numpy()-expected)))
    assert rotation_delta<1e-6
    c,p=load_backbones(0,17);idx=np.array(fold['test_indices']);x=inputs(data,meta,idx,c,p)
    saved=np.load(BASE/'fold_0/seed_17/separated_dual/predictions_reviewed.npz')
    component=(x['base_person']+x['origin'][:,None])*tensor(SCALE)
    delta=float(np.max(np.abs(component.cpu().numpy()-saved['component'])))
    assert delta<.003,delta
    full=np.load(CAM/'fold_0/seed_17/full/predictions.npz')['person_rotation'][idx]
    assert np.max(np.abs(full-x['base_camera'].cpu().numpy()))<1e-5
    short=dict(data)
    for key in ['xy','boxes','H','R','cam','seconds']:short[key]=data[key][:,:P].copy()
    xs=inputs(short,meta,idx,c,p)
    assert all(torch.equal(x[key],xs[key]) for key in x)
    tr=np.array(fold['train_indices']);xt=inputs(data,meta,tr,c,p);yt=target_tensors(data,tr,xt['origin'])
    model=JointModes(3,np.zeros((3,45)),np.ones(45)*.1,np.ones(30)*.02).to(DEVICE)
    total,details=loss(model,xt,yt,[.1,100.,100.]);total.backward()
    assert torch.isfinite(total)
    assert model.camera[-1].weight.grad.abs().sum()>0
    assert model.person[-1].weight.grad.abs().sum()>0
    assert not details['target'].requires_grad
    # Exactly aligned zero rotation has finite derivatives as well.
    zz=torch.zeros(2,15,3,device=DEVICE,requires_grad=True)
    angle_error(zz,torch.zeros_like(zz)).mean().backward();assert torch.isfinite(zz.grad).all()
    # Deliberately cancel wrong camera rotation with a wrong residual: final error
    # vanishes, but both branch penalties must remain nonzero.
    K=tensor(data['K'][:1]);wrong=torch.zeros(1,1,15,3,device=DEVICE);wrong[...,2]=.1
    truth=tensor(data['xy'][:1,6:]);hom=torch.cat([truth,torch.ones_like(truth[...,:1])],-1)
    H=K[:,None,None]@rotation_matrix(wrong)@torch.linalg.inv(K)[:,None,None]
    q=(H@hom[:,None,...,None]).squeeze(-1);comp=q[...,:2]/q[...,2:]
    penalty,terms=joint_losses(wrong,comp/tensor(SCALE),torch.zeros(1,1,device=DEVICE),
        torch.zeros(1,2,device=DEVICE),K,torch.zeros(1,15,3,device=DEVICE),truth/tensor(SCALE),truth,
        torch.zeros(1,dtype=torch.long,device=DEVICE),[.1,100.,100.])
    assert terms['eu'].max()<.01 and terms['er'].min()>.09 and terms['ep'].min()>1 and penalty>.3
    # Selected camera candidates must be exactly identical across same-time persons.
    z=infer(model,data,meta,idx,c,p)
    for g in np.unique(z['groups']):
        rows=np.flatnonzero(z['groups']==g)
        assert np.all(z['selected'][rows]==z['selected'][rows[0]])
        assert np.max(np.abs(z['camera_candidates'][rows]-z['camera_candidates'][rows[0]]))<1e-6
    report=dict(passed=True,rotation_matrix_max_error=rotation_delta,old_component_replay_max_px=delta,
                physical_future_truncation_exact=True,finite_camera_and_person_gradients=True,
                soft_score_targets_detached=True,same_time_camera_consistency=True,
                cancelling_wrong_branches_still_penalized=True)
    write(OUT/'preflight.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
