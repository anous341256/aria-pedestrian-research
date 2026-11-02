"""Independent SciPy ray projection of saved C/F predictions (Windows runtime)."""
from shared import *
from scipy.spatial.transform import Rotation


def main():
    data=dict(np.load(OLD/'dataset.npz'));cfg=read(OUT/'protocol.json');rows=[]
    for fold in read(OUT/'folds.json'):
        f=fold['fold'];idx=np.array(fold['test_indices'])
        for seed in cfg['seeds']:
            rv=np.load(CAMERA/f'fold_{f}/seed_{seed}/full/predictions.npz')['person_rotation']
            for v in ['C_separated','F_interacting_geometry']:
                z=np.load(OUT/f'fold_{f}/seed_{seed}/{v}/predictions_reviewed.npz');errors=[];roundtrips=[]
                for j,i in enumerate(idx):
                    K=data['K'][i];points=z['component'][j]
                    rays=np.linalg.solve(K,np.c_[points,np.ones(F)].T).T
                    R=Rotation.from_rotvec(rv[i].astype(np.float64)).as_matrix()
                    ray_future=np.einsum('tji,tj->ti',R,rays)
                    pixels=ray_future@K.T;pixels=pixels[:,:2]/pixels[:,2:]
                    errors.append(float(np.max(np.abs(pixels-z['pred'][j]))))
                    # Raw world-camera matrices independently validate history coordinate convention.
                    raw=data['R'][i].astype(np.float64);q=np.einsum('ij,tjk->tik',raw[P-1].T,raw)
                    xy=data['xy'][i].astype(np.float64)
                    ray0=np.linalg.solve(K,np.c_[xy,np.ones(P+F)].T).T
                    ray_stable=np.einsum('tij,tj->ti',q,ray0)
                    stable=ray_stable@K.T;stable=stable[:,:2]/stable[:,2:]
                    expected=warp(xy,data['H'][i])[0]
                    assert np.max(np.abs(stable-expected))<.01
                    restored=np.einsum('tji,tj->ti',q,ray_stable)@K.T
                    restored=restored[:,:2]/restored[:,2:]
                    roundtrips.append(float(np.max(np.abs(restored-xy))))
                assert max(errors)<.01 and max(roundtrips)<.01
                rows.append(dict(fold=f,seed=seed,variant=v,projection_max_px=max(errors),roundtrip_max_px=max(roundtrips)))
    write(OUT/'geometry_audit.json',dict(passed=True,implementation='Independent SciPy SO(3) and raw world-camera matrices',checks=rows,
                                       maximum_projection_difference_px=max(r['projection_max_px'] for r in rows)))
    print('Independent geometry check passed; max px',max(r['projection_max_px'] for r in rows))


if __name__=='__main__':main()
