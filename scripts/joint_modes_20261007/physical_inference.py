"""Exact evaluation projection; training keeps its finite stabilized surrogate.

Scores/modes are never changed by this function. Invalid rays remain explicit
and receive the pre-existing 32000px evaluation penalty, not silently repaired.
"""
from experiment import infer as stabilized_infer, OUT, OLD, read, write, np, sha
from common import project_future


def exact_projection(result, intrinsics):
    z=dict(result)
    images=[];depths=[]
    for j,K in enumerate(intrinsics):
        parts=[project_future(points.astype(np.float64),rv.astype(np.float64),K.astype(np.float64)) for points,rv in zip(z['component_candidates'][j],z['camera_candidates'][j])]
        images.append(np.stack([p[0] for p in parts]));depths.append(np.stack([p[1] for p in parts]))
    z['image_candidates']=np.stack(images);z['depth_candidates']=np.stack(depths)
    rows=np.arange(len(images));z['pred']=z['image_candidates'][rows,z['selected']]
    z['depth']=z['depth_candidates'][rows,z['selected']]
    return z


def infer(model,data,meta,idx,camera,person):
    return exact_projection(stabilized_infer(model,data,meta,idx,camera,person),data['K'][idx])


def finalize():
    data=dict(np.load(OLD/'dataset.npz'));records=[]
    for path in sorted(OUT.glob('fold_*/seed_*/K*/predictions_*.npz')):
        if path.stem not in ['predictions_test','predictions_validation']:continue
        archive=path.with_name(path.stem+'_initial_stabilized.npz')
        if not archive.exists():path.rename(archive)
        source=dict(np.load(archive));z=exact_projection(source,data['K'][source['indices']])
        for key in ['camera_candidates','component_candidates','logits','group_logits','selected','camera','component']:
            assert np.array_equal(z[key],source[key])
        np.savez_compressed(path,**z)
        records.append(dict(file=str(path.relative_to(OUT)),source_sha256=sha(archive),
            invalid_candidate_rays=int((z['depth_candidates']<=1e-6).sum()),invalid_selected_rays=int((z['depth']<=1e-6).sum()),
            max_physical_projection_change_px=float(np.max(np.abs(z['image_candidates']-source['image_candidates'])))))
    write(OUT/'projection_finalization.json',dict(reason='Evaluation must use exact projective geometry, rather than the training loss safe denominator. Detected during independent geometry audit, before aggregate metrics. No model or selected mode changed.',files=records))
    print('Finalized',len(records),'prediction files; selected invalid rays',sum(r['invalid_selected_rays'] for r in records))


if __name__=='__main__':finalize()
