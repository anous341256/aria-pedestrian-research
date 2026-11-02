"""Checkpoint replay, independent geometry and frozen paired comparisons."""
from experiment import *
from physical_inference import infer
from scipy.spatial.transform import Rotation


def independent_project(component, rv, K):
    component=np.asarray(component,dtype=np.float64);rv=np.asarray(rv,dtype=np.float64);K=np.asarray(K,dtype=np.float64)
    R=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(*rv.shape[:-1],3,3)
    H=K[:,None]@np.swapaxes(R,-1,-2)@np.linalg.inv(K)[:,None]
    p=np.concatenate([component,np.ones((*component.shape[:-1],1))],-1)
    q=np.einsum('btij,btj->bti',H,p)
    return q[...,:2]/q[...,2:],q[...,2]


def angular(a,b):
    aa=Rotation.from_rotvec(a.reshape(-1,3));bb=Rotation.from_rotvec(b.reshape(-1,3))
    return np.rad2deg((aa.inv()*bb).magnitude()).reshape(a.shape[:-1])


def pixel_errors(p,y,depth=None):
    e=np.linalg.norm(p-y,axis=-1);invalid=~np.isfinite(e)
    if depth is not None:invalid|=depth<=1e-6
    e[invalid]=32000.
    return e,invalid


def verify_replay(a,b):
    # CUDA index_add uses atomic additions: group sums can differ by a few ulps.
    # All per-person logits, candidates, selected modes and output trajectories
    # must still match bit-for-bit; only the group reduction allows roundoff.
    delta=float(np.max(np.abs(a['group_logits']-b['group_logits'])))
    assert delta<1e-5,delta
    for key,v in a.items():
        if key!='group_logits':assert np.array_equal(v,b[key]),(key,float(np.max(np.abs(v-b[key]))))
    return delta


def load_model(dest):
    checkpoint=torch.load(dest/'checkpoint.pt',map_location=DEVICE,weights_only=False)
    assert checkpoint['protocol_sha256']==sha(OUT/'protocol.json')
    state=checkpoint['model'];k=checkpoint['modes']
    model=JointModes(k,state['anchors'].cpu(),state['camera_std'].cpu(),state['person_std'].cpu()).to(DEVICE)
    model.load_state_dict(state);model.eval();return model,checkpoint


def add_rows(rows, method, seed, idx, meta, pred, truth, depth=None, component=None, stable=None, camera=None, true_camera=None):
    e,bad=pixel_errors(pred,truth,depth)
    pe=np.linalg.norm(component-stable,axis=-1) if component is not None else None
    ce=angular(camera,true_camera) if camera is not None else None
    for j,i in enumerate(idx):
        row=dict(method=method,seed=seed,index=int(i),key=meta[i]['key'],recording=meta[i]['recording'],
                 ADE=float(e[j].mean()),FDE=float(e[j,-1]),invalid=int(bad[j].sum()),nonoverlap=meta[i]['nonoverlap'])
        if pe is not None:row['residual_ADE']=float(pe[j].mean())
        if ce is not None:row['camera_deg']=float(ce[j].mean())
        rows.append(row)


def summarize(rr):
    result={key:float(np.mean([r[key] for r in rr])) for key in ['ADE','FDE']}
    result['recording_equal_ADE']=float(np.mean([np.mean([r['ADE'] for r in rr if r['recording']==g]) for g in sorted({r['recording'] for r in rr})]))
    result['nonoverlap_ADE']=float(np.mean([r['ADE'] for r in rr if r['nonoverlap']]))
    result['seed_ADE']={str(s):float(np.mean([r['ADE'] for r in rr if r['seed']==s])) for s in sorted({r['seed'] for r in rr})}
    result['invalid']=sum(r['invalid'] for r in rr)
    for key in ['residual_ADE','camera_deg']:
        if key in rr[0]:result[key]=float(np.mean([r[key] for r in rr]))
    return result


def replay_attention(data,idx,camera,seed,fold,variant,z):
    spec=importlib.util.spec_from_file_location('old_attention_network',ROOT/'scripts/transformer_fusion_20261005/network.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    spec=importlib.util.spec_from_file_location('old_attention_shared',ROOT/'scripts/transformer_fusion_20261005/shared.py')
    sh=importlib.util.module_from_spec(spec);spec.loader.exec_module(sh)
    cp=torch.load(ATT/f'fold_{fold}/seed_{seed}/{variant}/checkpoint.pt',map_location=DEVICE,weights_only=False)
    m=mod.Predictor(variant,cp['ff_width']).to(DEVICE);m.load_state_dict(cp['model']);m.eval()
    short=dict(data)
    for key in ['xy','boxes','H','R','cam','seconds']:short[key]=data[key][:,:P]
    h=sh.history(short,idx,m.geometry)
    with torch.no_grad():
        cam=tensor(short['cam'][idx,:P]);seq,last=camera.encoder(cam)
        feats=torch.cat([cam,seq],-1) if m.uses_camera_features else None
        off=m(tensor(h['seq']),tensor(h['condition']),feats).cpu().numpy().reshape(len(idx),F,2)
        points=(off+h['origin'][:,None])*SCALE
        if m.geometry:
            rv=camera.head(last[-1]).reshape(-1,F,3).cpu().numpy()
            pred,depth=independent_project(points,rv,data['K'][idx])
        else:pred=points
    error=float(np.max(np.abs(pred-z['pred'])))
    assert error<.03,(variant,error)
    return error


def main():
    cfg=read(OUT/'protocol.json');data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json')
    rows=[];audits=[];diagnostics=[];coverage={};attention_replay=0.;geometry_error=0.
    variants=read(ATT/'protocol.json')['variants']
    for fold in read(OLD/'folds.json'):
        f=fold['fold'];idx=np.asarray(fold['test_indices']);truth=data['xy'][idx,P:]
        h=input_history(data,idx);stable=targets(data,idx,h['origin'])['stable'][:,P:]
        for seed in cfg['seeds']:
            c,p=load_backbones(f,seed)
            bc=np.load(CAM/f'fold_{f}/seed_{seed}/full/predictions.npz')['person_rotation'][idx]
            old=np.load(BASE/f'fold_{f}/seed_{seed}/separated_dual/predictions_reviewed.npz')
            assert np.array_equal(old['indices'],idx)
            x=inputs(data,meta,idx,c,p)
            assert np.max(np.abs(((x['base_person']+x['origin'][:,None])*tensor(SCALE)).cpu().numpy()-old['component']))<.003
            op,od=independent_project(old['component'],bc,data['K'][idx])
            assert np.max(np.abs(op-old['full']))<.03
            add_rows(rows,'old_GRU_flow',seed,idx,meta,op,truth,od,old['component'],stable,bc,data['cam'][idx,P:])
            hold=np.repeat(data['xy'][idx,P-1:P],F,axis=1)
            add_rows(rows,'hold_last',seed,idx,meta,hold,truth)
            for variant in variants:
                z=np.load(ATT/f'fold_{f}/seed_{seed}/{variant}/predictions_reviewed.npz')
                assert np.array_equal(z['indices'],idx)
                attention_replay=max(attention_replay,replay_attention(data,idx,c,seed,f,variant,z))
                geometry=variant[0] in 'CF'
                add_rows(rows,variant,seed,idx,meta,z['pred'],truth,z['depth'],z['component'] if geometry else None,
                         stable if geometry else None,bc if geometry else None,data['cam'][idx,P:] if geometry else None)
            for k in cfg['modes']:
                dest=OUT/f'fold_{f}/seed_{seed}/K{k}';model,checkpoint=load_model(dest)
                completed=read(dest/'complete.json')
                assert all(sha(Path(__file__).with_name(name))==digest for name,digest in completed['code_hashes'].items())
                saved=dict(np.load(dest/'predictions_test.npz'));assert np.array_equal(saved['indices'],idx)
                actual=infer(model,data,meta,idx,c,p)
                replay_reduction_delta=verify_replay(actual,saved)
                short=dict(data)
                for key in ['xy','boxes','H','R','cam','seconds']:short[key]=data[key][:,:P].copy()
                truncated=infer(model,short,meta,idx,c,p)
                suffix_reduction_delta=verify_replay(truncated,actual)
                # Invariance to row order, and exact same camera across same-time people.
                order=np.arange(len(idx))[::-1].copy();rev=infer(model,short,meta,idx[order],c,p)
                assert np.array_equal(rev['selected'][order],actual['selected'])
                assert np.max(np.abs(rev['pred'][order]-actual['pred']))<.03
                for g in np.unique(actual['groups']):
                    jj=np.flatnonzero(actual['groups']==g)
                    assert np.all(actual['selected'][jj]==actual['selected'][jj[0]])
                    assert np.max(np.abs(actual['camera'][jj]-actual['camera'][jj[0]]))<1e-6
                # Reconstruct all branch scales using only the locked training indices.
                tri=np.asarray(fold['train_indices']);tx=inputs(data,meta,tri,c,p);ty=target_tensors(data,tri,tx['origin'])
                trcomponent=(tx['base_person']+tx['origin'][:,None]).cpu().numpy()*SCALE
                tp,_=independent_project(trcomponent,tx['base_camera'].cpu().numpy(),data['K'][tri])
                independent_scales=[max(float(np.deg2rad(angular(tx['base_camera'].cpu().numpy(),data['cam'][tri,P:])).mean()),np.pi/180),
                    max(float(np.linalg.norm((tx['base_person']-ty['true_residual']).cpu().numpy()*SCALE,axis=-1).mean()),10.),
                    max(float(np.linalg.norm(tp-data['xy'][tri,P:],axis=-1).mean()),10.)]
                assert np.allclose(checkpoint['scales'],independent_scales,rtol=2e-5,atol=.002)
                pred,depth=independent_project(actual['component'],actual['camera'],data['K'][idx])
                delta=float(np.max(np.abs(pred-actual['pred'])));geometry_error=max(geometry_error,delta)
                assert delta<.05,delta
                add_rows(rows,f'joint_K{k}',seed,idx,meta,pred,truth,depth,actual['component'],stable,actual['camera'],data['cam'][idx,P:])
                # Counterfactual diagnostic: same person with reference future camera.
                gt,gd=independent_project(actual['component'],data['cam'][idx,P:],data['K'][idx])
                add_rows(rows,f'joint_K{k}_GTcamera_diagnostic',seed,idx,meta,gt,truth,gd)
                person_error=gt-truth;camera_error=pred-gt
                cross=2*(person_error*camera_error).sum(-1)
                oldgt,_=independent_project(old['component'],data['cam'][idx,P:],data['K'][idx])
                oldcross=2*((oldgt-truth)*(op-oldgt)).sum(-1)
                all_e=[];all_joint=[]
                for mode in range(k):
                    pp,dd=independent_project(actual['component_candidates'][:,mode],actual['camera_candidates'][:,mode],data['K'][idx])
                    candidate_delta=float(np.max(np.abs(pp-actual['image_candidates'][:,mode])))
                    geometry_error=max(geometry_error,candidate_delta)
                    assert candidate_delta<.05,(f,seed,k,mode,candidate_delta)
                    ee,_=pixel_errors(pp,truth,dd);all_e.append(ee.mean(-1))
                    er=np.deg2rad(angular(actual['camera_candidates'][:,mode],data['cam'][idx,P:])).mean(-1)
                    ep=np.linalg.norm(actual['component_candidates'][:,mode]-stable,axis=-1).mean(-1)
                    ss=checkpoint['scales'];all_joint.append((er/ss[0]+ep/ss[1]+ee.mean(-1)/ss[2])/3)
                all_e=np.stack(all_e,-1);all_joint=np.stack(all_joint,-1)
                for g in np.unique(actual['groups']):
                    jj=np.flatnonzero(actual['groups']==g);oracle=int(all_joint[jj].mean(0).argmin());pick=int(actual['selected'][jj[0]])
                    for j in jj:
                        diagnostics.append(dict(method=f'joint_K{k}',seed=seed,index=int(idx[j]),selected=pick,oracle_joint_mode=oracle,
                            selected_ADE=float(all_e[j,pick]),oracle_joint_ADE=float(all_e[j,oracle]),minADE_K_diagnostic=float(all_e[j].min()),
                            normalized_selection_regret=float(all_joint[jj].mean(0)[pick]-all_joint[jj].mean(0)[oracle]),
                            camera_candidate_separation_deg=float(np.mean([angular(actual['camera_candidates'][j:j+1,a],actual['camera_candidates'][j:j+1,b]).mean() for a in range(k) for b in range(a+1,k)])) if k>1 else 0.,
                            selected_cross_term_px2=float(cross[j].mean()),old_cross_term_px2=float(oldcross[j].mean()),
                            negative_cross_fraction=float((cross[j]<0).mean())))
                audits.append(dict(fold=f,seed=seed,modes=k,selected_predictions_replay_exact=True,
                    future_truncated_selected_predictions_exact=True,group_reduction_max_roundoff=max(replay_reduction_delta,suffix_reduction_delta),
                    shared_camera=True,scale_recomputed=True,geometry_max_px=delta))
                coverage.setdefault((seed,k),[]).extend(idx.tolist())
            print(f'audited fold={f} seed={seed}',flush=True)
    assert all(sorted(v)==list(range(117)) for v in coverage.values())
    summary={m:summarize([r for r in rows if r['method']==m]) for m in sorted({r['method'] for r in rows})}
    comparisons={}
    for baseline in ['joint_K1','old_GRU_flow','A_early_fusion','D_bidirectional','E_oneway','F_interacting_geometry']:
        new={(r['seed'],r['index']):r for r in rows if r['method']=='joint_K3'}
        old={(r['seed'],r['index']):r for r in rows if r['method']==baseline}
        keys=sorted(new);change=np.array([new[key]['ADE']-old[key]['ADE'] for key in keys])
        item=dict(ADE_change=float(change.mean()),improved_predictions=int((change<0).sum()),predictions=len(keys))
        if 'camera_deg' in old[keys[0]]:
            item['all_three_improved_predictions']=sum(new[key]['ADE']<old[key]['ADE'] and new[key]['residual_ADE']<old[key]['residual_ADE'] and new[key]['camera_deg']<old[key]['camera_deg'] for key in keys)
            item['image_improved_both_branches_worse']=sum(new[key]['ADE']<old[key]['ADE'] and new[key]['residual_ADE']>old[key]['residual_ADE'] and new[key]['camera_deg']>old[key]['camera_deg'] for key in keys)
        comparisons[baseline]=item
    dg={}
    for k in [1,3]:
        rr=[r for r in diagnostics if r['method']==f'joint_K{k}']
        dg[f'K{k}']={key:float(np.mean([r[key] for r in rr])) for key in ['oracle_joint_ADE','minADE_K_diagnostic','normalized_selection_regret','camera_candidate_separation_deg','selected_cross_term_px2','old_cross_term_px2','negative_cross_fraction']}
        dg[f'K{k}']['selection_matches_joint_oracle']=float(np.mean([r['selected']==r['oracle_joint_mode'] for r in rr]))
        dg[f'K{k}']['mode_counts']={str(i):sum(r['selected']==i for r in rr) for i in range(k)}
    protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/p)==v for p,v in protected.items())
    write(OUT/'metrics.json',rows);write(OUT/'summary.json',summary);write(OUT/'comparisons.json',comparisons)
    write(OUT/'diagnostic_rows.json',diagnostics);write(OUT/'diagnostics.json',dg)
    capacities={}
    for k in cfg['modes']:
        record=read(OUT/f'fold_0/seed_17/K{k}/complete.json')
        capacities[f'K{k}']=dict(new_trainable_parameters=record['adapter_parameters'],
            frozen_backbone_allocated_parameters=record['frozen_backbone_parameters'],
            frozen_backbone_inference_parameters=84872,
            total_active_inference_parameters=84872+record['adapter_parameters'])
    write(OUT/'capacity.json',capacities)
    write(OUT/'verification.json',dict(passed=True,models=audits,protected_files=len(protected),attention_checkpoints_replayed=108,
        attention_max_replay_px=attention_replay,geometry_max_px=geometry_error,windows=117,seeds=cfg['seeds']))
    lines=['共享模式联合候选实验 / Joint paired modes — 2026-10-07','',
        '范围：原始117窗口、6个录制隔离折、种子17/29/43；保留现有GRU/flow主干权重，增加小型候选与评分头。',
        'K3：三个共享模式同时生成相机与行人未来；K1：同样新增训练预算的单候选对照。',
        '相机候选预训练2000次；联合训练240次，同时回放全量允许的相机训练数据。',
        '相机角度、行人残差、最终像素误差按训练集尺度归一化等权；同一时刻所有可用目标共同选择一个模式。',
        '旧模型与attention对照原权重回放；这不是参数/训练预算完全匹配的attention因果消融。新K1/K3均有额外训练，不能归因全部收益于多模态。','',
        'New adapter / total active inference parameters: '+json.dumps(capacities,ensure_ascii=False),'',
        'Method | ADE px | FDE px | recording-equal ADE | residual ADE px | camera deg | seed17/29/43 ADE']
    for name,s in summary.items():
        if 'diagnostic' in name:continue
        lines.append(f'{name} | {s["ADE"]:.2f} | {s["FDE"]:.2f} | {s["recording_equal_ADE"]:.2f} | {s.get("residual_ADE",float("nan")):.2f} | {s.get("camera_deg",float("nan")):.3f} | '+ '/'.join(f'{v:.2f}' for v in s['seed_ADE'].values()))
    lines+=['','K3 minus comparator (negative = lower ADE):',json.dumps(comparisons,ensure_ascii=False,indent=2),
        '', 'Candidate-selection / error-cancellation diagnostics:',json.dumps(dg,ensure_ascii=False,indent=2),
        '', 'Oracle/minADE and GT-camera results use future targets and are diagnostic only. Mode numbers are fold-local, not universal behavioral labels.',
        'Negative error cross terms indicate geometric error cancellation, not intention; this objective penalizes branch errors but does not guarantee zero cancellation.',
        'Three seeds do not make overlapping windows independent. Labels remain offline reviewed SAM references. Only rotation is compensated.',
        f'Verified: 36 new checkpoint replays, physical future removal, same-time shared camera, independent branch scales; 108 old attention checkpoint replays; {len(protected)} source files unchanged.',
        f'Independent geometry max difference: {geometry_error:.6f}px. Old selected baseline unchanged.']
    dest=ROOT/'reports/joint_modes_20261007';dest.mkdir(parents=True,exist_ok=True)
    (dest/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
