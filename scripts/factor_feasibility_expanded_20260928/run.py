"""Expanded frozen evaluation: use the unchanged pilot predictor functions."""
import sys,inspect,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare import ROOT,SRC,OUT,read,write,sha
sys.path.insert(0,str(ROOT/'scripts/factor_feasibility_20260928'))
import analyze as core
core.OUT=OUT
METHODS=core.METHODS

def main():
    assert not (OUT/'summary.json').exists(),'Completed results are preserved.'
    proposals=read(OUT/'proposed_candidates.json');review=read(Path(__file__).with_name('review_decisions.json'))
    write(OUT/'candidate_review.json',review)
    specs=[s for s in proposals if s['key'] not in review['excluded']]
    assert len(specs)==14 and sum(not s['previous_pilot'] for s in specs)==10
    for i,s in enumerate(specs):s['case_number']=i+1;s['review_notes']=review['accepted_notes'][s['key']]
    write(OUT/'candidates.json',specs)
    protocol=read(ROOT/'outputs/factor_feasibility_20260928/protocol.json')
    protocol.update(selection='Longest nonempty contiguous run >=41 frames for each of all 25 SAM IDs in five videos; 18 candidates, pre-score visual review excluded 4, 14 retained.',
        expansion_rule=read(OUT/'selection_rule.json'),candidate_review_sha256=sha(OUT/'candidate_review.json'),
        case_unit='Recording-local SAM identity segment, not necessarily distinct person across recordings.',
        sensitivity_expanded='Old four / new ten separately; equal-case and equal-recording means; nonoverlap, no-edge, common projectable, rotation strata and angular error unchanged. Report all cases and failures.',
        angular_check='Viewing-ray angular error over all predictions, including behind-camera rays; predeclared for expansion based on pilot diagnostic.',
        frozen_predictor_sha256=hashlib.sha256(inspect.getsource(core.predict_history).encode()).hexdigest(),
        original_pilot_summary_sha256=sha(ROOT/'outputs/factor_feasibility_20260928/summary.json'))
    write(OUT/'protocol.json',protocol);write(OUT/'verification.json',core.verification())
    cache={};manifest=[]
    for sid in sorted({s['recording'] for s in specs}):
        print('ALIGN',sid,flush=True);*cache[sid],info=core.align(sid);manifest.append(info)
    rows=[];excluded=[];examples=[]
    for s in specs:
        R,K,ns,audit=cache[s['recording']]
        annotations=SRC/s['recording']/'sam2_raw_v1/tracks.json';assert sha(annotations)==s['source_annotation_sha256']
        track=next(t for t in read(annotations)['tracks'] if t['id']==s['track'])
        boxes={r['frame']:r['bbox_xyxy'] for r in track['frames']}
        for cutoff in range(s['start']+5,s['end']-15+1,5):
            fs=np.arange(cutoff-5,cutoff+16);rr=R[fs]
            status='valid'
            if not np.isfinite(rr).all():status='invalid_reference_pose'
            elif len(set(audit[f]['graph'] for f in fs))!=1:status='reference_graph_boundary'
            if status!='valid':excluded.append({'key':s['key'],'frame':cutoff,'reason':status});continue
            b=np.array([boxes[int(f)] for f in fs],float);points=(b[:,:2]+b[:,2:])/2
            seconds=(ns[fs[:6]]-ns[cutoff])/1e9
            pred,ped,hist,omega=core.predict_history(points[:6],rr[:6],K,seconds)
            pred['comp_cv_refcam']=core.recompose_reference(ped,rr[5],rr[6:],K)
            rotation_deg=float(np.max(np.linalg.norm(Rotation.from_matrix(np.array([rr[5].T@r for r in rr[6:]])).as_rotvec(),axis=1))*180/np.pi)
            row={'key':s['key'],'recording':s['recording'],'track':s['track'],'frame':cutoff,'case_number':s['case_number'],'previous_pilot':s['previous_pilot'],
                 'history_frames':fs[:6].tolist(),'future_frames':fs[6:].tolist(),'history_xy':points[:6].tolist(),
                 'history_normalized_xy':core.serialize(hist),'history_xyxy':b[:6].tolist(),'target_xy':points[6:].tolist(),
                 'pedestrian_forecast_normalized_xy':core.serialize(ped),'predictions':{m:core.serialize(p) for m,p in pred.items()},
                 'metrics':{m:core.metrics(p,points[6:]) for m,p in pred.items()},'camera_omega_rad_s':omega.tolist(),
                 'max_future_rotation_deg':rotation_deg,'edge':bool(((b[:,0]<=5)|(b[:,1]<=5)|(b[:,2]>=2555)|(b[:,3]>=1915)).any()),
                 'nonoverlap':bool((cutoff-(s['start']+5))%20==0),'actual_horizon_s':float((ns[cutoff+15]-ns[cutoff])/1e9)}
            # Intrinsics used only as the same privileged geometry diagnostic.
            ki=np.linalg.inv(K);target_rays=np.c_[points[6:],np.ones(15)]@ki.T;target_rays/=np.linalg.norm(target_rays,axis=1,keepdims=True)
            row['angular_metrics']={}
            for m in METHODS:
                if m.startswith('comp_'):
                    rays=np.c_[ped,np.ones(15)]@ki.T
                    rotations=np.array([r.T@rr[5] for r in rr[6:]]) if m.endswith('refcam') else Rotation.from_rotvec(-np.arange(1,16)[:,None]/10*omega).as_matrix()
                    rays=np.einsum('nij,nj->ni',rotations,rays)
                else:rays=np.c_[pred[m],np.ones(15)]@ki.T
                rays/=np.linalg.norm(rays,axis=1,keepdims=True)
                angles=np.degrees(np.arccos(np.clip(np.sum(rays*target_rays,axis=1),-1,1)))
                row['angular_metrics'][m]={'mean_deg':float(angles.mean()),'final_deg':float(angles[-1])}
            rows.append(row)
        mine=[r for r in rows if r['key']==s['key']]
        if mine:examples.append(min(mine,key=lambda r:abs(r['frame']-(s['start']+50))))
        print('CASE',s['key'],'windows',len(mine),{m:round(core.aggregate(mine)['methods'][m]['ADE_1.5s'],1) for m in METHODS} if mine else 'NO VALID REFERENCE',flush=True)
    a=core.aggregate
    summary={'overall':a(rows),'per_case':{s['key']:a([r for r in rows if r['key']==s['key']]) for s in specs},
       'per_recording':{sid:a([r for r in rows if r['recording']==sid]) for sid in cache},
       'new_only':a([r for r in rows if not r['previous_pilot']]),'pilot_only':a([r for r in rows if r['previous_pilot']]),
       'nonoverlap':a([r for r in rows if r['nonoverlap']]),'no_edge':a([r for r in rows if not r['edge']]),
       'new_nonoverlap':a([r for r in rows if r['nonoverlap'] and not r['previous_pilot']]),
       'all_methods_projectable':a([r for r in rows if all(r['metrics'][m]['invalid_steps']==0 for m in METHODS)]),
       'rotation_strata':{label:a([r for r in rows if lo<=r['max_future_rotation_deg']<hi]) for label,lo,hi in [('lt5',0,5),('5to15',5,15),('ge15',15,1e9)]},
       'angular_overall':{m:{key:float(np.mean([r['angular_metrics'][m][key] for r in rows])) for key in ['mean_deg','final_deg']} for m in METHODS},
       'excluded_windows':excluded,'source_manifest':manifest,'protocol_sha256':sha(OUT/'protocol.json'),
       'accepted_cases':len(specs),'cases_with_valid_windows':len({r['key'] for r in rows}),'new_cases':sum(not s['previous_pilot'] for s in specs),
       'horizon_seconds_min_max':[min(r['actual_horizon_s'] for r in rows),max(r['actual_horizon_s'] for r in rows)]}
    for name,groups in [('equal_case',summary['per_case']),('equal_recording',summary['per_recording'])]:
        summary[name]={m:{metric:float(np.mean([v['methods'][m][metric] for v in groups.values() if v['windows']])) for metric in ['ADE_1.5s','FDE_1.5s']} for m in METHODS}
    summary['new_equal_case']={m:{metric:float(np.mean([summary['per_case'][s['key']]['methods'][m][metric] for s in specs if not s['previous_pilot'] and summary['per_case'][s['key']]['windows']])) for metric in ['ADE_1.5s','FDE_1.5s']} for m in METHODS}
    summary['case_wins_vs_raw']={m:sum(v['methods'][m]['ADE_1.5s']<v['methods']['raw_cv']['ADE_1.5s'] for v in summary['per_case'].values() if v['windows']) for m in ['comp_cv_predcam','comp_cv_refcam']}
    summary['recording_wins_vs_raw']={m:sum(v['methods'][m]['ADE_1.5s']<v['methods']['raw_cv']['ADE_1.5s'] for v in summary['per_recording'].values() if v['windows']) for m in ['comp_cv_predcam','comp_cv_refcam']}
    write(OUT/'predictions.json',rows);write(OUT/'examples.json',examples);write(OUT/'summary.json',summary)
    print('OVERALL',summary['overall'],flush=True);print('NEW ONLY',summary['new_only'],flush=True)
    print('WINS',summary['case_wins_vs_raw'],'POSE EXCLUSIONS',excluded,flush=True)

if __name__=='__main__':main()
