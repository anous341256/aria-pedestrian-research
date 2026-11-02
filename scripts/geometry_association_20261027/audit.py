from assoc_common import *
from tracker import *
from scipy.spatial.transform import Rotation
import sys,copy
def preflight():
 rng=np.random.default_rng(20261027);K=np.array([[1200.,0,1280],[0,1200,960],[0,0,1.]])
 maxerr=0.;jacerr=0.;staticerr=0.
 for _ in range(100):
  R=Rotation.from_rotvec(rng.normal(0,.1,3)).as_matrix();c=rng.normal(0,.2,3);x=rng.normal(0,.2,3)+[0,0,3];uv,J=image_project(x,K);w=lift_world(uv,x[2],R,c,K);back=R.T@(w-c);maxerr=max(maxerr,float(abs(back-x).max()))
  num=np.column_stack([(image_project(x+np.eye(3)[j]*1e-5,K)[0]-image_project(x-np.eye(3)[j]*1e-5,K)[0])/2e-5 for j in range(3)]);jacerr=max(jacerr,float(abs(J-num).max()))
  # Independent homogeneous 4x4 transform, including optical center translation.
  T=np.eye(4);T[:3,:3]=R;T[:3,3]=c;ind=np.linalg.inv(T)@np.r_[w,1.];uv2=(K@ind[:3]);uv2=uv2[:2]/uv2[2];staticerr=max(staticerr,float(abs(uv2-uv).max()))
 assert maxerr<1e-10 and jacerr<1e-5 and staticerr<1e-8
 uv,_=image_project(np.array([-.1,0,2.]),K);assert abs(uv[0]-1220)<1e-9
 tr=Tracker('rotation_translation_depth',K,np.eye(3));wp=np.array([0.,0.,3.]);errors=[]
 for f in range(10):
  R=Rotation.from_rotvec([0,.01*f,0]).as_matrix();c=np.array([.01*f,0,0]);x=R.T@(wp-c);p,_=image_project(x,K);o=dict(uid=str(f),uv=p,box=np.r_[p-[100,200],p+[100,200]],embedding=np.array([1.,0.,0.]),z=float(x[2]),iqr=.02,depth_valid=True)
  rows,_=tr.step([o],R,c,f*.1,f);assert rows[0]['track_id']==1
  if f:errors.append(np.linalg.norm(np.array(rows[0]['predicted_uv'])-p));assert rows[0]['prior_kind']=='full3D'
 assert max(errors)<1e-7
 write(OUT/'geometry_preflight.json',dict(passed=True,random_lift_transform_roundtrip_m=maxerr,projection_jacobian_finite_difference_max=jacerr,independent_homogeneous_projection_max_px=staticerr,right_translation_pixel_shift_px=float(uv[0]-1280),static_world_tracker_prediction_max_px=float(max(errors))))
 print(read(OUT/'geometry_preflight.json'))
def replay():
 from run import load_clip,infer
 inv=read(OUT/'inventory.json');runs=prefix=permuted=controls=0;states=0
 for cl in inv['clips']:
  cid=cl['clip'];p,frames=load_clip(cid)
  for cond in ['natural','blackout3']:
   for arm in ARMS:
    rows,logs=infer(p,frames,arm,cond);stored=read(OUT/'tracking'/cond/arm/f'{cid}.json');assert rows==stored['rows'] and logs==stored['frames'];runs+=1
    end=min(34,len(frames)-1);pp={k:v[:end+1].copy() if k in ['R','c','times_ns','selected_mps_ns'] else v.copy() for k,v in p.items()};r,l=infer(pp,frames[:end+1],arm,cond);assert r==[a for a in rows if a['frame']<=end];assert l==logs[:end+1];prefix+=1
    altered=[[dict(o,source_id='FAKE',source_segment='WRONG') for o in reversed(obs)] for obs in frames];r,l=infer(p,altered,arm,cond);assert r==rows and l==logs;permuted+=1
    if arm!='rotation_translation_depth':
     pp={k:v.copy() for k,v in p.items()};pp['c']+=np.arange(len(frames))[:,None]*123
     ff=[[dict(o,z=999.,iqr=100.,depth_valid=not o['depth_valid']) for o in obs] for obs in frames];r,l=infer(pp,ff,arm,cond);assert r==rows and l==logs;controls+=1
    for a in rows:
     if 'prior_cov' in a:assert np.linalg.eigvalsh(np.array(a['prior_cov'])).min()>-1e-6
    states+=len(rows)
  print('AUDIT',cid,flush=True)
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items())
 code=read(OUT/'execution_code_hashes.json');assert all(sha(Path(__file__).parent/k)==v for k,v in code.items())
 write(OUT/'verification.json',dict(passed=True,exact_replays=runs,future_prefix_replays=prefix,candidate_order_and_fake_identity_invariance=permuted,controls_ignore_translation_and_depth=controls,assignment_rows=states,protected_sources_unchanged=len(protected)))
 print(read(OUT/'verification.json'))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='replay':replay()
 else:preflight()
