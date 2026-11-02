"""Bounded balanced design retaining every eligible observed window."""
from design import *
from scipy.optimize import minimize

def main():
 rows=load();x=np.array([r['levels'] for r in rows],float);n=len(x)
 moment=moment_matrix(x);p=moment[8:];recs=sorted({r['recording'] for r in rows})
 rc=np.array([[r['recording']==rec for r in rows] for rec in recs],float)
 ae=np.r_[np.ones((1,n)),x.T];be=np.r_[1,np.zeros(8)]
 lower=.25/n;upper=4/n
 # Exact first moments; minimize worst absolute pair moment with bounded weights.
 obj=np.r_[np.zeros(n),1.]
 au=np.r_[np.c_[p,-np.ones(len(p))],np.c_[-p,-np.ones(len(p))],np.c_[rc,np.zeros(len(rc))],np.c_[-rc,np.zeros(len(rc))]]
 bu=np.r_[np.zeros(2*len(p)),np.full(len(rc),.25),np.full(len(rc),-.02)]
 lp=linprog(obj,A_eq=np.c_[ae,np.zeros(len(ae))],b_eq=be,A_ub=au,b_ub=bu,bounds=[(lower,upper)]*n+[(0,1)],method='highs')
 assert lp.success,lp.message
 minimum=float(lp.x[-1]);limit=float(np.ceil((minimum-1e-10)/.025)*.025)
 # Round upward on a fixed .025 grid, then choose the most even weights.
 res=minimize(lambda w:n*np.dot(w,w),lp.x[:-1],jac=lambda w:2*n*w,bounds=Bounds(lower,upper),constraints=[LinearConstraint(ae,be,be),LinearConstraint(p,-limit,limit),LinearConstraint(rc,.02,.25)],method='SLSQP',options={'maxiter':3000,'ftol':1e-12})
 assert res.success,res.message
 w=res.x;assert abs(ae@w-be).max()<1e-7
 assert abs(p@w).max()<=limit+1e-7
 assert w.min()>=lower-1e-7 and w.max()<=upper+1e-7
 weighted_gram=x.T@(w[:,None]*x)
 # Main-effect estimability after including recording nuisance intercepts.
 z=rc.T;resid=x-z@np.linalg.solve(z.T@(w[:,None]*z),z.T@(w[:,None]*x))
 adjusted=resid.T@(w[:,None]*resid);adjcorr=adjusted/np.sqrt(np.outer(np.diag(adjusted),np.diag(adjusted)))
 triple=np.array([np.prod(x[:,cols],axis=1) for cols in itertools.combinations(range(8),3)])
 effective=1/np.dot(w,w)
 win=[dict(key=r['key'],clip=r['clip'],id=r['id'],recording=r['recording'],start_s=r['start_s'],end_s=r['end_s'],levels=r['levels'],weight=float(w[i]),relative_to_equal=float(n*w[i]),features=r['features']) for i,r in enumerate(rows)]
 write(OUT/'analysis_weights.json',win)
 paircells={F[i]+F[j]:{f'{a:+d},{b:+d}':float(w[(x[:,i]==a)&(x[:,j]==b)].sum()) for a,b in itertools.product([-1,1],repeat=2)} for i,j in itertools.combinations(range(8),2)}
 out=dict(n=n,recordings=len(recs),bank='matched97_new',all_rows_positive=True,ess_weights_only=float(effective),window_weight_bounds=[lower,upper],relative_weight_bounds=[.25,4],minimax_abs_pair_correlation=minimum,selected_pair_correlation_cap=limit,observed_max_abs_pair_correlation=float(abs(weighted_gram-np.eye(8)).max()),actual_relative_weight_range=[float(w.min()*n),float(w.max()*n)],recording_weights=dict(zip(recs,(rc@w).tolist())),weighted_main_gram=weighted_gram.tolist(),recording_adjusted_gram=adjusted.tolist(),recording_adjusted_correlations=adjcorr.tolist(),recording_adjusted_rank=int(np.linalg.matrix_rank(adjusted)),recording_adjusted_condition=float(np.linalg.cond(adjusted)),recording_adjusted_vif=np.diag(np.linalg.inv(adjcorr)).tolist(),max_main_interaction_moment=float(abs(triple@w).max()),pair_cells=paircells,factor_levels={F[j]:dict(low_count=int((x[:,j]<0).sum()),high_count=int((x[:,j]>0).sum()),low_weight=float(w[x[:,j]<0].sum()),high_weight=float(w[x[:,j]>0].sum())) for j in range(8)},protected_files={str(p.relative_to(ROOT)):digest(p) for p in [SOURCE/'combined_windows.json',SOURCE/'thresholds_locked.json',SOURCE/'checkpoint_availability.json']},method='Constrained minimax pair correlation then minimum sum squared weights; no outcome-dependent selection.',solver_message=res.message)
 write(OUT/'design_audit.json',out)
 print({k:v for k,v in out.items() if k not in ['weighted_main_gram','recording_adjusted_gram','recording_adjusted_correlations','pair_cells','protected_files','factor_levels']},flush=True)

if __name__=='__main__':main()
