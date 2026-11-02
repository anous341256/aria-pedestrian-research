from design import *
from scipy.optimize import minimize

def main():
 rows=load();x=np.array([r['levels'] for r in rows]);n=len(x);m=moment_matrix(x);ae=np.r_[np.ones((1,n)),m];be=np.r_[1,np.zeros(len(m))]
 triple=np.array([np.prod(x[:,cols],axis=1) for cols in itertools.combinations(range(8),3)])
 strength3=linprog(np.zeros(n),A_eq=np.r_[ae,triple],b_eq=np.r_[be,np.zeros(len(triple))],bounds=(0,None),method='highs')
 print('Strength3:',strength3.message,flush=True)
 recs=sorted({r['recording'] for r in rows});rc=np.array([[r['recording']==rec for r in rows] for rec in recs],float)
 cons=[LinearConstraint(ae,be,be),LinearConstraint(rc,.02,.25)]
 b=Bounds(1e-6,.03)
 lp=linprog(np.zeros(n),A_eq=ae,b_eq=be,A_ub=np.r_[rc,-rc],b_ub=np.r_[np.full(len(rc),.25),np.full(len(rc),-.02)],bounds=list(zip(np.full(n,b.lb),np.full(n,b.ub))),method='highs')
 print('Bounded strength2:',lp.message,flush=True)
 if not lp.success:return
 res=minimize(lambda w:n*np.dot(w,w),lp.x,jac=lambda w:2*n*w,bounds=b,constraints=cons,method='SLSQP',options={'maxiter':3000,'ftol':1e-12})
 w=res.x
 out=dict(success=bool(res.success),message=res.message,n=n,ess=float(1/np.dot(w,w)),min_weight=float(w.min()),max_weight=float(w.max()),equal_weight=1/n,max_moment_error=float(abs(ae@w-be).max()),strength3_status=int(strength3.status),max_main_interaction_moment=float(abs(triple@w).max()),recording_weights=dict(zip(recs,(rc@w).tolist())),keys=[r['key'] for r in rows],weights=w.tolist())
 write(OUT/'weight_probe.json',out)
 print({k:v for k,v in out.items() if k not in ['keys','weights']},flush=True)
if __name__=='__main__':main()
