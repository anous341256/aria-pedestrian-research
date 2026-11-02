"""Scientific plots of the locked enlarged-cohort experiment."""
from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 s=read(OUT/'summary.json');p=OUT/'presentation';p.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
 colors=['#2563a6','#748399','#8a64ad','#1b9b8a','#dfa642','#c96754'];x=np.arange(6)
 fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
 for ax,cohort,title in zip(axes,[s['reviewed'],s['cohorts']['old117'],s['cohorts']['new82']],['All 199 held-out windows','Original 117 test windows','New 82 test windows']):
  means=[cohort[v]['window_mean']['ADE15'] for v in VARIANTS];seeds=np.array([[cohort[v]['per_seed'][str(seed)]['ADE15'] for seed in [17,29,43]] for v in VARIANTS])
  ax.bar(x,means,color=colors,width=.68,alpha=.88)
  for j in range(3):ax.scatter(x+(j-1)*.09,seeds[:,j],color='#162c3d',s=12,zorder=3)
  hold=cohort['hold_last']['window_mean']['ADE15'];ax.axhline(hold,color='#333333',ls='--',lw=1.3,label=f'Hold-last {hold:.1f}')
  ax.set(xticks=x,xticklabels=list('ABCDEF'),ylabel='ADE at 1.5 s (pixels; lower is better)',title=title);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8)
  for i,y in enumerate(means):ax.text(i,y+1,f'{y:.1f}',ha='center',va='bottom',fontsize=8)
 fig.suptitle('Same six architectures and training budget | recording-disjoint evaluation',fontsize=13)
 fig.savefig(p/'cohort_results.png');fig.savefig(p/'cohort_results.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
 before=[s['old_test_before_after'][v]['before']['window_mean']['ADE15'] for v in VARIANTS];after=[s['old_test_before_after'][v]['after']['window_mean']['ADE15'] for v in VARIANTS]
 ax.bar(x-.18,before,width=.36,label='Original training set',color='#a5b2c0');ax.bar(x+.18,after,width=.36,label='Expanded training / validation',color='#2563a6');ax.set(xticks=x,xticklabels=list('ABCDEF'),ylabel='ADE (pixels)',title='Identical original 117 test windows; original six camera exclusion pairs')
 ax.axhline(s['cohorts']['old117']['hold_last']['window_mean']['ADE15'],ls='--',c='#333',label='Hold-last');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.legend(frameon=False)
 fig.savefig(p/'same_old_test_before_after.png');fig.savefig(p/'same_old_test_before_after.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
 for v,c in zip(VARIANTS,colors):ax.plot(np.arange(1,16)/10,s['horizon'][v],color=c,lw=1.8,label=v[0])
 ax.plot(np.arange(1,16)/10,s['horizon']['hold_last'],c='#333',ls='--',label='Hold-last');ax.set(xlabel='Forecast horizon (s)',ylabel='Center error (pixels)',title='199 windows: error by forecast horizon');ax.legend(ncol=7,frameon=False);ax.grid(alpha=.15);fig.savefig(p/'horizon.png');plt.close(fig)
 print('Figures saved.',flush=True)
if __name__=='__main__':main()
