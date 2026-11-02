"""Within-clip temporal-block uncertainty and motion strata; no population CI."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT,CONFIG,read_json,write_json

def interval(rows,a,b):
    blocks=[[r for r in rows if r['frame']//10==i] for i in range(10)];blocks=[b for b in blocks if b]
    rng=np.random.default_rng(20260927);draws=[]
    for _ in range(2000):
        sample=[r for j in rng.integers(0,len(blocks),len(blocks)) for r in blocks[j]]
        draws.append(np.median([r[a]-r[b] for r in sample]))
    return {'median_paired_delta_px':float(np.median([r[a]-r[b] for r in rows])),
            'temporal_block_95_interval':np.quantile(draws,[.025,.975]).tolist(),'block_seconds':1,
            'meaning':f'{a} minus {b}; negative favors {a}. Within-clip temporal sensitivity only, NOT generalization uncertainty.'}

def main():
    result=[];allrows=[]
    for s in read_json(CONFIG)['clips']:
        rows=read_json(OUT/'clips'/s['id']/'geometry/frame_metrics.json');allrows+=rows
        result.append({'clip':s['id'],'valid_pairs':len(rows),'possible_pairs':99,'comparisons':{
           'aria_rgb_vs_rgb_similarity':interval(rows,'aria_rgb','rgb_similarity'),
           'aria_rgb_vs_rgb_homography':interval(rows,'aria_rgb','rgb_homography'),
           'aria_rotation_vs_none':interval(rows,'aria_rotation','none')}})
    strata=[]
    for name,predicate in [('rotation_lt_1deg',lambda r:r['rotation_deg']<1),('rotation_1_to_5deg',lambda r:1<=r['rotation_deg']<5),
                           ('rotation_ge_5deg',lambda r:r['rotation_deg']>=5),('translation_lt_2cm',lambda r:r['translation_m']<.02),
                           ('translation_ge_2cm',lambda r:r['translation_m']>=.02)]:
        selected=[r for r in allrows if predicate(r)]
        strata.append({'group':name,'pairs':len(selected),'median_px':{m:float(np.median([r[m] for r in selected])) if selected else None for m in ['none','rgb_similarity','rgb_homography','aria_rotation','aria_rgb']}})
    write_json(OUT/'results/geometry_statistics.json',{'clips':result,'strata':strata,'limitations':'4 source recordings, 6 ten-second clips; no cross-subject/domain claim; translation is device translation, not pedestrian depth or motion; dense sampled time blocks are dependent.'})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,title in [(axes[0],'aria_rgb_vs_rgb_similarity','Aria+RGB minus RGB similarity'),(axes[1],'aria_rgb_vs_rgb_homography','Aria+RGB minus RGB homography')]:
        vals=np.array([r['comparisons'][key]['median_paired_delta_px'] for r in result]);ci=np.array([r['comparisons'][key]['temporal_block_95_interval'] for r in result])
        ax.errorbar(vals,np.arange(len(result)),xerr=np.maximum(0,np.array([vals-ci[:,0],ci[:,1]-vals])),fmt='o');ax.axvline(0,c='gray');ax.set_yticks(np.arange(len(result)),[r['clip'] for r in result]);ax.set_xlabel('Paired residual difference (pixels)');ax.set_title(title);ax.grid(alpha=.2)
    fig.suptitle('1-second temporal-block intervals (within clip only)');fig.savefig(OUT/'results/geometry_intervals.png',dpi=170);plt.close(fig)
    print(result)

if __name__=='__main__':main()
