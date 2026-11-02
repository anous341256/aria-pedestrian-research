"""Small report figures plus independent metric/source verification."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.transform import Rotation
from analyze import ROOT,SRC,OUT,METHODS,PENALTY,read,write,project,sha

FONT='/mnt/c/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':11})
NAMES={'street_S1':'白天街道','station_S4':'站台→扶梯','dusk_follow_S1':'傍晚跟随','dusk_cross_S4':'傍晚过街'}
LABEL={'stationary':'保持图像位置','raw_cv':'原始轨迹＋匀速',
       'comp_cv_predcam':'补偿＋外推相机转动','comp_cv_refcam':'补偿＋参考未来转动（离线）'}
COL={'stationary':'#7c8799','raw_cv':'#e18139','comp_cv_predcam':'#b2528f','comp_cv_refcam':'#218d85'}

def verify_and_angular(rows,s):
    maxdiff=0;angle={m:[] for m in METHODS};invalid_windows=0
    for row in rows:
        target=np.array(row['target_xy']);ref=np.load(OUT/f"{row['recording']}_reference.npz");K=ref['K'];ki=np.linalg.inv(K)
        rr=ref['R'][row['frame']];fr=ref['R'][row['future_frames']]
        target_ray=np.c_[target,np.ones(15)]@ki.T;target_ray/=np.linalg.norm(target_ray,axis=1,keepdims=True)
        for m in METHODS:
            p=np.array(row['predictions'][m],dtype=float)
            e=np.sqrt(np.sum((p-target)**2,axis=1));e[~np.isfinite(e)]=PENALTY
            for n in [5,10,15]:
                for metric,value in [('ADE',e[:n].sum()/n),('FDE',e[n-1])]:maxdiff=max(maxdiff,abs(value-row['metrics'][m][f'{metric}_{n/10:.1f}s']))
            if m.startswith('comp_'):
                ped=np.array(row['pedestrian_forecast_normalized_xy']);rays=np.c_[ped,np.ones(15)]@ki.T
                rotations=np.array([r.T@rr for r in fr]) if m.endswith('refcam') else Rotation.from_rotvec(-np.arange(1,16)[:,None]/10*np.array(row['camera_omega_rad_s'])).as_matrix()
                rays=np.einsum('nij,nj->ni',rotations,rays)
            else:rays=np.c_[p,np.ones(15)]@ki.T
            rays/=np.linalg.norm(rays,axis=1,keepdims=True)
            angles=np.degrees(np.arccos(np.clip(np.sum(rays*target_ray,axis=1),-1,1)))
            angle[m].append({'key':row['key'],'frame':row['frame'],'mean_deg':float(angles.mean()),'final_deg':float(angles[-1])})
        invalid_windows+=any(row['metrics'][m]['invalid_steps']>0 for m in METHODS)
    assert maxdiff<1e-9
    for source in s['source_manifest']:
        sid=source['recording'];assert source['annotations_sha256']==sha(SRC/sid/'sam2_raw_v1/tracks.json')
        vrs=ROOT/'data/Aria/raw_vrs'/f'unknown_20260803_{sid}.vrs';st=vrs.stat()
        assert st.st_size==source['source_vrs_size'] and st.st_mtime_ns==source['source_vrs_mtime_ns']
    angular={m:{'mean_angular_error_deg':float(np.mean([r['mean_deg'] for r in values])),
                'final_angular_error_deg':float(np.mean([r['final_deg'] for r in values]))} for m,values in angle.items()}
    write(OUT/'posthoc_angular_check.json',{'reason':'Added after seeing projective singularities, to test whether the failure conclusion depends on the predeclared 32000px invalid-projection penalty. Original pixel results unchanged.',
         'metric':'Angle between calibrated predicted and observed viewing rays, including behind-camera rays, averaged over all 205 windows / 15 targets. Calibration remains privileged.',
         'results':angular,'windows':len(rows)})
    write(OUT/'package_verification.json',{'independent_metric_max_error_px':maxdiff,'source_annotation_hashes_unchanged':True,
         'source_vrs_size_mtime_unchanged':True,'all_windows':len(rows),'windows_with_invalid_projection':int(invalid_windows),
         'protocol_hash_matches':sha(OUT/'protocol.json')==s['protocol_sha256'],'posthoc_angular_result':angular})

def figures(s,rows):
    fig,(a,b)=plt.subplots(1,2,figsize=(13.5,5.6),gridspec_kw={'width_ratios':[1.3,1]})
    entries=[*s['per_track'].items(),('overall',s['overall'])];xx=np.arange(len(entries));width=.25
    for j,m in enumerate(['raw_cv','comp_cv_refcam','stationary']):
        yy=[v['methods'][m]['ADE_1.5s'] for _,v in entries]
        bars=a.bar(xx+(j-1)*width,yy,width,color=COL[m],label=LABEL[m])
        a.bar_label(bars,fmt='%.0f',padding=3,fontsize=9)
    a.set_xticks(xx,[NAMES.get(k,'全部') for k,_ in entries],rotation=15);a.set_ylabel('1.5 秒中心 ADE / 原图像素 ↓');a.set_ylim(0,830)
    a.set_title('相机未来使用参考值时，简单预测能改善多少？',fontsize=12);a.legend(fontsize=8.5);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    for m in METHODS:
        b.plot([.5,1,1.5],[s['overall']['methods'][m][f'ADE_{h:.1f}s'] for h in [.5,1,1.5]],'o-',color=COL[m],label=LABEL[m])
    b.set_yscale('log');b.set_xticks([.5,1,1.5]);b.set_xlabel('预测时长 / 秒');b.set_ylabel('ADE / 像素（对数轴）');b.set_title('相机未来仍需预测：简单角速度外推失败',fontsize=12);b.grid(alpha=.2);b.legend(fontsize=8.5,loc='upper left')
    fig.suptitle('固定同一个匀速预测器：旋转补偿可行性诊断',fontsize=17,y=.99)
    fig.text(.5,.045,'4 条 SAM 轨迹 / 3 段录制 / 205 个重叠窗口；历史 0.5 秒。绿色使用未来 MPS 参考，仅作离线诊断。',ha='center',fontsize=10)
    fig.text(.5,.012,'紫色含 94 个投影无效点，按预定 32,000 像素计错；可投影子集与无罚值的角度误差也未显示改善。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.10,1,.94]);fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)

    examples=read(OUT/'examples.json');fig,axs=plt.subplots(4,2,figsize=(13.5,12),sharex=True)
    for i,row in enumerate(examples):
        ref=np.load(OUT/f"{row['recording']}_reference.npz");K=ref['K'];ki=np.linalg.inv(K);R=ref['R'];cur=R[row['frame']]
        gt=np.array(row['target_xy']);norm=np.array([project([p],K@cur.T@R[f]@ki)[0][0] for p,f in zip(gt,row['future_frames'])])
        values=[]
        for j,(hist,target,pred,title) in enumerate([
            (np.array(row['history_xy']),gt,np.array(row['predictions']['raw_cv']),'原图横坐标位移'),
            (np.array(row['history_normalized_xy']),norm,np.array(row['pedestrian_forecast_normalized_xy']),'旋转补偿后的横坐标位移')]):
            origin=hist[-1,0];ax=axs[i,j];ax.plot(np.arange(-5,1)/10,hist[:,0]-origin,'o-',color='#4285bc',label='历史观测')
            ax.plot(np.arange(1,16)/10,target[:,0]-origin,'o-',color='#218d85',ms=3,label='后续参考')
            ax.plot(np.r_[0,np.arange(1,16)/10],np.r_[hist[-1,0],pred[:,0]]-origin,'--',color='#e18139',label='同一匀速预测器')
            values.extend(np.r_[hist[:,0],target[:,0],pred[:,0]]-origin)
            ax.axvline(0,color='#7c8799',lw=1);ax.grid(alpha=.2);ax.set_ylabel('相对当前横坐标 / 像素')
            ax.set_title(f"{NAMES[row['key']]} · t={row['frame']/10:.1f}s · {title}",fontsize=11)
            if i==0:ax.legend(fontsize=9)
        lo,hi=min(values),max(values);pad=max(10,(hi-lo)*.08)
        for ax in axs[i]:ax.set_ylim(lo-pad,hi+pad)
    for ax in axs[-1]:ax.set_xlabel('相对预测起点 / 秒')
    fig.suptitle('固定样例：观察坐标序列在补偿前后的变化',fontsize=17)
    fig.text(.5,.01,'每行左右纵轴相同；统一取开始后约 5 秒。右列后续参考经未来 MPS 转到当前朝向，只用于评价。',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.96]);fig.savefig(OUT/'trajectory_examples.png',dpi=165);plt.close(fig)

def main():
    s=read(OUT/'summary.json');rows=read(OUT/'predictions.json');verify_and_angular(rows,s);figures(s,rows)
    print(read(OUT/'package_verification.json'),flush=True)

if __name__=='__main__':main()
