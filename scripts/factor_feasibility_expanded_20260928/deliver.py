"""Create all-case figures, 14 fixed example cards, and a concise report."""
import sys,inspect,hashlib
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from prepare import ROOT,SRC,OUT,read,write,sha
sys.path.insert(0,str(ROOT/'scripts/factor_feasibility_20260928'))
import analyze as core
FONT='/mnt/c/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':10})
NAMES={'151153':'站台／扶梯／站厅','162538':'食品卖场','164557':'街道／商场','193709':'傍晚过街','201129':'夜间站外'}
COL={'stationary':'#7c8799','raw_cv':'#df803b','comp_cv_predcam':'#ae518c','comp_cv_refcam':'#188e85'}
LABEL={'stationary':'保持图像位置','raw_cv':'原始匀速','comp_cv_predcam':'补偿＋外推相机','comp_cv_refcam':'补偿＋参考未来相机（离线）'}

def verify(rows,s,specs):
    pilot=read(ROOT/'outputs/factor_feasibility_20260928/predictions.json')
    lookup={(r['recording'],r['track'],r['frame']):r for r in pilot};diff=0.;metricdiff=0.
    for row in rows:
        y=np.array(row['target_xy'])
        for m in core.METHODS:
            p=np.array(row['predictions'][m],float);e=np.linalg.norm(p-y,axis=1);e[~np.isfinite(e)]=core.PENALTY
            for n in (5,10,15):
                metricdiff=max(metricdiff,abs(e[:n].mean()-row['metrics'][m][f'ADE_{n/10:.1f}s']),abs(e[n-1]-row['metrics'][m][f'FDE_{n/10:.1f}s']))
            if row['previous_pilot']:
                old=np.array(lookup[(row['recording'],row['track'],row['frame'])]['predictions'][m],float)
                assert np.array_equal(np.isnan(old),np.isnan(p));good=np.isfinite(old)&np.isfinite(p)
                diff=max(diff,float(np.max(abs(old[good]-p[good]))))
    assert diff<1e-9 and metricdiff<1e-9
    for source in s['source_manifest']:
        sid=source['recording'];assert source['annotations_sha256']==sha(SRC/sid/'sam2_raw_v1/tracks.json')
        st=(ROOT/'data/Aria/raw_vrs'/f'unknown_20260803_{sid}.vrs').stat()
        assert st.st_size==source['source_vrs_size'] and st.st_mtime_ns==source['source_vrs_mtime_ns']
    p=read(OUT/'protocol.json')
    assert p['original_pilot_summary_sha256']==sha(ROOT/'outputs/factor_feasibility_20260928/summary.json')
    assert p['frozen_predictor_sha256']==hashlib.sha256(inspect.getsource(core.predict_history).encode()).hexdigest()
    assert len({r['key'] for r in rows})>=10 and len(specs)==14
    write(OUT/'package_verification.json',{'pilot_prediction_max_difference_px':diff,'independent_metric_max_difference_px':metricdiff,
       'original_pilot_summary_unchanged':True,'predictor_source_unchanged':True,'source_annotations_unchanged':True,'source_vrs_size_mtime_unchanged':True,
       'accepted_cases':len(specs),'evaluated_cases':len({r['key'] for r in rows}),'recordings':len({r['recording'] for r in rows}),
       'windows':len(rows),'invalid_projection_windows':sum(any(r['metrics'][m]['invalid_steps'] for m in core.METHODS) for r in rows),
       'fixed_example_cards':len(read(OUT/'examples.json')),'protocol_hash_matches':sha(OUT/'protocol.json')==s['protocol_sha256']})

def overview(s,specs):
    fig,(a,b)=plt.subplots(1,2,figsize=(15.5,8.5),gridspec_kw={'width_ratios':[1.7,1]})
    yy=np.arange(len(specs));width=.22
    for j,m in enumerate(['raw_cv','comp_cv_refcam','stationary']):
        values=[s['per_case'][x['key']]['methods'][m]['ADE_1.5s'] for x in specs]
        bars=a.barh(yy+(j-1)*width,values,width,color=COL[m],label=LABEL[m]);a.bar_label(bars,fmt='%.0f',padding=2,fontsize=8)
    a.set_yticks(yy,[f"{x['case_number']:02d}  {x['key']}  {'原例' if x['previous_pilot'] else '新增'}" for x in specs]);a.invert_yaxis()
    a.set_xlim(0,1050);a.set_xlabel('1.5 秒中心 ADE / 原图像素 ↓');a.set_title('全部 14 个案例：保留完整结果，按录制与 ID 排序');a.grid(axis='x',alpha=.2);a.set_axisbelow(True);a.legend(fontsize=9,loc='lower right')
    groups=[('原先 4 例',s['pilot_only']),('新增 10 例',s['new_only']),('全部 14 例',s['overall'])]
    xx=np.arange(3)
    for j,m in enumerate(['raw_cv','comp_cv_refcam','stationary']):
        bars=b.bar(xx+(j-1)*.25,[g['methods'][m]['ADE_1.5s'] for _,g in groups],.25,color=COL[m]);b.bar_label(bars,fmt='%.0f',padding=3,fontsize=10)
    b.set_xticks(xx,[name for name,_ in groups]);b.set_ylim(0,800);b.set_ylabel('1.5 秒中心 ADE / 像素');b.set_title('单独验证新增样例');b.grid(axis='y',alpha=.2);b.set_axisbelow(True)
    fig.suptitle('扩大验证：同一匀速预测器，5 段录制，14 条轨迹，381 个窗口',fontsize=18)
    fig.text(.5,.055,'绿色使用 Aria 的参考未来转动，只支持条件可行性；未来相机转动按历史外推时，14 个案例的平均误差均未改善。',ha='center',fontsize=11)
    fig.text(.5,.025,'4 个候选因可见身份／mask 错误在评分前排除；4 个窗口缺少有效参考位姿。案例不等于独立人物或独立场景。',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(OUT/'comparison_14_cases.png',dpi=165);fig.savefig(OUT/'comparison_14_cases.pdf');plt.close(fig)

def example_cards(specs):
    index={s['key']:s for s in specs};directory=OUT/'examples';directory.mkdir(exist_ok=True);paths=[]
    for row in read(OUT/'examples.json'):
        spec=index[row['key']];cap=cv2.VideoCapture(str(SRC/row['recording']/'rgb.mp4'));ims=[]
        for f in [row['frame'],row['frame']+15]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;ims.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        cap.release()
        fig,axs=plt.subplots(2,2,figsize=(13,9.8),gridspec_kw={'height_ratios':[1.25,1]})
        for j,im in enumerate(ims):
            ax=axs[0,j];ax.imshow(im);ax.set_xlim(0,2560);ax.set_ylim(1920,0);ax.set_xticks([]);ax.set_yticks([])
        box=np.array(row['history_xyxy'])[-1];axs[0,0].add_patch(plt.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,color='#42a5ef',lw=2))
        axs[0,0].set_title(f"当前画面 t={row['frame']/10:.1f}s；蓝框为历史末帧")
        target=np.array(row['target_xy']);axs[0,1].scatter(*target[-1],s=90,color='#61d36b',marker='x',linewidths=2,label='SAM 后续参考')
        for m,marker in [('raw_cv','o'),('comp_cv_refcam','D')]:
            pt=np.array(row['predictions'][m],float)[-1]
            if np.isfinite(pt).all() and 0<=pt[0]<2560 and 0<=pt[1]<1920:axs[0,1].scatter(*pt,s=65,color=COL[m],marker=marker,label=LABEL[m])
            elif np.isfinite(pt).all():
                pp=np.clip(pt,[60,60],[2500,1860]);axs[0,1].scatter(*pp,s=75,color=COL[m],marker='>',label=LABEL[m]+'（画外）')
        axs[0,1].legend(fontsize=8,loc='upper right');axs[0,1].set_title('1.5 秒后画面：仅用于事后核对，不输入预测器')
        ref=np.load(OUT/f"{row['recording']}_reference.npz");K=ref['K'];ki=np.linalg.inv(K);rs=ref['R'];cur=rs[row['frame']]
        norm=np.array([core.project([p],K@cur.T@rs[f]@ki)[0][0] for p,f in zip(target,row['future_frames'])]);values=[]
        for j,(hist,fut,pred,title) in enumerate([
            (np.array(row['history_xy']),target,np.array(row['predictions']['raw_cv']),'原始图像横坐标序列'),
            (np.array(row['history_normalized_xy']),norm,np.array(row['pedestrian_forecast_normalized_xy']),'旋转补偿后的横坐标序列')]):
            ax=axs[1,j];origin=hist[-1,0]
            ax.plot(np.arange(-5,1)/10,hist[:,0]-origin,'o-',color='#4285bc',label='历史')
            ax.plot(np.arange(1,16)/10,fut[:,0]-origin,'o-',ms=3,color='#218d85',label='后续参考')
            ax.plot(np.r_[0,np.arange(1,16)/10],np.r_[origin,pred[:,0]]-origin,'--',color=COL['raw_cv'],label='同一匀速预测')
            values.extend(np.r_[hist[:,0],fut[:,0],pred[:,0]]-origin);ax.axvline(0,color='gray',lw=1);ax.grid(alpha=.2)
            ax.set_title(title);ax.set_xlabel('相对预测起点 / 秒');ax.set_ylabel('相对当前横坐标 / 像素');ax.legend(fontsize=8)
        lo,hi=min(values),max(values);pad=max(10,(hi-lo)*.08)
        for ax in axs[1]:ax.set_ylim(lo-pad,hi+pad)
        met=row['metrics'];fig.suptitle(f"案例 {spec['case_number']:02d} / 14 · {row['key']} · {NAMES[row['recording']]} · {'原例' if row['previous_pilot'] else '新增'}",fontsize=16)
        fig.text(.5,.065,f"本窗口二维 ADE：原始匀速 {met['raw_cv']['ADE_1.5s']:.1f}px；参考未来相机 {met['comp_cv_refcam']['ADE_1.5s']:.1f}px；外推相机 {met['comp_cv_predcam']['ADE_1.5s']:.1f}px",ha='center',fontsize=11)
        fig.text(.5,.035,'左右曲线同尺度。未来参考相机仅用于离线诊断；样例起点按固定规则选取，不按预测效果挑选。',ha='center',fontsize=10)
        fig.tight_layout(rect=[0,.09,1,.95]);path=directory/f"{spec['case_number']:02d}_{row['key']}.png";fig.savefig(path,dpi=135);plt.close(fig);paths.append(path)
        print('CARD',path.name,flush=True)
    # Contact sheet includes every case for compact visual QA.
    thumbs=[]
    for p in paths:
        im=cv2.imread(str(p));thumbs.append(cv2.resize(im,(650,490)))
    for page in range((len(thumbs)+5)//6):
        ts=thumbs[page*6:(page+1)*6]
        if len(ts)%2:ts.append(np.full_like(thumbs[0],255))
        cv2.imwrite(str(OUT/f'example_contact_{page+1}.jpg'),np.vstack([np.hstack(ts[i:i+2]) for i in range(0,len(ts),2)]))
    write(OUT/'example_manifest.json',[{'path':str(p.relative_to(OUT)),'sha256':sha(p)} for p in paths])

def main():
    s=read(OUT/'summary.json');rows=read(OUT/'predictions.json');specs=read(OUT/'candidates.json')
    verify(rows,s,specs);overview(s,specs);example_cards(specs)
    print('DELIVERY VERIFIED',read(OUT/'package_verification.json'),flush=True)
if __name__=='__main__':main()
