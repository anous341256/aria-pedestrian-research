"""Independent artifact-based metric audit, paired contrasts, coverage and report."""
import sys,json,csv,collections,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/depth_forecast_20261008';TRAIN=OUT/'forecast'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def main():
    plan=read(OUT/'extraction_plan.json');meta=read(ROOT/'outputs/transformer_uncovered200_20261006/data/samples.json');data=dict(np.load(ROOT/'outputs/transformer_uncovered200_20261006/data/dataset.npz'));raw=np.load(OUT/'depth_all21.npz')['depth'];folds=read(TRAIN/'folds.json');comp=read(TRAIN/'comparison.json')
    rows=list(csv.DictReader((TRAIN/'metrics.csv').open()));metric={(r['arm'],int(r['seed']),int(r['index'])):r for r in rows};maxerr=0.;maxgeom=0.;count=0
    for f in folds:
        for s in [17,29,43]:
            for arm in ['no_depth','with_depth']:
                p=np.load(TRAIN/f'fold_{f["fold"]}/seed_{s}/{arm}/predictions.npz')
                idx=p['indices'];err=np.linalg.norm(p['image']-data['xy'][idx,6:],axis=-1);err[(p['depth']<=1e-6)|~np.isfinite(err)]=32000.
                for j,i in enumerate(idx):
                    row=metric[arm,s,int(i)];maxerr=max(maxerr,abs(err[j].mean()-float(row['image_ADE'])),abs(err[j,-1]-float(row['image_FDE'])))
                    rays=np.c_[p['component'][j],np.ones(15)]@np.linalg.inv(data['K'][i]).T;R=Rotation.from_rotvec(p['rotation'][j].astype(float)).as_matrix();q=np.einsum('tji,tj->ti',R,rays)@data['K'][i].T;uv=q[:,:2]/q[:,2:]
                    maxgeom=max(maxgeom,float(np.max(np.abs(uv-p['image'][j]))));count+=1
    assert maxerr<1e-6 and maxgeom<.01
    frames=[];measure=[];fail=[]
    for e in plan['entries']:
        folder=OUT/'frames'/e['key']
        if (folder/'complete.json').exists():
            r=read(folder/'complete.json');t=r['timing'];assert max(t['left_timestamp_ns'],t['right_timestamp_ns'])<=e['rgb_timestamp_ns'];assert t['stereo_skew_ms']<=1 and t['rgb_skew_ms']<=40
            frames.append(r);measure += [dict(frame_key=e['key'],historical=e['historical'],**x) for x in read(folder/'measurements.json')]
        else:assert (folder/'failure.json').exists();fail.append(e['key'])
    assert len(frames)+len(fail)==2163
    rapid=[]
    # Diagnostic only: never changes frozen depth features or inclusion rules.
    for i,m in enumerate(meta):
        for j in range(1,21):
            if raw[i,j-1,4] and raw[i,j,4] and abs(float(raw[i,j,0]-raw[i,j-1,0]))>1:
                rapid.append(dict(key=m['key'],frame_from=m['frames'][j-1],frame_to=m['frames'][j],delta_z_m=float(raw[i,j,0]-raw[i,j-1,0]),entirely_historical=j<6))
    write(OUT/'rapid_depth_change_diagnostics.json',dict(rule='Absolute adjacent Z change>1m, both measurements accepted; diagnostic only, never filters model data. Overlapping windows can duplicate events.',events=rapid))
    covered={}
    for version in sorted({m['source_version'] for m in meta}):
        ii=[i for i,m in enumerate(meta) if m['source_version']==version];v=raw[ii,:6,4]
        covered[version]=dict(windows=len(ii),usable_history_frames=int(v.sum()),history_frames=v.size,windows_any=int((v.sum(1)>0).sum()),windows_all6=int((v.sum(1)==6).sum()))
    # Equal-weight recordings, first average seeds within each fixed test window.
    contrast={};recs=sorted({meta[i]['recording'] for i in range(117)});rng=np.random.default_rng(20261008)
    for name in ['image_ADE','image_FDE','residual_ADE','camera_deg_ADE']:
        delta=np.array([np.mean([float(metric['with_depth',s,i][name])-float(metric['no_depth',s,i][name]) for s in [17,29,43]]) for i in range(117)])
        recdelta=np.array([np.mean(delta[[i for i in range(117) if meta[i]['recording']==r]]) for r in recs]);boots=np.mean(rng.choice(recdelta,(2000,len(recs)),replace=True),axis=1)
        contrast[name]=dict(window_equal_delta=float(delta.mean()),improved_windows=int((delta<0).sum()),recording_equal_delta=float(recdelta.mean()),improved_recordings=int((recdelta<0).sum()),recording_deltas=dict(zip(recs,recdelta.tolist())),exploratory_recording_bootstrap95=np.percentile(boots,[2.5,97.5]).tolist())
    subgroup={}
    for title,ii in [('all6_depth',[i for i in range(117) if raw[i,:6,4].sum()==6]),('partial_depth',[i for i in range(117) if 0<raw[i,:6,4].sum()<6]),('no_depth',[i for i in range(117) if raw[i,:6,4].sum()==0])]:
        subgroup[title]=dict(windows=len(ii),values={arm:{k:float(np.mean([float(metric[arm,s,i][k]) for s in [17,29,43] for i in ii])) for k in ['image_ADE','image_FDE']} for arm in ['no_depth','with_depth']} if ii else {})
    audit=dict(passed=True,metric_rows=count,independent_metric_max_abs=maxerr,independent_SciPy_projection_max_px=maxgeom,causal_stereo_all_completed=True,frames_completed=len(frames),failed_frames=fail,max_rgb_skew_ms=max(r['timing']['rgb_skew_ms'] for r in frames),max_stereo_skew_ms=max(r['timing']['stereo_skew_ms'] for r in frames),unique_person_frames=len(measure),usable_unique_person_frames=sum(r['usable'] for r in measure),rejection_reasons=dict(collections.Counter(reason for r in measure for reason in r['reasons'])))
    write(OUT/'audit.json',audit);write(TRAIN/'contrasts.json',contrast);write(OUT/'coverage_by_cohort.json',covered);write(TRAIN/'depth_subgroups.json',subgroup)
    with (OUT/'unique_person_depth.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(measure[0]));w.writeheader();w.writerows(measure)
    a,b=comp['arms']['no_depth'],comp['arms']['with_depth'];gain=(a['image_ADE']-b['image_ADE'])/a['image_ADE']*100
    lines=['# 全训练数据双目深度提取与预测消融 — 2026-10-08','',
        f"有深度相对无深度的 ADE 变化：{a['image_ADE']:.2f} → {b['image_ADE']:.2f} px（改善率 {gain:+.2f}%）。这是一组固定协议的结果，结合种子及录制层面的差异判断稳定性。",'',
        '## 相同条件下的主要对照','',
        '|输入|图像 ADE↓|图像 FDE↓|行人残差 ADE↓|相机角度均值↓|','|---|---:|---:|---:|---:|']
    for arm,label in [('no_depth','无深度'),('with_depth','加入历史深度')]:
        r=comp['arms'][arm];lines.append(f"|{label}|{r['image_ADE']:.2f}|{r['image_FDE']:.2f}|{r['residual_ADE']:.2f}|{r['camera_deg_ADE']:.2f}°|")
    lines += [f"|保持最后位置|{comp['hold']['ADE']:.2f}|{comp['hold']['FDE']:.2f}|—|—|",'',
        '## 数据与设计','',
        f"当前扩充数据 374 个准备窗口，原实际训练并集 373 个。为 61 个片段的 2163 个去重 RGB 时刻提取双目深度：成功 {len(frames)}，失败 {len(fail)}；缺失保留，不替换样本。每个窗口存全 21 帧深度，但模型只读取过去 6 帧。",'',
        '|来源|窗口|历史深度有效帧/总帧|至少1帧有效|6帧都有效|','|---|---:|---:|---:|---:|']
    for k,r in covered.items():lines.append(f"|{k}|{r['windows']}|{r['usable_history_frames']}/{r['history_frames']}|{r['windows_any']}|{r['windows_all6']}|")
    lines += ['',
        '两个新组都从上一版已选定的 cross-attention 权重开始，冻结原始 GRU 编码器、相机头和行人 flow 头，继续训练注意力适配层。新增相同的无偏置 5→64 深度映射层，初始为零；无深度组把整组深度值/有效性/质量通道置零。两组都保留两个运动输出头和原来的旋转重投影。',
        '深度作为历史行人 token 的额外信息，通过相机与行人的交互影响两头。不是训练新的深度网络，也不是直接监督三维行人轨迹；此轮没有同时加入相机平移补偿。',
        '输入通道：log(Z)/log30、相邻 logZ 变化、有效位、左右一致性比例、IQR/Z。至少30个内部深度点、左右一致性比例≥0.7、Z在0.5–30m且IQR/Z≤0.25才使用；这是预先固定的质量先验，非真实精度保证。SAM mask 腐蚀7px；全组都保留缺失/拒绝窗口，不以深度可用性筛选测试集。',
        '沿用原117测试窗口、原验证窗口和6组录制隔离划分；每折从原373训练并集中排除测试及验证的整段录制。新数据中属于这两个录制的窗口也不训练。两组相同240更新、48录制均衡锚点、同时间多人分组、种子17/29/43、相同初始化/抽样/扰动及验证选择。共36次拟合。',
        '上一轮150.96px是原117训练规模的交互模型结果，当前又扩大了训练数据及继续训练；因此只能用两个新组的配对差异隔离深度，不能把与旧数值的全部差异都归因于深度。',
        '所有双目曝光均不晚于对应 RGB 时刻：左图取 RGB−1ms 之前、右图同步匹配，且显式复核时间。最近帧可能来自未来的初始试提取已归档，未用于训练。双目结果通过完整外参及时间对应的MPS位姿映射到RGB。原始mask和MPS本身仍是离线参考，不因此成为在线独立GT。','',
        '## 稳定性与分支诊断','',
        '|种子|无深度 ADE/FDE|有深度 ADE/FDE|','|---|---:|---:|']
    for s in ['17','29','43']:
        x=a['per_seed'][s];y=b['per_seed'][s];lines.append(f"|{s}|{x['image_ADE']:.2f}/{x['image_FDE']:.2f}|{y['image_ADE']:.2f}/{y['image_FDE']:.2f}|")
    c=contrast['image_ADE'];lines += ['',f"录制等权 ADE：{a['recording_equal']['image_ADE']:.2f} → {b['recording_equal']['image_ADE']:.2f} px。稀疏窗口 ADE：{a['sparse']['image_ADE']:.2f} → {b['sparse']['image_ADE']:.2f} px。",f"按种子均值，{c['improved_windows']}/117 个窗口、{c['improved_recordings']}/6 个录制改善。录制等权 ADE 差（有−无）的探索性95% bootstrap区间：{c['exploratory_recording_bootstrap95'][0]:.2f} 至 {c['exploratory_recording_bootstrap95'][1]:.2f} px；只有6个录制，不作为强因果结论。",'',
        'GT-camera、GT-person、相机角度、残差、每种子、录制等权、稀疏窗口及深度完整/部分/缺失子组均保存在forecast/comparison.json、contrasts.json和depth_subgroups.json。不能只根据最终ADE判断哪个分支单独贡献了全部变化。','',
        '## 核验与文件','',
        f"已核验 {count} 行独立误差统计，差异上限 {maxerr:.3g}；独立SciPy几何差异最大 {maxgeom:.6f}px。另见forecast/verification.json中的权重复现、未来后缀删除、无深度输入不变性、冻结骨干、配对抽样以及源文件哈希检查。",'',
        '- frames/：逐帧原始视差、深度、有效区域、标定、时间戳和人物测量。',
        '- depth_all21.npz / depth_history.npz：按374个原样本顺序对齐的深度数据；未来15帧不进入模型。',
        '- forecast/：36个新模型、完整预测、ADE/FDE对照、训练日志和核验。',
        '- audit.json / coverage_by_cohort.json / unique_person_depth.csv：完整覆盖情况与质量记录。',
        '- 原始视频、SAM标注、既有模型、PPT和已选基线不变。',
        '- 深度为可见表面的估计，不是独立身体中心GT；失败模式包括视野限制、多人遮挡、跨相机边界混合和低纹理误配。']
    path=ROOT/'reports/depth_forecast_20261008/RESULTS.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(ADE_before=a['image_ADE'],ADE_after=b['image_ADE'],relative_gain_percent=gain,audit=audit),indent=2))

if __name__=='__main__':main()
