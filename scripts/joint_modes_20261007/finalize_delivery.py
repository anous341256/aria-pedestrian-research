"""Interpretation, per-fit mode concentration and reproducible delivery manifest."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/joint_modes_20261007'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
rows=read(OUT/'diagnostic_rows.json');folds=read(OUT/'folds.json');lookup={i:f['fold'] for f in folds for i in f['test_indices']}
con=[]
for f in folds:
    for s in [17,29,43]:
        rr=[r for r in rows if r['method']=='joint_K3' and r['seed']==s and lookup[r['index']]==f['fold']]
        counts=np.bincount([r['selected'] for r in rr],minlength=3)
        con.append(dict(fold=f['fold'],seed=s,counts=counts.tolist(),dominant_share=float(counts.max()/counts.sum())))
projection=read(OUT/'projection_finalization.json')['files'];test=[r for r in projection if r['file'].endswith('predictions_test.npz') and '/K3/' in r['file']]
concentration=dict(per_fit=con,mean_dominant_share=float(np.mean([r['dominant_share'] for r in con])),
    candidate_invalid_rays=sum(r['invalid_candidate_rays'] for r in test),selected_invalid_rays=sum(r['invalid_selected_rays'] for r in test))
write(OUT/'mode_concentration.json',concentration)
text='''主要发现与限制

三候选共享模式已完成：117窗口、6个录制隔离折、3个种子；36个适配器模型。
K3 ADE/FDE=160.50/247.40像素，K1=168.15/263.84，原分离GRU/flow=162.44/238.93。
K3比K1的ADE降低4.55%，三个种子均改善；录制等权改善只有0.89像素。
K3比原分离GRU平均ADE降低1.94像素，但FDE增加8.47像素，仅一个种子ADE改善。
K3未超过旧六组attention中的任何一组，也未超过保持最后位置。

分支与误差抵消：
相对原分离模型，角度误差7.646→7.498度，残差ADE74.77→80.31像素。
相对K1，K3的相机与残差平均误差都略差，最终ADE却更好。
K3的平均误差交叉项为负；这说明仍存在抵消，不能把像素改善解释成分解更准确。
相对原模型，351次窗口/种子预测中有11次最终ADE改善但两项分支误差都恶化。
三项监督实现正确，但有限权重允许折中，尚未达到两分支同时改进的目标。

候选与选择：
使用未来参考，按同一三项标准事后挑选一整个模式，诊断ADE为141.59像素；实际选择160.50。
这不是可部署结果，也不保证能通过评分学习达到。只按最终像素挑选的minADE3为139.50。
每个折/种子内部，主导模式平均占96.1%的测试人窗口，三种模式尚未均衡成为有用候选。
模式编号仅在本折内有效，不能把不同折的同编号当成同一个行为。

工程核查：
36个新模型及108个旧attention检查点回放通过；实际预测在删除未来输入后保持一致。
CUDA group-logit求和存在浮点末位差异，单目标logit、模式选择、候选及最终预测一致。
评测已从训练用稳定分母改为float64严格投影；未修改权重、候选旋转/残差、评分或模式选择。
初始投影保留在 *_initial_stabilized.npz。验证集未选中候选含3个相机后方视线；全部选中结果无无效视线。
候选几何也通过独立SciPy核对。447个保护文件未改变。
视频使用旧固定九案例，种子17，37.8秒，378帧全部解码；未按新结果选展示案例。

本版保留为研究记录，不替换已选基线，也未根据本次测试结果调整权重或重训。

===== DETAILED REPORT =====

'''
report=ROOT/'reports/joint_modes_20261007/RESULTS.txt'
previous=report.read_text(encoding='utf-8')
if '===== DETAILED REPORT =====' in previous:previous=previous.split('===== DETAILED REPORT =====',1)[1].lstrip()
report.write_text(text+previous,encoding='utf-8')
paths=list(Path(__file__).parent.glob('*.*'))+[report]
paths += [p for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json' and p.suffix not in ['.log','.partial']]
write(OUT/'delivery_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
print('Final report and delivery manifest saved.')
