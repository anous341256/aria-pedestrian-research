from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/transformer_d_trimmed_20261029';OLD=ROOT/'outputs/translation_forecast_20261020';TRIM=ROOT/'outputs/trimmed_forecast_finetune_20261028'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=read(OUT/'comparison.json');r=c['results'];data=np.load(OLD/'dataset.npz');meta=read(OLD/'samples.json');nm=read(TRIM/'trimmed_samples.json');rows=read(OUT/'metrics.json');diff=0.;count=0
 for f,p in zip(read(OLD/'folds.json'),read(TRIM/'fold_plan.json')):
  for seed in [17,29,43]:
   for arm in ['original_data','trimmed_data']:
    folder=OUT/arm/f'fold_{f["fold"]}/seed_{seed}';z=np.load(folder/'predictions.npz');e=np.sqrt(np.square(z['image']-data['xy'][z['indices'],16:21]).sum(-1));assert np.max(abs(e-z['image_error']))<1e-10
    for j,i in enumerate(z['indices']):
     rr=next(x for x in rows if x['arm']==arm and x['seed']==seed and x['index']==i);diff=max(diff,abs(rr['ADE']-e[j].mean()),abs(rr['FDE']-e[j,-1]));count+=1
    m=meta if arm=='original_data' else nm;idx=read(folder/'complete.json')['training_indices'];assert all(m[i]['recording'] not in [f['test'],f['validation']] for i in idx)
 assert diff<1e-10;write(OUT/'independent_audit.json',dict(passed=True,prediction_rows=count,max_metric_difference_px=diff,recording_disjoint=True))
 lines=['D版Transformer：原数据与质量裁剪数据对照（2026-10-29）','',
 '用户确认D版：行人、相机历史分别两层self-attention，然后同时双向cross-attention，拼接后直接预测未来图像位置。',
 '结论：裁剪数据小幅改善D版，但D版没有超过原双分支＋平移模型，也没有超过保持最后位置。','',
 '设置：16点历史1.5秒→5点未来0.5秒。原117与裁剪412仅作两组训练池，保持同一原117测试窗口、6录制划分、3种子。',
 '两组均从头训练D版，配对初始权重、240步、batch48、lr.001、wd.01、clip1、无增强、验证最终图像ADE每20步选择、允许step0。',
 '不能沿用旧D的检查点直接微调：旧D为6点→15点，输入输出维度不同。此次没有用大批新构造的G1数据。',
 '行人输入6维（原图位置偏移2、相邻位移2、有效性1、时间1），当前中心2＋内参4作为条件。',
 '相机输入38维：历史旋转3＋平移3＋冻结的匹配划分相机GRU隐状态32。只用历史编码器，不输入相机未来预测。',
 '保留旧D的64维/4头/FF55/零dropout设计，因历史长度和相机特征改变，参数量改变，详见complete.json。',
 '与GRU的比较同时改变架构、损失和输出表示，是系统对比，不是单纯attention消融。','',
 '方法 | ADE px | FDE px | 录制等权ADE | 稀疏窗口ADE']
 for a,name in [('saved_translation','原双分支＋平移'),('original_data','D＋原数据'),('trimmed_data','D＋裁剪数据'),('hold_last','保持最后位置')]:
  v=r[a];lines.append(f'{name} | {v["window_mean"]["ADE"]:.4f} | {v["window_mean"]["FDE"]:.4f} | {v["recording_equal"]["ADE"]:.4f} | {v["sparse"]["ADE"]:.4f}')
 lines+=['','裁剪减原数据对照：',json.dumps(c['trimmed_vs_original'],ensure_ascii=False,indent=2),'',
 '不同种子的D原数据／D裁剪数据ADE：']
 for seed in ['17','29','43']:lines.append(seed+': '+str(r['original_data']['per_seed'][seed]['ADE'])+' / '+str(r['trimmed_data']['per_seed'][seed]['ADE']))
 lines+=['','各录制的D原数据／D裁剪数据ADE：']
 for rec in r['original_data']['per_recording']:lines.append(rec+': '+str(r['original_data']['per_recording'][rec]['ADE'])+' / '+str(r['trimmed_data']['per_recording'][rec]['ADE']))
 lines+=['',f'训练累计{c["training_seconds"]:.2f}秒，含保存/验证；不是部署速度。',
 '验证：36检查点重放与物理移除未来输入完全一致；两组初始权重一致；所有训练与测试/验证录制隔离。',
 '独立NumPy重算702个预测行的ADE/FDE，差<1e-10px。源文件哈希核对不变。',
 '固定原9例seed17三栏对比：左旧双分支、中D原数据、右D裁剪数据；同一参考帧，改善/失败都保留。',
 '两个D模型均直接输出像素位置，不能再把该模型分成独立可评测的相机/行人运动预测头。',
 '已完成但不替换默认基线；不因这次测试结果调超参数或筛样本。']
 report=ROOT/'reports'/OUT.name/'RESULTS.txt';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 assert read(OUT/'verification.json')['passed'] and read(OUT/'presentation/verification.json')['passed'];protected=read(OUT/'protected_sources.json');assert all(sha(ROOT/k)==v for k,v in protected.items());files=[p for folder in [OUT,report.parent,Path(__file__).parent] for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['delivery_hashes.json','delivery_complete.json']];write(OUT/'delivery_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files});write(OUT/'delivery_complete.json',dict(passed=True,files=len(files),protected_files=len(protected)));print(c['trimmed_vs_original'])
if __name__=='__main__':main()
