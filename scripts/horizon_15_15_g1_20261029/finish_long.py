from pathlib import Path
import json,hashlib,shutil,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/horizon_15_15_g1_20261029';REG=ROOT/'experiment_registry/experiments'/OUT.name
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 cfg=read(OUT/'protocol.json');c=read(OUT/'comparison.json');aud=read(OUT/'independent_audit.json');assert aud['passed'] and aud['prediction_rows']==390;assert read(OUT/'verification.json')['passed'];d=dict(np.load(OUT/'dataset.npz'));meta=read(OUT/'samples.json');folds=read(OUT/'folds.json');res=c['results'];rows=read(OUT/'metrics.json')
 # Same target frames, but not a matched training ablation against old short-horizon checkpoints.
 oldroot=ROOT/'outputs/g1_expanded_forecast_20261029';om=read(ROOT/'outputs/translation_forecast_20261020/samples.json');oldlookup={(m['recording'],m['clip'],m['id'],tuple(m['timestamps_ns'])):i for i,m in enumerate(om)};oldrows=read(oldroot/'metrics.json');short=[];mapping=[];unmatched=[]
 for i,m in enumerate(meta[:cfg['original_windows']]):
  key=(m['recording'],m['clip'],m['id'],tuple(m['timestamps_ns'][:21]))
  if key not in oldlookup:unmatched.append(i);continue
  oi=oldlookup[key];mapping.append(dict(long_index=i,old117_index=oi));short += [dict(r,long_index=i) for r in oldrows if r['arm']=='gru_expanded' and r['index']==oi]
 matchset={m['long_index'] for m in mapping};newmatch=[r for r in rows if r['arm']=='expanded_long' and r['index'] in matchset]
 shortref=dict(windows=len(mapping),ADE=float(np.mean([r['ADE'] for r in short])),FDE=float(np.mean([r['FDE'] for r in short])),new_long_first5_ADE=float(np.mean([r['ADE_05'] for r in newmatch])),new_long_first5_FDE=float(np.mean([r['FDE_05'] for r in newmatch])),mapping=mapping,unmatched_long_indices=unmatched,note='Exact old21-frame match only. Long-window stride phase can differ; unmatched windows excluded ONLY from this secondary comparison, never from main65 evaluation. Saved 1.5s->0.5s expanded GRU on same first5 future targets. Descriptive system reference: training pool, initialization and one validation pairing differ. Test recording excluded in both.')
 write(OUT/'saved_short_reference.json',shortref)
 curves={}
 for a in cfg['arms']:
  errors=[]
  for f in folds:
   for s in cfg['seeds']:errors.append(np.load(OUT/a/f'fold_{f["fold"]}/seed_{s}/predictions.npz')['image_error'])
  curves[a]=np.concatenate(errors).mean(0)
 curves['hold_last']=np.linalg.norm(d['xy'][:cfg['original_windows'],15,None]-d['xy'][:cfg['original_windows'],16:31],axis=-1).mean(0)
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
 for a,col,label in [('original_long','#d58c36','Old-data training'),('expanded_long','#278456','Old + new G1'),('hold_last','#7b818b','Hold last')]:ax.plot(np.arange(1,16)/10,curves[a],'-o',markersize=3,color=col,label=label)
 ax.set(xlabel='Prediction time (seconds)',ylabel='Mean center error (pixels)',title='Same 65 held-out windows, three seeds');ax.legend(frameon=False);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'presentation/error_by_time.png',dpi=180);plt.close(fig)
 write(OUT/'error_by_time.json',{k:v.tolist() for k,v in curves.items()})
 vis=dict(reviewed=True,fixed_test_cases=5,new_recording_target_windows=12,new_target_times=36,notes=['Fixed first eligible test window per recording, seed17; all five final frames inspected.','Old08_N02 remains poor; old09_N01 expanded improves locally while aggregate worsens.','New target first/middle/last views span station, mall, street and night; some targets start relatively small or have partial views. No clean dense manual-GT claim.','No post-result target removal or retraining.'])
 write(OUT/'visual_review.json',vis)
 lines=['双分支＋历史平移＋新G1：过去1.5秒预测未来1.5秒','2026-10-29','',
 '结论：本次扩充没有改善长时域预测。旧长窗口训练ADE171.9538/FDE242.9987，加入新G1后ADE209.1662/FDE366.8783；保持最后位置141.6382/201.4532。',
 '扩充相对旧数据ADE增加21.64%，三个随机种子总体ADE均变差；65窗口中17改善，五录制中两组改善。不能据此单独断定数据质量差或增加数据无效。','',
 '设置与数据：',
 '16个历史点跨1.5秒，预测15个未来点到1.5秒；需要真实连续31帧。',
 '原117个21帧窗口的同身份/同segment一致标签并集形成65个长窗口，覆盖5个测试录制。原193311录制没有足够连续31帧，不能继续声称是原117测试集。',
 '191个已锁定G1短片段中找到28组时间连续的相邻片段；使用原全几何＋局部/整体外观关联算法连续重跑，复用SAM候选、深度和特征，不按旧source-ID强接，不填补缺帧。',
 '得到171个连续31帧候选；质量/人工标记筛选后164个，再在整31帧上核查原相机条件（最大1秒旋转20度，角速度P95不超过50度/秒），跨边界22个不合格，最终142个。',
 f'总池65+142={cfg["combined_windows"]}，14个录制，零完全重复；同录制仍可能有重叠窗口，不能当作独立人物数。新数据只参与训练。',
 '五个旧测试录制按排序循环选择下一录制验证；因此相对旧六折有一处验证录制变化。所有人物训练及相机预训练都排除测试和验证两个录制。','',
 '模型与训练：',
 '保留双分支结构：相机GRU输入历史旋转＋三维平移特征，输出未来15个旋转；行人GRU输入旋转补偿历史轨迹＋历史平移特征，future-flow输出15个残差位置，history-flow辅助重建16点；最后旋转重投影。',
 '平移仍仅作为特征，不做三维平移补偿，也不预测未来平移；没有RGB、眼动、深度预测特征或cross-attention。',
 '未来输出维度由5改15，本轮相机和行人都从头训练，不是原0.5秒最佳权重的直接续训。',
 '15个相机模型在26901个相机长窗口/28录制上训练，每个2000更新；两组共享对应相机。30个行人模型各240更新，batch48、录制均衡抽样、AdamW lr.001/wd.01、梯度裁剪1，无增强。',
 '三个种子17/29/43；相机按验证旋转向量MSE选模，行人按验证残差ADE选模，step0可选，本轮行人均选非零步。未来损失+0.5历史重建；5次固定采样，10步flow。',
 f'累计训练计时{c["training_seconds"]:.2f}秒（含验证/保存，不含数据整理与关联），不是部署耗时。','',
 '同一65测试窗口，三个种子平均，像素↓：',
 '方法 | 前0.5s ADE | 前1.0s ADE | 完整1.5s ADE | 1.5s FDE | 录制等权ADE | 稀疏ADE']
 for a in ['original_long','expanded_long','hold_last','raw_cv']:
  r=res[a];v=r['window_mean'];lines.append(f'{a} | {v["ADE_05"]:.4f} | {v["ADE_10"]:.4f} | {v["ADE"]:.4f} | {v["FDE"]:.4f} | {r["recording_equal"]["ADE"]:.4f} | {r["sparse"]["ADE"]:.4f}')
 lines+=['','分支诊断：',
 '两组相机输出完全相同：平均角误差8.4695度，1.5秒末端11.5203度。',
 '行人残差ADE：旧70.7421 -> 扩充110.2914；末端残差118.9618 ->229.1041。',
 '保持行人预测，换GT未来旋转后最终ADE：旧71.0457、扩充108.0541；仅为诊断，不能算可部署结果。',
 '因此本轮扩充引起的差异来自行人模型输出，长时域相机误差仍是两个系统共同的限制。最终像素误差不是两个分支误差的简单相加。',
 '新增数据前0.5秒ADE109.7969->108.4546略降，后段明显恶化；FDE前0.5秒152.5464->154.6280也没有改善。','',
 '逐种子ADE，旧/扩充：']
 for s in ['17','29','43']:lines.append(f'{s}: {res["original_long"]["per_seed"][s]["ADE"]:.4f} / {res["expanded_long"]["per_seed"][s]["ADE"]:.4f}')
 lines+=['','逐录制ADE，旧/扩充/保持位置：']
 for g in cfg['test_recordings']:lines.append(g+': '+' / '.join(f'{res[a]["per_recording"][g]["ADE"]:.4f}' for a in ['original_long','expanded_long','hold_last']))
 lines+=['','与之前86.68的关系：',
 f'只有{shortref["windows"]}个长窗口的前21帧与旧117窗口完全重合；其余因取窗步长相位不同，仅不参与这项附加比较，主评测65窗全部保留。在这{shortref["windows"]}窗相同前5个目标上，旧0.5秒扩充模型ADE/FDE为{shortref["ADE"]:.4f}/{shortref["FDE"]:.4f}；新1.5秒扩充模型的前0.5秒为{shortref["new_long_first5_ADE"]:.4f}/{shortref["new_long_first5_FDE"]:.4f}。训练数据、初始化及一处验证划分不同，不是仅改变预测长度的严格消融。',
 '不能把本次209.17与旧117窗0.5秒的86.68直接相减，认定为纯预测长度代价。','',
 '限制：',
 '本次可用长窗口池不同于旧299短窗口池；关联状态、样本组成、长度资格也改变。固定240更新下的采样暴露和训练数量并不相同。',
 '五录制中192702占45/65窗口；已同时给录制等权和稀疏统计，扩充组仍更差，但六/五录制都不足以声称广泛泛化。',
 '五录制等权bootstrap扩充减旧数据区间为'+str(c['contrasts']['original_long']['recording_bootstrap_95'])+'像素，跨零；不做统计显著性宣称。',
 '标签与MPS来自离线处理和抽样检查，非独立密集人工GT、非在线端到端成绩；质量筛选使用完整片段。','',
 '验证：30组联合模型检查点/未来物理截断重放完全一致；15个相机状态由对应检查点加载，共45个已训练模型。同时间相机输出一致，零无效投影射线；284个受保护源文件未变。',
 f'独立检查{aud["old_target_frames"]}+{aud["new_target_frames"]}个目标帧、{aud["camera_histories"]}组相机历史；手写四元数平移差最大{aud["manual_quaternion_translation_max_m"]:.3g}米；390行指标重算差0，SciPy重投影差{aud["scipy_projection_max_px"]:.3g}像素。',
 '展示视频固定每个测试录制第一窗、seed17，共5例15.5秒/155帧，全部解码通过，五例末帧已检查。另检查12个新增录制代表窗口的起点/中点/终点，共36个目标时刻。',
 '原模型、标签和结果全部保留；不替换0.5秒最佳基线，不按本次测试结果调参或删样本。',
 '入口：outputs/horizon_15_15_g1_20261029/presentation/START_HERE.txt','',
 '每折数量：',json.dumps(read(OUT/'preflight.json')['fold_counts'],ensure_ascii=False)]
 report=ROOT/'reports'/OUT.name/'RESULTS.txt';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 p=OUT/'presentation/START_HERE.txt';p.write_text(p.read_text().replace('黄色参考未来轨迹','青色参考未来轨迹'),encoding='utf-8')
 REG.mkdir(parents=True,exist_ok=True)
 for filename in ['protocol.json','preflight.json','folds.json','samples.json','joined_inventory.json','duplicates.json','comparison.json','metrics.json','error_by_time.json','saved_short_reference.json','source_hashes.json','pose_sources.json','execution_code_hashes.json','verification.json','independent_audit.json','training_complete.json','visual_review.json']:
  shutil.copyfile(OUT/filename,REG/filename)
 external=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='delivery_complete.json':external.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
 write(REG/'external_artifacts.json',external);write(OUT/'delivery_complete.json',dict(passed=True,files=len(external),video_frames=155,report=str(report.relative_to(ROOT)),code_commit='5287431',git_registry=str(REG.relative_to(ROOT)),preserved_original_baseline=True));shutil.copyfile(OUT/'delivery_complete.json',REG/'delivery_complete.json');print(dict(short_reference=shortref['ADE'],external_artifacts=len(external),report=str(report)),flush=True)
if __name__=='__main__':main()
