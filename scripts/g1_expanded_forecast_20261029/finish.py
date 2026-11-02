from expanded_common import *
from scipy.spatial.transform import Rotation

def main():
 assert read(OUT/'training_complete.json')['passed'];assert read(OUT/'independent_data_audit.json')['passed'];rows=read(OUT/'metrics.json');data=dict(np.load(OLD/'dataset.npz'));meta=read(OLD/'samples.json');allmeta=read(OUT/'expanded_samples.json');count=0;maxerr=0.;maxproj=0.;arms=['gru_original','gru_expanded','d_original','d_expanded']
 for fold in read(OUT/'folds.json'):
  for seed in [17,29,43]:
   for arm in arms:
    folder=OUT/arm/f'fold_{fold["fold"]}/seed_{seed}';z=dict(np.load(folder/'predictions.npz'));idx=z['indices'];e=np.sqrt(((z['image']-data['xy'][idx,16:21])**2).sum(-1));maxerr=max(maxerr,float(abs(e-z['image_error']).max()));trainidx=read(folder/'complete.json')['training_indices'];m=meta if arm.endswith('original') else allmeta;assert all(m[i]['recording'] not in [fold['test'],fold['validation']] for i in trainidx)
    for j,i in enumerate(idx):
     row=next(r for r in rows if r['arm']==arm and r['seed']==seed and r['index']==i);maxerr=max(maxerr,abs(row['ADE']-e[j].mean()),abs(row['FDE']-e[j,-1]));count+=1
     if arm.startswith('gru'):
      for t in range(5):
       ray=np.linalg.solve(data['K'][i],np.r_[z['component'][j,t],1.]);rot=Rotation.from_rotvec(z['rotation'][j,t]).as_matrix();q=data['K'][i]@(rot.T@ray);p=q[:2]/q[2];maxproj=max(maxproj,float(abs(p-z['image'][j,t]).max()))
 assert maxerr<1e-9 and maxproj<1e-8
 write(OUT/'independent_prediction_audit.json',dict(passed=True,prediction_rows=count,metric_max_px=maxerr,scipy_geometric_projection_max_px=maxproj,heldout_recording_disjoint=True))
 c=read(OUT/'comparison.json');r=c['results'];pre=read(OUT/'preflight.json');lines=['新G1加入预测训练：原双分支＋平移与D版对照（2026-10-29）','',
 '本轮结果：GRU加入G1后ADE86.6761/FDE130.1423，优于匹配原数据继续训练88.4597/132.8921，也略优于保存原版88.2420/131.2916。',
 'GRU对匹配对照ADE改善2.0163%，三个种子均改善，67/117窗口、5/6录制改善；录制等权及稀疏统计也改善。6录制探索性区间仍跨0，属于初步小幅收益。',
 'D加入G1后ADE114.7078->112.6991，但FDE160.5268->161.8664变差；三个种子仅一个ADE改善，不能称为稳定提升。D仍弱于GRU与保持最后位置。',
 'GRU残差ADE51.3570->44.6801；相机输出完全不变。行人分支预测改善更明显，但最终图像误差改善幅度较小，不能将两者误差直接相加或作因果比例分解。','',
 '在同一117窗口、6录制隔离折、3种子上评测；输入历史1.5秒16点，预测未来0.5秒5点。',
 '新增数据只使用上一轮299个自动质量通过窗口；未用其余30个候选，也不把旧412裁剪池重复合入。',
 f'原117＋新299，共{pre["expanded_windows"]}窗口、{pre["expanded_recordings"]}次录制；完全重复删除{pre["exact_duplicates_removed"]}条。13条新增窗口与旧数据有部分画面时间重叠，同录制始终在同一训练/验证/测试划分中。',
 '每折排除验证与测试两个录制，新数据仅进入训练。原测试/验证标签与窗口完全不变。','',
 '训练方法：',
 'GRU：原双分支＋平移模型，冻结匹配保存的相机；行人从匹配保存权重开始，以lr1e-4微调全部行人权重。原数据继续训练是配对对照。',
 'D：各自两层self-attention，再同时双向cross-attention，拼接直接预测图像位置；两组从头训练，lr.001。冻结匹配相机历史GRU编码器。',
 '两类各2组，各6折×3种子，共72次训练；240步、batch48、录制均衡采样、wd.01、clip1、无增强。每20步按原验证集选权重，允许step0。',
 'GRU使用未来＋0.5历史重建flow损失、验证残差ADE；D使用图像MSE、验证图像ADE。模型之间是系统比较，不能孤立归因给attention。',
 '相机模型不重训。新双目深度仅用于此前生成关联标注，本次未作为预测特征。历史输入不使用未来姿态/深度。','',
 '方法 | ADE像素 | FDE像素 | 录制等权ADE | 稀疏窗口ADE']
 names={'saved_translation':'保存的原双分支＋平移','gru_original':'GRU仅原数据继续训练','gru_expanded':'GRU原数据＋新G1','d_original':'D仅原数据','d_expanded':'D原数据＋新G1','hold_last':'保持最后位置'}
 for arm in ['saved_translation']+arms+['hold_last']:
  v=r[arm];lines.append(f'{names[arm]} | {v["window_mean"]["ADE"]:.4f} | {v["window_mean"]["FDE"]:.4f} | {v["recording_equal"]["ADE"]:.4f} | {v["sparse"]["ADE"]:.4f}')
 lines+=['','新G1减对应原数据对照（负差为改善）：']
 for family in ['gru','d']:lines.append(family+': '+json.dumps(c['contrasts'][family+'_expanded_vs_'+family+'_original'],ensure_ascii=False))
 lines+=['','各随机种子ADE（原数据／加入G1）：']
 for family in ['gru','d']:
  for seed in ['17','29','43']:lines.append(f'{family} seed{seed}: '+str(r[family+'_original']['per_seed'][seed]['ADE'])+' / '+str(r[family+'_expanded']['per_seed'][seed]['ADE']))
 lines+=['','各录制ADE（GRU原／扩充；D原／扩充）：']
 for rec in r['gru_original']['per_recording']:lines.append(rec+': '+' / '.join(f'{r[a]["per_recording"][rec]["ADE"]:.4f}' for a in arms))
 lines+=['','各折原训练数量＋新G1数量：',json.dumps(pre['fold_counts'],ensure_ascii=False),'',
 '采用同样训练步数而非同样epoch；录制均衡采样意味着新增录制改变各录制在训练中的占比。当前结果反映整个扩充训练方案，不能单独解释为纯数据量效应。',
 '新增标注仍为自动关联后经过抽样检查的参考，不是逐帧人工GT；质量筛选使用离线完整片段，不声称在线标注流程。',
 '置信区间是6录制重采样的探索性区间；测试录制数仍只有6，不能因训练录制增多就声称大规模测试泛化。',
 f'72次训练累计{c["training_seconds"]:.2f}秒（含验证和保存），不是部署耗时；step0选择数量：'+json.dumps(c['step0']),
 f'核验：72检查点/未来截断重放完全一致；36个原数据对照与前次保存输出完全一致；独立{count}预测行ADE/FDE差{maxerr:.3g}px，SciPy重投影差{maxproj:.3g}px。',
 '新增数据416窗口/8736帧位置与源数据逐项匹配；位姿、补偿矩阵、历史时间、训练录制隔离检查通过。',
 '原版本全部保留，新权重独立存储；未覆盖默认基线、原标签、PPT或CVAT。',
 '展示：presentation/gru和presentation/d各一组固定原9例seed17对比，左保存GRU、中该模型原数据、右该模型加入G1；包含失败，不按结果换例。']
 report=ROOT/'reports'/OUT.name/'RESULTS.txt';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8');print({a:r[a]['window_mean'] for a in names})

if __name__=='__main__':main()
