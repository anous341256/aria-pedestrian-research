from depth_common import *
import shutil,collections
def main():
 for name in ['extraction_complete','training_complete','verification','independent_audit']:assert read(OUT/f'{name}.json')['passed']
 assert read(OUT/'presentation/verification.json')['passed']
 c=read(OUT/'comparison.json');d=read(OUT/'extraction_complete.json');m=read(BASE/'samples.json');f=np.load(OUT/'depth_features.npz')['features'];inv=read(OUT/'inventory.json');rows=read(OUT/'measurements.json');reasons=collections.Counter(r for row in rows for r in row['reasons'])
 coverage={rec:dict(windows=sum(x['recording']==rec for x in m),valid=float(f[[i for i,x in enumerate(m) if x['recording']==rec],:,2].mean())) for rec in sorted({x['recording'] for x in m})}
 write(OUT/'coverage.json',dict(per_recording=coverage,overlapping_invalid_reasons=dict(reasons)))
 report=ROOT/'reports'/OUT.name;report.mkdir(parents=True,exist_ok=True)
 lines=['历史深度加入行人预测分支：配对实验结果','2026-10-21','',
 '范围：117窗口、6个录制级划分、3个种子；过去1.5秒16点预测未来0.5秒5点。',
 '继承对应fold/seed的双分支＋平移权重。相机冻结；行人GRU9->15，新六列零初始化，两组都微调全部行人参数。',
 '六维深度：归一化logZ、相邻logZ差、当前有效位、差分有效位、左右一致率、IQR/Z。',
 '无深度组所有新增输入置零；有深度组用历史测量。没有未来深度，没有深度几何补偿，没有未来平移头。',
 '训练：每组18次、240步、batch48、lr.001、原flow未来损失+.5历史重建损失；每20步验证剩余ADE，step0可选。',
 '两组均无轨迹扰动增强以保持mask深度对应。原88.24作为额外参考；新无深度组控制额外微调。','',
 f'历史覆盖：{inv["unique_frames"]}唯一画面，复用{inv["cached"]}、新增{inv["missing"]}；全部成功处理。',
 f'有效人体帧：{d["valid"]}/{d["target_frame_occurrences"]} ({d["valid_fraction"]:.1%})；有任意深度窗口{d["windows_with_any"]}；16帧全有效窗口{d["windows_with_all"]}。',
 '完整处理不等于所有人都测到；缺测保留，没有按深度删窗口。左右一致性和分散度是质量线索，不是准确率。','',
 '方法 | 图像ADE | 图像FDE | 剩余ADE | 相机平均角误差']
 for a,label in [('saved_translation','原双分支＋平移'),('no_depth','配对微调无深度'),('with_depth','配对微调有深度'),('hold_last','保持最后位置')]:
  r=c['results'][a]['window_mean'];res=f'{r["residual_ADE"]:.4f}' if 'residual_ADE' in r else '-';angle=f'{r["camera_mean_deg"]:.4f}' if 'camera_mean_deg' in r else '-';lines.append(f'{label} | {r["ADE"]:.4f} | {r["FDE"]:.4f} | {res} | {angle}')
 lines+=['','配对对照与原模型参考：',json.dumps(c['contrasts'],ensure_ascii=False,indent=2),'',
  '按录制等权/种子/稀疏窗口：',json.dumps(c['results'],ensure_ascii=False,indent=2),'',
  '深度可用性：',json.dumps(coverage,ensure_ascii=False,indent=2),json.dumps(dict(reasons),ensure_ascii=False,indent=2),'',
  f'第0步入选次数：{c["step0_selected"]}；累计训练秒数：{c["training_seconds"]:.2f}。',
  '解释边界：比较的是深度数值＋有效性＋质量这一输入包，未单独分离距离与缺测模式的贡献。只有6个录制；bootstrap为探索性统计。',
  '测量为人体可见表面的RGB轴向深度，不是人体中心三维GT；深度变化仍混合相机/人体运动及标注误差。',
  '固定原九例seed17视频：presentation/fixed_9_depth_comparison.mp4。白色参考中心，橙色预测，下方为16帧历史深度有效性。',
  '未覆盖旧实验、旧权重、标签或PPT；没有替换选定基线。','',
  '验证：',json.dumps(read(OUT/'verification.json'),indent=2),json.dumps(read(OUT/'independent_audit.json'),indent=2)]
 (report/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf8')
 snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
 for k,v in read(OUT/'source_hashes.json').items():assert sha(ROOT/k)==v
 files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.pt','.mp4','.png','.jpg','.py'] and 'frames' not in p.parts and p.name not in ['delivery_hashes.json','delivery_complete.json']]+[report/'RESULTS.txt']
 write(OUT/'delivery_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files});write(OUT/'delivery_complete.json',dict(passed=True,files=len(files)))
 print(report/'RESULTS.txt')
if __name__=='__main__':main()
