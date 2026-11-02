from geometry_common import *
import shutil
def main():
 for name in ['training_complete','verification','synthetic_geometry','geometry_preflight','source_geometry_audit','depth_audit','prediction_audit']:assert read(OUT/f'{name}.json')['passed']
 assert read(OUT/'presentation/verification.json')['passed'];c=read(OUT/'comparison.json');r=c['results'];depth=read(OUT/'extraction_complete.json');report=ROOT/'reports'/OUT.name;report.mkdir(parents=True,exist_ok=True)
 lines=['全三维历史补偿、行人预测、相机旋转/平移预测与重投影','2026-10-21','',
 '117窗口、六个录制级划分、种子17/29/43。过去1.5秒16点，预测未来0.5秒5点。',
 '历史像素+深度先还原相机XYZ，再用MPS旋转/位置统一到预测起点RGB相机坐标系。',
 'A=R0.T Rt, b=R0.T(ct-c0), Xfixed=A(Z K^-1 pixel)+b。',
 '未来分别预测Xfixed、A、b，Xcam=A.T(Xfixed-b)，pixel=K Xcam再除Z。',
 '相机继承原旋转网络，新增历史OLS速度基线上的平移残差头，联合微调18次×2000步；原27185相机窗口保持按录制分割。',
 '行人全新GRU64/context/未来flow15维/历史flow48维，18次×240步；masked3D future MSE+.5history MSE，验证3DADE选checkpoint。',
 '没有图像合成loss；未来深度/MPS只做监督和诊断。所有预测只读取16历史点。相机预测在同一时刻多个目标间共享。',
 '此版与旧88.24的区别包括几何、输入、预测目标、架构和训练，属于系统比较，不能单独归因深度。','',
 f'深度：417唯一帧，372复用、45新增。历史有效{depth["history_valid"]}/1872；未来标签有效{depth["future_valid"]}/585。',
 '缺测不删窗口；所有117评图像误差，3D误差及GT-person替换仅在有效未来深度时刻评价。',
 'SAM框中心射线×mask表面中位深度是近似三维代理，不是实际人体中心三维GT；姿态/遮挡/深度噪声仍会影响轨迹。','',
 '方法 | 图像ADE(px) | 图像FDE(px)']
 for name in ['saved_translation','full3d','fixed3D','CV3D','hold_last']:v=r[name]['window_mean'];lines.append(f'{name} | {v["ADE"]:.4f} | {v["FDE"]:.4f}')
 lines+=['','full3d分支误差与替换诊断：',json.dumps(r['full3d']['window_mean'],ensure_ascii=False,indent=2),'',
 'GT_camera替换未来旋转和位置；GT_rotation/GT_translation分别仅替换一项；均保留同一个行人预测。不是可部署成绩。',
 'GT_person使用深度参考三维位置，仅有效标签时刻；与其他全时刻统计分母不同。误差不是可相加的因果贡献。',
 'fixed3D保持最近历史有效三维锚点，CV3D按有效历史三维点OLS速度外推，两者均使用同一学习相机输出。','',
 '统计：',json.dumps(c,ensure_ascii=False,indent=2),'',
 '数学与数据检查：']
 for name in ['synthetic_geometry','geometry_preflight','source_geometry_audit','depth_audit','prediction_audit','verification']:lines += [name,json.dumps(read(OUT/f'{name}.json'),indent=2)]
 lines+=['','所有旧模型/标注/结果/PPT保留。未替换选定基线。','固定九例视频：presentation/fixed_9_full3d_comparison.mp4；右侧参考未来相机面板仅作诊断。','各例 *_branches.png展示行人三维输出、相机旋转输出、相机平移输出和最终图像轨迹。']
 (report/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf8');snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items());assert all(sha(ROOT/k)==v for k,v in read(OUT/'execution_code_hashes.json').items())
 files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.pt','.mp4','.png','.jpg','.py'] and 'frames' not in p.parts and p.name not in ['delivery_hashes.json','delivery_complete.json']]+[report/'RESULTS.txt'];write(OUT/'delivery_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files});write(OUT/'delivery_complete.json',dict(passed=True,protected_files=len(protected),delivery_files=len(files)));print(report/'RESULTS.txt')
if __name__=='__main__':main()
