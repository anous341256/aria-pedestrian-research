from bulk_common import *

def main():
 for f in ['association_complete.json','dataset_complete.json','source_audit.json','verification.json','presentation/verification.json']:assert read(OUT/f)['passed']
 s=read(OUT/'inventory_summary.json');d=read(OUT/'dataset_complete.json');a=read(OUT/'comparison.json');dep=read(OUT/'depth_complete.json');names={'baseline':'原几何＋整体外观','partial':'几何＋局部/整体外观'};lines=['G1批量标注升级与预测数据构造（2026-10-29）','',
 '本轮把已存在但大部分未进入原小规模预测实验的G1素材，采用新身份关联方法重新整理。',
 '不是新增录制或全新画面，也没有重训/重跑SAM2。SAM mask形状固定，只重新决定身份归属。',
 f'{s["intervals"]}个不重叠时间段，{s["recordings"]}次录制，{s["source_clips"]}个来源片段，{s["frames"]}帧（{s["seconds"]}秒），{s["candidates"]}个人物mask。',
 f'其中{s["frames_outside_original117"]}帧不在原117窗口内，{s["frames_outside_original10"]}帧不在原10段画面内；新增原始视频画面0帧。',
 '旧用户拒绝区间和场景排除继续保留，014/050/057的人工检查警示继续携带。','',
 '方法：原完整几何关联与新局部外观关联分别运行，候选完全相同。',
 '当前相机旋转/光心平移＋双目深度，把前一时刻人的表面深度代理点变换到世界系，匀速估计并投影到当前画面。',
 '深度不可靠时回退旋转二维预测；身份匹配由整体外观、有效局部patch对应、位置、尺度以及旧可靠性规则共同决定。',
 '局部匹配不是显式人体部位标签，也不是经过训练的遮挡分类器。',
 f'深度覆盖{dep["frames"]}帧，{s["cached_depth_frames"]}帧复用原始视差，{s["new_depth_frames"]}帧新增官方顺序推理；{dep["usable"]}/{dep["targets"]}个人物深度通过旧质量规则。',
 f'记录的提取失败数：{len(dep["failures"])}。深度是表面Z代理，不是身体中心3D真值。','',
 '数据交付分层：',
 f'1. 全量：保留所有{s["candidates"]}个mask及其重新关联ID和逐帧质量警示，不统一掐头去尾。',
 f'2. 全部连续预测窗口：{d["all_continuous_windows"]}个21帧窗口，覆盖{d["window_recordings"]}次录制。',
 f'3. 自动质量规则子集：{d["quality_eligible_windows"]}个窗口，覆盖{d["quality_recordings"]}次录制，单独保存，不删除其他轨迹。',
 '每窗口16帧历史1.5秒→5帧未来0.5秒。191个相机时间段本身互不重叠，但同段多人共享画面与相机，不能当成独立场景。',
 '短轨迹、出入画面和带质量警示的轨迹保留在全量标注与视频中；不足21帧无法构成本次预测窗口。',
 '质量规则沿用上次逐帧mask/外观/位置/深度阈值和面积突变标记，本次不执行固定头尾2帧裁剪。',
 '质量子集不是逐帧人工核准的GT；原SAM候选错误不会因为重新关联ID自动消失。','',
 '身份关联诊断（旧SAM来源ID作为参考，不能称为真实身份准确率）：']
 for arm,label in names.items():
  z=a[arm]['total'];lines.append(f'{label}: 来源ID输出变化{z["source_id_assignment_changes"]}，同输出轨迹跨来源变化{z["within_track_reference_identity_changes"]}，额外碎片{z["fragmentation_excess"]}，输出轨迹{z["output_tracks"]}。')
 lines+=['','预测窗口按录制分布（全部／质量子集）：']
 for rec,n in d['per_recording'].items():lines.append(f'{rec}: {n["all"]} / {n["quality"]}')
 lines+=['','数据文件：dataset/all_continuous.npz + metadata.json；dataset/quality_subset.npz + quality_metadata.json。',
 'NPZ含中心、框、内参、参考旋转/补偿矩阵、历史相机平移、深度参考及质量位。未来姿态/深度字段只用于监督或诊断，不能作为历史网络输入。',
 'dataset/future_training_eligibility.json预先列出原6折的训练可用索引，均排除验证和测试录制。尚未把本轮数据并入任何预测训练。',
 '第一任务D版实验只用了旧117与上次412裁剪窗口，不能把D版结果归于本次扩充。','',
 '展示：presentation/all191_association_comparison.mp4（左原关联，右局部外观关联）；191个单段视频；24页中帧总览；连续目标首/中/末检查条带。',
 '检测/分割来自离线双向SAM流程，相机位姿为离线MPS，不能声称整体在线因果跟踪或穷尽行人召回。',
 '独立核验：原始MPS四元数/光心、原始视差→RGB深度统计、RLE外接框、全部预测窗口索引、固定每录制前缀重放、未来移除与候选顺序/假ID检查。',
 '原标签、权重、PPT、CVAT及原选择基线不变。质量分级和可视化供后续人工检查。']
 report=ROOT/'reports'/OUT.name/'RESULTS.txt';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(lines)+'\n',encoding='utf-8');(OUT/'presentation/START_HERE.txt').write_text('先看 all191_association_comparison.mp4 或 clips/编号.mp4。\n左：原几何＋整体外观；右：几何＋局部/整体外观。?表示质量或人工检查警示，不代表已经确认错误。\noverview_*.jpg覆盖全部191段中帧，targets_*.jpg为连续预测目标首/中/末。\n完整标注保留，自动质量子集单独保存，尚未加入预测训练。\n',encoding='utf-8');write(OUT/'packaging_complete.json',dict(passed=True,awaiting_final_visual_review=True,report=str(report.relative_to(ROOT))))
if __name__=='__main__':main()
