from assoc_common import *
import collections
def main():
 for f in ['geometry_preflight.json','pose_complete.json','depth_complete.json','appearance_complete.json','tracking_complete.json','verification.json','source_audit.json','presentation/verification.json']:assert read(OUT/f)['passed'],f
 core=read(OUT/'execution_code_hashes.json');assert all(sha(Path(__file__).parent/k)==v for k,v in core.items())
 protected=read(OUT/'source_hashes.json');assert all(sha(ROOT/k)==v for k,v in protected.items())
 inv=read(OUT/'inventory.json');comp=read(OUT/'comparison.json');extra=read(OUT/'identity_diagnostics.json');rep=ROOT/'reports'/VERSION;rep.mkdir(parents=True,exist_ok=True)
 lines=['SAM2固定候选＋独立身份关联：相机旋转、平移和深度辅助位置估计','2026-10-27','',
 '本轮已完成：10段原始近距离视频，530帧/53秒，1486个固定SAM2候选，46个原有片段内参考身份。',
 '原标注、原预测模型均未改动。未继续3秒→1.5秒预测实验。',
 '这轮重新运行独立关联模块，复用之前的SAM2 mask，没有重新训练或重新运行SAM2。候选按mask内容哈希匿名化，原身份/segment只进入事后评价。',
 '候选来自双向、带身份引导和人工核查的离线标注，可能已利用未来上下文。因此这是固定候选条件下的关联试验，不是完整在线跟踪基准，也不检验漏检召回。',
 '', '数据流与数学定义：',
 'R_t为RGB相机到世界坐标的旋转；c_t为RGB光心在世界中的位置，包含设备到RGB的杆臂。K为去畸变RGB内参；Z是RGB光轴方向深度，不是欧氏距离。',
 '1. 上次像素q=[u,v,1]^T加深度：X_world=R_prev [Z K^-1 q]+c_prev。',
 '2. 用已接受的历史观测维护三维位置/速度及协方差，匀速Kalman传播到当前时刻：Xhat_world=X_world+v_world*dt。',
 '3. 使用当前已发生的相机运动：Xhat_cam=R_current^T (Xhat_world-c_current)。',
 '4. h=K Xhat_cam，预测像素=(h_x/h_z,h_y/h_z)。通过投影雅可比传播位置不确定性。',
 '5. 当前SAM2候选的外观、像素位置、可用深度共同计算关联代价，匈牙利算法做全局一对一匹配，允许未匹配及新建轨迹。',
 '这里无需预测未来相机运动；但离线MPS/深度生成不构成实时系统的证明。',
 '', '三组：',
 'image_motion：原图2D匀速Kalman＋相同外观/尺寸/生命周期。',
 'rotation：统一到片段初始朝向的2D匀速Kalman，投影回当前图像。',
 'rotation_translation_depth：三维匀速Kalman＋当前相机旋转/平移＋深度关联线索，失效时退回rotation。',
 '三组都使用相同冻结DINOv2 ViT-S/14外观特征、候选、基础权重与一对一关联规则。没有训练和结果驱动调参。',
 '完整组同时改变状态维度、不确定性传播并加入深度距离证据，不能将全部差异单独归因于平移或深度均值。',
 '', '质量与降级：',
 '462个有候选的RGB时刻需要深度，复用417帧、新算45帧。1325/1486个候选深度通过既有质量规则；161个缺失/不可靠候选仍保留。',
 '深度年龄>0.5秒、相对Z标准差>25%、投影不确定性太大或落到相机后方时退回旋转模型。',
 '深度创新超过阈值时拒绝该次3D更新；自然序列有5次。含糊匹配不更新外观，2D观测噪声加大并跳过3D更新。目标缺测保留最长1秒。',
 '噪声、阈值和权重是固定先验，不是经过校准的概率；图中95%椭圆仅是该高斯模型的名义范围，绘制半轴上限180显示像素，不保证真实95%覆盖。',
 '三维点仍是框中心射线乘mask表面中位Z，姿态、裁边和遮挡会改变它，不是准确的人体中心。',
 '', '自然序列结果（越少越好；仅对原有身份编号的一致性诊断）：',
 '方法 | 同一参考人输出ID变化 | 输出轨迹中参考身份变化 | 额外轨迹碎片 | 输出轨迹数']
 for a in ARMS:
  t=comp['natural'][a]['total'];lines.append(f"{a} | {t['source_id_assignment_changes']} | {extra['natural'][a]['within_track_reference_identity_changes']} | {t['fragmentation_excess']} | {t['output_tracks']}")
 lines += ['', '“同一参考人输出ID变化”统计该参考人相邻可见候选被分配不同输出ID，包含断轨、新建ID和重新接回旧ID；不是标准人工GT的IDSW。',
  '“输出轨迹中参考身份变化”统计一条输出轨迹相邻候选的参考身份发生变化，更接近串人诊断；原参考本身仍可能有错误。',
  '“额外轨迹碎片”是每个参考人使用的不同输出ID数量减一再求和。不能把较低断轨率直接叫做人类核验身份准确率。',
  '', '按片段：clip | image/rotation/full 的参考ID变化次数']
 for c in inv['clips']:
  cid=c['clip'];lines.append(cid+' | '+' / '.join(str(comp['natural'][a]['per_clip'][cid]['source_id_assignment_changes']) for a in ARMS))
 lines += ['', '全几何组相对旋转组：4/10片段改善，6/10持平；4/6录制改善，2/6持平。总体63→40，少23次（36.5%）。',
  '自然序列全几何组1409次匹配中，1201次采用full3D、208次退回rotation；另有77次新建轨迹。',
  '', '人工构造短缺测对照：每20帧后的前三帧（20..22、40..42、60..62）统一隐藏所有候选、外观及深度，相机位姿仍可用；不是实际自然遮挡。',
  '保留1308候选/1262条参考身份相邻连接，不与自然序列原始次数直接比较。',
  '方法 | 参考ID变化 | 输出轨迹中参考身份变化']
 for a in ARMS:lines.append(f"{a} | {comp['blackout3'][a]['total']['source_id_assignment_changes']} | {extra['blackout3'][a]['within_track_reference_identity_changes']}")
 lines += ['', '具体案例（选择用于定位机制，不改变全体统计）：',
 '10/N02/帧8：蓝衣迎面行人。旋转组原轨迹预测误差199.57px，Mahalanobis平方18.48，被运动门限拒绝后新建ID。完整组56.49px、0.58，沿用原ID；两组外观相似度此刻相同0.79990。',
 '07/N05/帧23：画面边缘仅剩部分人体；完整组保留ID，但像素预测误差157.31px反而大于旋转组110.01px。更宽的不确定性与此前记忆差异共同影响关联，不能将此例解释为三维位置均值更准确。',
 '09/N02/帧38：夜间背向行人；当前深度无效，仍利用最近历史3D状态；原轨迹预测19.23px、旋转27.25px，完整组维持原ID。画面较暗，视觉判断有限。',
 '07/N04/帧26负例：接近相机边缘的迎面行人。完整组深度不确定性过大已降级rotation，但先前关联历史已不同；原轨迹位置偏371.22px、外观0.537，未通过运动门限而新建ID，旋转组则维持。不能把该错误直接归因当前帧深度值。',
 '01/N01/帧4：单人场景仍断轨。旋转/完整组位置误差12.31/18.47px已很小，但外观0.4285低于固定0.45门限，导致新建ID。说明外观硬门限仍可能过严，本轮保留问题，没有看结果后改参数。',
 '已目视检查10个固定中间帧，以及上述5组前后时序裁剪；没有逐帧独立人工GT审核。',
 '', '验证：',
 '60次完整回放精确；60次物理截断未来输入的前缀回放精确；60次候选顺序反转/注入伪身份字段不影响输出；40次对照改变平移和深度完全不影响输出。',
 '独立四元数重建530帧位姿；逐一核查462帧原始视差、RGB投影及1486候选深度统计，统计差异为0。',
 '静止世界点在相机平移和旋转下的预测误差<1e-8px；投影雅可比有限差分检查通过；56个受保护源文件未变。',
 '22个视频全部解码完成：自然/合成缺测各10个单片＋各1个53秒合集。',
 '', '交付：',
 str(OUT/'presentation/all10_natural_comparison.mp4'),str(OUT/'presentation/all10_blackout3_comparison.mp4'),str(OUT/'presentation/all10_midframes.jpg'),str(OUT/'case_diagnostics.json'),
 '', '结论：这批固定SAM2候选上，相机旋转补偿明显提高参考身份连续性；再加入平移、深度和三维不确定性后有进一步改善。尚不能宣称完整在线身份准确率提升，也没有解决全部身份关联失败。']
 (rep/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
 (OUT/'presentation/START_HERE.txt').write_text('先看 all10_natural_comparison.mp4：左原图运动，中旋转补偿，右旋转＋平移＋深度。53秒，全部10段。\n再看 all10_blackout3_comparison.mp4：人工屏蔽3帧候选的对照，53秒。\n彩色mask和T编号为各组独立跟踪输出；ref编号只供事后核对，未输入关联。橙色十字是预测位置，椭圆是不确定范围。*表示相对原参考人输出编号发生变化，不能直接当成人工GT身份错误。\ncontrast_01..05 为定位问题的时序裁剪，含改善和失败。\n',encoding='utf8')
 hashes={str(p.relative_to(ROOT)):sha(p) for base in [OUT,Path(__file__).parent,rep] for p in base.rglob('*') if p.is_file() and p.suffix not in ['.pyc','.log'] and p.name not in ['delivery_hashes.json','delivery_complete.json']}
 write(OUT/'delivery_hashes.json',hashes);write(OUT/'delivery_complete.json',dict(passed=True,delivery_files=len(hashes),protected_files=len(protected),videos=22,association_clip_arm_runs=60,source_masks_unchanged=True))
 print(str(rep/'RESULTS.txt'),flush=True)
if __name__=='__main__':main()
