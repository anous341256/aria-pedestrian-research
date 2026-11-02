"""Build delivery report after the frozen experiment verification."""
import run as core
from prepare import *
import numpy as np
def main():
 assert read(OUT/'training_complete.json')['passed'] and read(OUT/'verification.json')['passed']
 c=read(OUT/'comparison.json');r=c['results'];dest=ROOT/'reports/horizon_swap_20261017';dest.mkdir(parents=True,exist_ok=True)
 lines=['过去1.5秒预测未来0.5秒：原始分离双分支实验','',
 '模型：PPT版相机GRU + 行人GRU/未来flow/历史重建flow，最后旋转重投影。无cross-attention、深度、眼动。',
 '固定117段原始两秒序列、6录制、3随机种子。新预测时刻是原切窗第16帧，输入16点、预测5点。',
 '短历史对照取同一预测时刻之前6点，预测完全相同5点。原0.5->1.5评分仅作描述，预测时刻和目标时间不同。','',
 '方法 | ADE px | FDE px | 录制等权ADE px']
 for arm in ['saved_0.5_to_1.5','short_history_control','long_history','hold_last','raw_constant_velocity']:
  v=r[arm];lines.append(f"{arm} | {v['window_mean']['ADE']:.3f} | {v['window_mean']['FDE']:.3f} | {v['recording_equal']['ADE']:.3f}")
 lines+=['','分支诊断：方法 | 剩余ADE | 相机平均角误差deg | 相机终点角误差deg | GT相机+预测行人ADE | 预测相机+GT行人ADE']
 for arm in core.ARMS:
  v=r[arm]['window_mean'];lines.append(f"{arm} | {v['residual_ADE']:.3f} | {v['camera_mean_deg']:.3f} | {v['camera_final_deg']:.3f} | {v['gt_camera_ADE']:.3f} | {v['gt_person_ADE']:.3f}")
 lines+=['','各随机种子ADE：']
 for arm in core.ARMS:lines.append(f"{arm}: "+str({s:round(v['ADE'],3) for s,v in r[arm]['per_seed'].items()}))
 lines+=['','长历史减去短历史/保持最后位置的配对对照（负数为改善）：',json.dumps(c['contrasts'],ensure_ascii=False,indent=2),
  '',f"训练72个模型合计{c['training_seconds']:.1f}秒。各组相机/行人总参数：{c['parameters_camera_person']}。",
  '比较两组共享相同目标、划分、训练预算和抽样索引；历史重建头输出长度与历史时长一起改变，不能称为完全相同参数数量的历史长度纯因果试验。',
  '新系统分别报告相机与行人分支；剩余轨迹仍含平移、视差、标注噪声。',
  '样本重叠、录制数只有6；录制bootstrap是探索性敏感性区间。未根据测试分数调参，不替换旧最佳模型。',
  '自动离线SAM/MPS参考标签并非独立人工逐帧GT；历史输入截断检验只验证预测网络没有读取未来数组。',
  '验证、独立几何复算和固定展示案例参见outputs/horizon_swap_20261017。']
 (dest/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
