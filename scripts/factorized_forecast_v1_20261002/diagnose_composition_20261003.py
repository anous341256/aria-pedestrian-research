"""Independent ray geometry and reference substitutions for the shown failure.

No training or model changes. Reference future information is diagnostic-only.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
from PIL import Image, ImageDraw
from draw_heads_20261003 import label, sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/factorized_forecast_v1_20261002/presentation/composition_diagnostic_20261003'
SOURCE=ROOT/'outputs/factorized_forecast_v1_20261002/presentation/head_outputs_20261003'


def project(points,rotations,K):
    rays=np.linalg.solve(K,np.c_[points,np.ones(len(points))].T).T
    rotated=np.einsum('tji,tj->ti',rotations,rays)
    pixels=(K@rotated.T).T
    assert (rotated[:,2]>0).all()
    return pixels[:,:2]/pixels[:,2:],rotated[:,2]


def error(pred,truth):
    values=np.linalg.norm(pred-truth,axis=1)
    return dict(ADE=float(values.mean()),FDE=float(values[-1]),per_step=values.tolist())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(SOURCE/'head_outputs.npz'))
    data=np.load(ROOT/'outputs/biflow_aria_v1_20261002/dataset.npz')
    meta=json.loads((ROOT/'outputs/biflow_aria_v1_20261002/samples.json').read_text(encoding='utf-8'))
    i=next(i for i,m in enumerate(meta) if m['key']=='10_N01_020')
    m=meta[i]
    raw=np.load(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/reference.npz')
    # Use original float64 world-from-camera rotations rather than deriving
    # reference rotations from the compensated points or the stored homographies.
    R=raw['R'][m['frames']].astype(float);K=data['K'][i].astype(float)
    Q=R[5].T @ R[6:]
    Qhat=Rotation.from_rotvec(a['camera_prediction'].astype(float)).as_matrix()
    truth=data['xy'][i,6:].astype(float)
    true_rays=np.linalg.solve(K,np.c_[truth,np.ones(15)].T).T
    fixed_rays=np.einsum('tij,tj->ti',Q,true_rays)
    fixed_pixels=(K@fixed_rays.T).T
    reference_residual=fixed_pixels[:,:2]/fixed_pixels[:,2:]
    results={}
    for name,ped,cam in [('pred_ped_pred_camera',a['future_pixel'],Qhat),
                          ('pred_ped_ref_camera',a['future_pixel'],Q),
                          ('ref_ped_pred_camera',reference_residual,Qhat),
                          ('ref_ped_ref_camera',reference_residual,Q)]:
        points,depth=project(ped,cam,K)
        results[name]=dict(points=points,depth=depth,metrics=error(points,truth))
    camera_only_pred=project(np.repeat(a['xy'][5][None],15,axis=0),Qhat,K)[0]
    camera_only_ref=project(np.repeat(a['xy'][5][None],15,axis=0),Q,K)[0]
    replay=float(np.max(abs(results['pred_ped_pred_camera']['points']-a['final_pixel'])))
    roundtrip=float(np.max(abs(results['ref_ped_ref_camera']['points']-truth)))
    assert replay<.001 and roundtrip<1e-7
    dataset_residual_difference=float(np.max(abs(reference_residual-a['stable_reference'][6:])))
    assert dataset_residual_difference<.002
    angle_error=np.degrees(Rotation.from_matrix(np.einsum('tji,tjk->tik',Qhat,Q)).magnitude())
    source_time=raw['times_ns'][m['frames']]
    time_error=float(np.max(abs((source_time-source_time[5])/1e9-data['seconds'][i])))
    assert time_error<1e-6
    record=dict(case=m['key'],coordinate_convention='R is world-from-camera; Q=R_t.T R_k is future-camera to t. Reprojection uses Q.T.',
                predicted_camera_rotvec_rad=a['camera_prediction'].tolist(),reference_camera_rotvec_rad=Rotation.from_matrix(Q).as_rotvec().tolist(),
                angular_error_deg=angle_error.tolist(),fx_px=float(K[0,0]),
                source_timestamp_max_difference_seconds=time_error,
                source_pose_float32_max_difference=float(np.max(abs(R-data['R'][i]))),
                independent_projection_vs_saved_max_px=replay,
                ref_both_roundtrip_max_px=roundtrip,
                independently_compensated_reference_vs_saved_max_px=dataset_residual_difference,
                results={k:dict(points=v['points'].tolist(),metrics=v['metrics'],min_ray_depth=float(v['depth'].min())) for k,v in results.items()},
                camera_only_reference_xy=camera_only_ref.tolist(),camera_only_predicted_xy=camera_only_pred.tolist(),
                reference_future_used='Only controlled diagnostic substitutions. No change to model output or headline result.',
                visualization_correction='A predicted-rotation warp of the current image shares the same prediction as the displayed point. Their alignment is tautological, not evidence of camera accuracy.',
                source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'head_outputs.npz',ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/reference.npz']})
    (OUT/'diagnosis.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'diagnosis_arrays.npz',truth=truth,reference_residual=reference_residual,Q=Q,Qhat=Qhat,
                        camera_only_pred=camera_only_pred,camera_only_ref=camera_only_ref,
                        **{k:v['points'] for k,v in results.items()})
    # Every panel uses the very same real future frame and crop.
    video=ROOT/f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,m['frames'][-1]);ok,bgr=cap.read();cap.release();assert ok
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    crop=(940,950,1740,1530);x1,y1,x2,y2=crop
    titles=[('pred_ped_pred_camera','A  原结果：预测行人 + 预测相机','#BB6E24'),
            ('pred_ped_ref_camera','B  只换相机：预测行人 + 参考旋转','#2E817A'),
            ('ref_ped_pred_camera','C  只留相机误差：参考行人 + 预测相机','#3F719A'),
            ('ref_ped_ref_camera','D  几何自检：参考行人 + 参考旋转','#647180')]
    canvas=Image.new('RGB',(1780,1670),'white');dc=ImageDraw.Draw(canvas)
    label(dc,(40,20),'最终合成的误差来源：同帧替换诊断',43,bold=True,maxw=1700)
    label(dc,(40,83),'10_N01_020  |  t + 1.5 s  |  四幅图均为同一真实未来帧、同一裁剪',27,fill='#647180')
    for n,(name,title,color) in enumerate(titles):
        x=35+(n%2)*875;y=145+(n//2)*735
        dc.rounded_rectangle((x,y,x+835,y+713),radius=12,outline=color,width=3)
        label(dc,(x+17,y+13),title,27,fill=color,bold=True,maxw=797)
        im=Image.fromarray(rgb[y1:y2,x1:x2].copy());di=ImageDraw.Draw(im)
        q=truth[-1]-np.array([x1,y1]);p=results[name]['points'][-1]-np.array([x1,y1])
        di.line([tuple(q),tuple(p)],fill='#1D2937',width=6);di.line([tuple(q),tuple(p)],fill='#FFD06B',width=3)
        px,py=p;di.ellipse((px-13,py-13,px+13,py+13),outline='#1D2937',width=7)
        di.ellipse((px-12,py-12,px+12,py+12),outline='#FFD06B',width=3)
        qx,qy=q
        for fill,width in [('#1D2937',7),('white',3)]:
            di.line([(qx-10,qy),(qx+10,qy)],fill=fill,width=width);di.line([(qx,qy-10),(qx,qy+10)],fill=fill,width=width)
        di.rounded_rectangle((12,530,788,569),radius=4,fill='#1D2937')
        endpoint=results[name]['points'][-1];e=results[name]['metrics']['FDE']
        label(di,(22,531),f'预测 ({endpoint[0]:.1f}, {endpoint[1]:.1f})  |  末端误差 {e:.1f} px',24,fill='white',maxw=755)
        canvas.paste(im,(x+17,y+59))
        ade=results[name]['metrics']['ADE']
        label(dc,(x+20,y+651),f'未来全部 15 点的 ADE：{ade:.2f} px',29,fill=color,bold=True)
    label(dc,(40,1620),'白 +：SAM2 参考中心；橙 ○：合成位置。参考未来量仅用于替换诊断，不作为实际预测输入。',23,fill='#647180',maxw=1700)
    canvas.save(OUT/'same_frame_substitution_zh.png')
    print(json.dumps({k:v['metrics'] for k,v in results.items()},ensure_ascii=False))
    print(json.dumps(dict(replay_max_px=replay,roundtrip_max_px=roundtrip,angular_error_last_deg=float(angle_error[-1]))))


if __name__=='__main__':main()
