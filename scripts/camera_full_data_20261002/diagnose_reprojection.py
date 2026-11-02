"""Same-window attribution of the frozen-person projection diagnostic."""
from common import *
import cv2


def project(points,rv,K):
    R=np.stack([cv2.Rodrigues(v.astype(float))[0] for v in rv])
    H=K@R.transpose(0,2,1)@np.linalg.inv(K)
    h=np.einsum('tij,tj->ti',H,np.column_stack([points,np.ones(15)]))
    assert (h[:,2]>1e-6).all()
    return h[:,:2]/h[:,2:]


def main():
    data=np.load(OLD/'dataset.npz');cameras=np.load(OUT/'person_camera_predictions.npz')
    lookup={int(i):j for j,i in enumerate(cameras['indices'])};meta=read(OLD/'samples.json')
    rows=[];reference=[];motion={'old_small':[],'full':[]};angles={'old_small':[],'full':[]};perfect={'old_small':[],'full':[]}
    for fold in read(OLD/'folds.json'):
        for s,seed in enumerate(read(OUT/'protocol.json')['seeds']):
            z=np.load(OLD/f'fold_{fold["fold"]}'/f'seed_{seed}/comp_dual/predictions_reviewed.npz')
            for j,i in enumerate(z['indices']):
                row=lookup[int(i)];K=data['K'][i];target=data['xy'][i,6:];points=z['component'][j]
                ref=project(points,data['cam'][i,6:],K);a=ref-target;reference.append(a)
                h=np.einsum('tij,tj->ti',data['H'][i,6:],np.column_stack([target,np.ones(15)]))
                exact_component=h[:,:2]/h[:,2:]
                record=dict(key=meta[i]['key'],recording=meta[i]['recording'],seed=seed)
                for name in motion:
                    rv=cameras[name][row,s];p=project(points,rv,K);b=p-ref;motion[name].append(b)
                    ang=degrees(rv,data['cam'][i,6:]);angles[name].append(ang)
                    pure=project(exact_component,rv,K)-target;perfect[name].append(pure)
                    record[name]=dict(ADE=float(np.linalg.norm(a+b,axis=1).mean()),camera_shift=float(np.linalg.norm(b,axis=1).mean()),
                                      angle=float(ang.mean()),camera_only_at_reference_person=float(np.linalg.norm(pure,axis=1).mean()))
                rows.append(record)
    a=np.array(reference);stats={}
    for name in motion:
        b=np.array(motion[name]);total=a+b
        amse=float(np.mean(np.sum(a*a,-1)));bmse=float(np.mean(np.sum(b*b,-1)));cross=float(2*np.mean(np.sum(a*b,-1)))
        tmse=float(np.mean(np.sum(total*total,-1)));assert abs(tmse-amse-bmse-cross)<1e-6
        stats[name]=dict(angle_degrees=float(np.mean(angles[name])),reference_camera_person_error_px=float(np.linalg.norm(a,axis=-1).mean()),
                         camera_induced_shift_px=float(np.linalg.norm(b,axis=-1).mean()),final_ADE=float(np.linalg.norm(total,axis=-1).mean()),
                         perfect_person_camera_error_px=float(np.linalg.norm(perfect[name],axis=-1).mean()),
                         MSE_person=amse,MSE_camera=bmse,MSE_cross=cross,MSE_total=tmse,
                         opposing_error_directions_fraction=float(np.mean(np.sum(a*b,axis=-1)<0)))
    groups={}
    for record in {r['recording'] for r in rows}:
        selected=[r for r in rows if r['recording']==record]
        groups[record]={name:{k:float(np.mean([r[name][k] for r in selected])) for k in selected[0][name]} for name in motion}
    point_angle_change=np.array(angles['full'])-np.array(angles['old_small'])
    point_pixel_change=np.linalg.norm(a+np.array(motion['full']),axis=-1)-np.linalg.norm(a+np.array(motion['old_small']),axis=-1)
    paired=dict(point_count=int(point_angle_change.size),angle_better_pixel_worse_points=int(np.sum((point_angle_change<0)&(point_pixel_change>0))))
    result=dict(scope='Same 117 windows, 3 seeds, fixed comp_dual pedestrian forecast; no model retraining. Future references used for diagnostic decomposition only.',
                identity='e_total = e_reference_camera_person + (projection_predicted_camera - projection_reference_camera); squared norm includes cross term',
                summary=stats,per_recording=groups,paired=paired,rows=rows)
    write(OUT/'reprojection_attribution.json',result)
    old,new=stats['old_small'],stats['full']
    report=f'''# 相机预测改善而人物预测变差的原因核查

对同一批 117 个窗口、三个种子、固定的补偿双分支人物预测进行拆解。未来参考旋转仅用于诊断，不进入正式预测。

| 指标 | 旧相机 | 新相机 |
|---|---:|---:|
| 相机平均角误差，度 | {old['angle_degrees']:.3f} | {new['angle_degrees']:.3f} |
| 相机预测引入的图像偏移，像素 | {old['camera_induced_shift_px']:.2f} | {new['camera_induced_shift_px']:.2f} |
| 使用参考未来旋转后剩余的人物预测误差，像素 | {old['reference_camera_person_error_px']:.2f} | {new['reference_camera_person_error_px']:.2f} |
| 最终人物 ADE，像素 | {old['final_ADE']:.2f} | {new['final_ADE']:.2f} |

相机偏移按“同一个人物预测点分别使用预测旋转与参考旋转重投影，两次结果之差”计算。误差向量满足 e = a + b，其中 a 是参考旋转下剩余的预测误差，b 是相机旋转预测引入的偏移。因此平方误差满足 ||e||² = ||a||² + ||b||² + 2a·b，不能将表中的平均像素距离直接相加。

本轮 a 不变。相机偏移的均方项从 {old['MSE_camera']:.2f} 降到 {new['MSE_camera']:.2f} 像素平方，交叉项却从 {old['MSE_cross']:.2f} 变成 {new['MSE_cross']:.2f} 像素平方。负交叉项意味着误差存在方向抵消；抵消减弱超过了相机项的改善，最终均方误差上升。两误差方向相反的预测点比例也从 {old['opposing_error_directions_fraction']*100:.1f}% 降到 {new['opposing_error_directions_fraction']*100:.1f}%。这是本批固定输出的数值分解，不证明训练主动学会了利用错误抵消。

全录制 9,231 窗口上的 8.6% 收益与此处 117 个选定人物窗口并非同一评估集。后者相机误差仅从 7.97° 降到 7.65°。其中 164557 录制的人物窗口角误差反而从约 6.65° 增到 8.58°；全录制平均变好不代表每个选定子集都改善。

还有一项实现上的限定：本次只替换最终重投影旋转。旧人物网络的输入包含旧相机预测，其输出没有随新相机条件重新计算。新相机与人物网络的完整配合尚未验证；不能将这个诊断当作同步替换条件并重训后的结果。当前证据足以说明误差方向组合发生变化，尚不能唯一判定训练耦合、条件不一致或特定场景各自的因果贡献。

后续可先做冻结人物网络的两处同步替换诊断，再决定是否用新的相机条件训练人物网络。上述诊断没有修改任何模型或筛选样本。

[逐窗口数据](E:/New_research/outputs/camera_full_data_20261002/reprojection_attribution.json) · [主实验报告](E:/New_research/reports/camera_full_data_20261002/RESULTS.md)
'''
    path=ROOT/'reports/camera_full_data_20261002/REPROJECTION_DIAGNOSIS.md'
    path.write_text(report,encoding='utf8')
    print(json.dumps(dict(summary=stats,per_recording=groups,paired=paired),ensure_ascii=False))


if __name__=='__main__':main()
