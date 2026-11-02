"""Read out the CAMERA and PEDESTRIAN forecast branches separately.

Camera-only display: transform the current image and current target point by
the predicted rotation, fixing the residual pedestrian coordinate.
Pedestrian-only display: use frame t as a static coordinate canvas for the
predicted residual point. No future human appearance is generated.
Composition: show the actual predicted point in the corresponding future frame.
"""
from pathlib import Path
import json, math
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import draw_heads_20261003 as previous
from draw_architecture import Architecture, COLORS

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'outputs/factorized_forecast_v1_20261002/presentation/camera_pedestrian_outputs_20261003'
SOURCE = previous.DEST
CASE = previous.CASE
INK, MUTED, WHITE = previous.INK, previous.MUTED, '#FFFFFF'
label = previous.label
STAGES = ['camera','pedestrian','composed']
COLORS_OUT = {'camera':('#3F719A','#63C9FF'), 'pedestrian':('#2E817A','#5AE8BF'), 'composed':('#9A7132','#FFD06B')}
# All panels share the same crop and scale, in the same calibrated image grid.
CROP = (940,950,1790,1570)
SIZE = (850,620)
TITLES = {
 'zh': {'camera':'①  相机运动预测头：仅施加预测旋转', 'pedestrian':'②  行人运动预测头：固定相机朝向', 'composed':'③  两路输出合成：最终未来轨迹'},
 'en': {'camera':'1  Camera forecast: predicted rotation only', 'pedestrian':'2  Pedestrian forecast: camera orientation fixed', 'composed':'3  Compose both branches: final future trajectory'}}
DESC = {
 'zh': {'camera':'输出 15 × 3 旋转向量；固定当前行人残差位置，显示相机造成的位移。',
        'pedestrian':'输出 15 × 2 残差位置；以当前帧为背景，显示相机合成前的位移。',
        'composed':'将行人头预测的位置按相机头预测的旋转重投影，得到 15 × 2 图像坐标。'},
 'en': {'camera':'15 × 3 rotation vectors. Hold the current residual point fixed to show the camera effect.',
        'pedestrian':'15 × 2 residual positions. Frame t is a fixed canvas showing motion before camera composition.',
        'composed':'Reproject predicted pedestrian positions with predicted camera rotations into 15 × 2 future image coordinates.'}}
LEGENDS = {
 'zh': {'camera':'白 +：当前位置    蓝 ○：仅相机作用后的位置    箭头：相机造成的位移',
        'pedestrian':'白 +：当前位置    绿 ○：行人头预测位置    左上角：局部放大 3 倍',
        'composed':'白 +：未来 SAM2 标注中心    橙 ○：最终预测位置    连线：预测误差'},
 'en': {'camera':'White +: current position    Blue circle: camera-only position    Arrow: camera-induced displacement',
        'pedestrian':'White +: current position    Green circle: pedestrian prediction    Top left: 3× magnified inset',
        'composed':'White +: future SAM2 reference    Orange circle: composed prediction    Line: prediction error'}}
NOTES = {
 'zh': {'camera':'背景为当前帧经预测旋转变换的效果示意，不是真实未来画面。当前相机分支仅预测旋转。',
        'pedestrian':'背景始终是当前帧；模型只输出轨迹点，不生成未来外观。残差中仍有平移、视差等影响。',
        'composed':'背景为对应的真实未来帧，仅用于评价；最终合成只使用预测旋转，不使用参考未来旋转。'},
 'en': {'camera':'Current-frame rotation warp illustrates the camera output; it is not future video. This camera model predicts rotation only.',
        'pedestrian':'The model predicts points, not future appearance. Residual motion still includes translation and parallax.',
        'composed':'Actual future video is evaluation-only. Composition uses predicted rotations, not reference future rotations.'}}


def warp(p,H):
    q=np.asarray(H) @ np.r_[p,1.]
    return q[:2]/q[2]


def prepare():
    old=previous.load()
    a,frames=old[0],old[4]
    K=a['K'];p0=a['xy'][5]
    matrices=np.stack([K @ cv2.Rodrigues(v)[0].T @ np.linalg.inv(K) for v in a['camera_prediction']])
    camera_points=np.stack([warp(p0,H) for H in matrices])
    final=np.stack([warp(p,H) for p,H in zip(a['future_pixel'],matrices)])
    error=float(np.max(np.abs(final-a['final_pixel'])))
    assert error < .001
    arrays=dict(camera_rotation_vector_rad=a['camera_prediction'], camera_projection=matrices,
                camera_only_xy=camera_points, pedestrian_only_xy=a['future_pixel'],
                composed_xy=final, origin=p0, future_reference_xy=a['xy'][6:], seconds=a['seconds'][6:])
    np.savez_compressed(DEST/'branch_outputs.npz',**arrays)
    (DEST/'branch_outputs.json').write_text(json.dumps({k:v.tolist() for k,v in arrays.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    state=dict(arrays=arrays,frames=frames,sources=old[5],composition_error=error)
    return state


def markers(im,base,point,color,arrow=True):
    d=ImageDraw.Draw(im)
    d.line([tuple(base),tuple(point)],fill='#1E2D37',width=7)
    d.line([tuple(base),tuple(point)],fill=color,width=3)
    v=point-base
    length=np.linalg.norm(v)
    if arrow and length>26:
        u=v/length;n=np.array([-u[1],u[0]])
        tip=point-u*13
        d.polygon([tuple(tip),tuple(tip-u*14+n*6),tuple(tip-u*14-n*6)],fill=color)
    x,y=base
    for fill,width in [('#1E2D37',7),(WHITE,3)]:
        d.line([(x-10,y),(x+10,y)],fill=fill,width=width)
        d.line([(x,y-10),(x,y+10)],fill=fill,width=width)
    x,y=point
    d.ellipse((x-12,y-12,x+12,y+12),outline='#17222D',width=6)
    d.ellipse((x-11,y-11,x+11,y+11),outline=color,width=3)


def photo(stage,j,lang,state):
    a=state['arrays'];rgb=state['frames'][5]
    if stage=='camera':
        rgb=cv2.warpPerspective(rgb,a['camera_projection'][j],(2560,1920),flags=cv2.INTER_LINEAR)
        point=a['camera_only_xy'][j];base=a['origin']
    elif stage=='pedestrian':
        point=a['pedestrian_only_xy'][j];base=a['origin']
    else:
        rgb=state['frames'][j+6];point=a['composed_xy'][j];base=a['future_reference_xy'][j]
    x1,y1,x2,y2=CROP
    im=Image.fromarray(rgb[y1:y2,x1:x2].copy())
    p,q=point-np.array([x1,y1]),base-np.array([x1,y1])
    assert min(p.min(),q.min())>16 and (p<np.array(SIZE)-16).all() and (q<np.array(SIZE)-16).all()
    markers(im,q,p,COLORS_OUT[stage][1],stage!='composed')
    if stage=='pedestrian':
        # Same fixed source patch in every frame; real 3x raster magnification.
        mid=a['origin']-np.array([x1,y1])+np.array([0,10])
        zx,zy=np.round(mid-43).astype(int)
        zoom=im.crop((zx,zy,zx+86,zy+86)).resize((258,258),Image.Resampling.LANCZOS)
        im.paste(zoom,(25,83))
        dz=ImageDraw.Draw(im)
        dz.rectangle((23,81,285,343),outline=COLORS_OUT[stage][1],width=3)
        dz.rectangle((24,344,286,379),fill='#1D2A38')
        dz.rectangle((zx,zy,zx+86,zy+86),outline=COLORS_OUT[stage][1],width=2)
        dz.line([(285,212),(zx,zy+43)],fill=COLORS_OUT[stage][1],width=2)
        label(dz,(38,345),'局部 ×3' if lang=='zh' else 'Detail ×3',24,fill=WHITE,lang=lang)
    d=ImageDraw.Draw(im)
    d.rounded_rectangle((14,14,201,61),radius=7,fill='#1D2A38')
    label(d,(24,18),f't +{a["seconds"][j]:.1f} s',29,fill=WHITE,lang='en')
    d.rounded_rectangle((14,523,SIZE[0]-14,606),radius=5,fill='#1D2A38')
    displacement=float(np.linalg.norm(point-base))
    if stage=='camera':
        rv=np.degrees(a['camera_rotation_vector_rad'][j])
        first=(f'旋转向量 (°)：({rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f})' if lang=='zh' else
               f'Rotation vector (deg): ({rv[0]:.2f}, {rv[1]:.2f}, {rv[2]:.2f})')
        second=(f'相机单独造成位移 {displacement:.1f} px' if lang=='zh' else f'Camera-only displacement: {displacement:.1f} px')
    else:
        first=(f'输出位置：({point[0]:.1f}, {point[1]:.1f}) px' if lang=='zh' else f'Output position: ({point[0]:.1f}, {point[1]:.1f}) px')
        if stage=='pedestrian':
            second=f'行人残差位移 {displacement:.1f} px' if lang=='zh' else f'Residual displacement: {displacement:.1f} px'
        else:
            second=f'最终预测误差 {displacement:.1f} px' if lang=='zh' else f'Final prediction error: {displacement:.1f} px'
    label(d,(26,528),first,25,WHITE,lang=lang,maxw=SIZE[0]-53)
    label(d,(26,566),second,25,WHITE,lang=lang,maxw=SIZE[0]-53)
    return im


def card(stage,lang,indices,state,full=False):
    cols=5 if full else 3;rows=math.ceil(len(indices)/cols)
    margin,gap=35,25
    w=2*margin+cols*SIZE[0]+(cols-1)*gap
    h=170+rows*SIZE[1]+(rows-1)*28+142
    im=Image.new('RGB',(w,h),WHITE);d=ImageDraw.Draw(im)
    d.rounded_rectangle((2,2,w-3,h-3),radius=16,outline=COLORS_OUT[stage][0],width=3)
    label(d,(margin,18),TITLES[lang][stage],48,bold=True,fill=COLORS_OUT[stage][0],lang=lang,maxw=w-2*margin)
    label(d,(margin,92),DESC[lang][stage],32,fill=MUTED,lang=lang,maxw=w-2*margin)
    for n,j in enumerate(indices):im.paste(photo(stage,j,lang,state),(margin+n%cols*(SIZE[0]+gap),160+n//cols*(SIZE[1]+28)))
    bottom=160+rows*SIZE[1]+(rows-1)*28
    label(d,(margin,bottom+19),LEGENDS[lang][stage],30,lang=lang,maxw=w-2*margin)
    label(d,(margin,bottom+70),NOTES[lang][stage],28,fill=MUTED,lang=lang,maxw=w-2*margin)
    return im


def architecture(lang,cards,state):
    cn=lang=='zh';d=Architecture(lang);fig=d.fig
    fig.set_size_inches(31,18)
    d.ax.set_position([.002,.430,.581,.555]);d.draw()
    for x,y,num,kind in [(145,95,'1','camera'),(145,63,'2','person'),(218,64,'3','geometry')]:
        d.ax.add_patch(Circle((x,y),1.85,facecolor=COLORS[kind],edgecolor=WHITE,lw=.9,zorder=12))
        d.text(x,y,num,size=12,bold=True,color=WHITE).set_zorder(13)
    def text(x,y,s,size=15,color=INK,bold=False):
        return fig.text(x,y,s,fontproperties=d.bold if bold else d.body,fontsize=size,
                        color=color,va='center',linespacing=1.55)
    text(.600,.978,'相机预测 / 行人预测 / 最终合成' if cn else 'Camera / pedestrian / composed output',19,bold=True)
    text(.600,.950,CASE+'  |  fold 0 / seed 17',12,MUTED)
    # Three equally scaled strips; all use the same crop and pixel grid.
    for stage,y in [('camera',.664),('pedestrian',.360),('composed',.056)]:
        ax=fig.add_axes([.593,y,.399,.262]);ax.imshow(cards[stage]);ax.axis('off')
    text(.036,.399,'分开查看两种预测的作用' if cn else 'Isolate each forecast branch',18,bold=True)
    ax=fig.add_axes([.033,.084,.29,.286]);ax.imshow(state['frames'][5])
    ax.set_xlim(850,2180);ax.set_ylim(1610,750);ax.axis('off')
    p0=state['arrays']['origin'];ax.scatter(*p0,s=110,facecolors='none',edgecolors='#FFD06B',linewidths=2.2)
    ax.annotate('N01',p0,xytext=(-35,45),textcoords='offset points',color=WHITE,fontsize=12,
                bbox=dict(boxstyle='round,pad=.2',fc='#17222D',alpha=.8),arrowprops=dict(arrowstyle='->',color=WHITE,lw=1.5))
    text(.036,.063,'t 帧与目标 N01；两路输出来自同一次预测。' if cn else 'Frame t and target N01; both outputs from one forecast.',12,MUTED)
    text(.345,.370,'① 固定行人位置，看相机' if cn else '1  Fix the pedestrian; inspect camera',16,COLORS['camera'],True)
    text(.345,.324,('把当前画面按预测旋转变换，\n显示相机单独引起的位移。' if cn else
                     'Warp frame t by predicted rotations\nto visualize the camera-only effect.'),12.5,MUTED)
    text(.345,.258,'② 固定相机朝向，看行人' if cn else '2  Fix orientation; inspect pedestrian',16,COLORS['person'],True)
    text(.345,.214,('在当前画面坐标下显示行人预测，\n放大局部，保留真实的位移尺度。' if cn else
                     'Show the residual prediction at t,\nwith a 3x inset for small displacements.'),12.5,MUTED)
    text(.345,.146,'③ 两路输出经重投影合成' if cn else '3  Compose through reprojection',16,COLORS['geometry'],True)
    text(.345,.102,'在真实未来帧上检查最终预测位置。' if cn else 'Check the final point on actual future video.',12.5,MUTED)
    text(.036,.025,('这里的两个预测头指相机分支与行人分支。历史重建仍是辅助训练模块，不作为本图的两个展示头。' if cn else
                   'The two displayed forecast heads are the camera and pedestrian branches. History reconstruction remains an auxiliary training module.'),11,MUTED)
    d.validate()
    for suffix in ['png','pdf','svg']:fig.savefig(DEST/f'architecture_camera_pedestrian_{lang}.{suffix}',dpi=200,facecolor=WHITE)
    plt.close(fig)
    preview=Image.open(DEST/f'architecture_camera_pedestrian_{lang}.png');preview.thumbnail((3100,1800))
    preview.save(DEST/f'architecture_camera_pedestrian_{lang}_preview.png')


def main():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'frames').mkdir(exist_ok=True)
    state=prepare();selection=[4,9,14]
    for lang in ['zh','en']:
        cards={}
        for stage in STAGES:
            cards[stage]=card(stage,lang,selection,state)
            cards[stage].save(DEST/f'{stage}_selected_{lang}.png')
            card(stage,lang,list(range(15)),state,full=True).save(DEST/f'{stage}_all_steps_{lang}.png')
            if lang=='zh':
                for j in range(15):photo(stage,j,lang,state).save(DEST/f'frames/{stage}_{j+1:02d}.png')
        overview=Image.new('RGB',(cards['camera'].width,sum(im.height for im in cards.values())+214),WHITE)
        ds=ImageDraw.Draw(overview)
        title='相机运动预测 + 行人运动预测 → 最终合成' if lang=='zh' else 'Camera forecast + pedestrian forecast → composition'
        label(ds,(35,16),title,49,bold=True,lang=lang,maxw=overview.width-70)
        label(ds,(35,91),CASE+'  |  '+('同一窗口、同一时间、相同裁剪与显示比例' if lang=='zh' else 'Same window, same time, same crop and scale'),30,fill=MUTED,lang=lang)
        y=155
        for stage in STAGES:overview.paste(cards[stage],(0,y));y+=cards[stage].height+17
        overview.save(DEST/f'camera_pedestrian_overview_{lang}.png')
        architecture(lang,cards,state)
    arrays=state['arrays']
    record=dict(case=CASE,passed=True,head_semantics=['camera motion prediction','pedestrian residual prediction'],
                current_camera_model='rotation only',camera_output_shape=[15,3],pedestrian_output_shape=[15,2],
                final_output_shape=[15,2],camera_only_rule='pi(K Qhat_k.T K^-1 p_t)',
                pedestrian_only_rule='predicted residual point in the orientation of t',
                composed_rule='pi(K Qhat_k.T K^-1 predicted_residual_k)',
                camera_background='current frame warped by predicted rotation; illustration, not generated or true future video',
                pedestrian_background='unchanged current frame t, same crop each time',
                composed_background='actual future frame, evaluation only',
                future_reference_rotation_used=False,
                composition_max_difference_from_saved_px=state['composition_error'],
                crop_xyxy=CROP,selected_horizon_indices=selection,sources=state['sources'],
                source_outputs_sha256=previous.sha(SOURCE/'head_outputs.npz'),
                identity_of_model_outputs='Reused the exact saved/replayed same-window outputs; no retraining.')
    (DEST/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(destination=str(DEST),composition_max_difference_px=state['composition_error'])))


if __name__=='__main__':main()
