"""Real frame-based decoder readouts. No generated imagery or retraining.

Historical and future head outputs live in the orientation of frame t. Their
display backgrounds are therefore rotation-aligned to t, not raw future frames.
Reference future RGB and rotations are used ONLY in this rendering program.
"""
from pathlib import Path
import json, gzip, hashlib, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from draw_architecture import Architecture, COLORS

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'outputs/factorized_forecast_v1_20261002/presentation/head_outputs_20261003'
CASE = '10_N01_020'
FRAME0 = 15
INK = '#263443'
MUTED = '#647180'
WHITE = '#FFFFFF'
STYLES = {'history': ('#826D94', '#D6ADFF'),
          'future': ('#2E817A', '#FFD06B'), 'final': ('#9A7132', '#FFD06B')}
# Constant within each coordinate system, across all frames. No per-frame zoom.
CROPS = {'history': (1230, 990, 1810, 1490),
         'future': (1230, 990, 1810, 1490),
         'final': (970, 990, 1550, 1490)}
PHOTO_SIZE = (580, 500)
TITLES = {
    'zh': {'history': '①  历史重建头的输出', 'future': '②  未来预测头的输出', 'final': '③  相机与行人合成后的输出'},
    'en': {'history': '1  History-head output', 'future': '2  Future-head output', 'final': '3  Composed image-space output'}}
DESC = {
    'zh': {'history': '6 个历史点 · 已恢复像素尺度 · 统一到 t 时刻朝向',
           'future': '15 个未来点 · 相机合成前 · 统一到 t 时刻朝向',
           'final': '未来头输出 + 预测的相机旋转 → 未来图像坐标'},
    'en': {'history': '6 reconstructed points | pixel units | orientation of frame t',
           'future': '15 future points | before composition | orientation of frame t',
           'final': 'Future-head output + predicted camera rotation → future image pixels'}}
NOTES = {
    'zh': {'history': '背景按历史旋转对齐；重建头用于辅助训练，不接入最终合成。',
           'future': '背景按参考未来旋转对齐，仅便于观察头输出；不进入推理。',
           'final': '真实未来画面仅用于核对；此处使用预测旋转完成合成。'},
    'en': {'history': 'Historical frames aligned for display. This auxiliary head is not composed into the forecast.',
           'future': 'Reference future rotation aligns the display background only; it is not an inference input.',
           'final': 'Actual future frames are shown for evaluation only. Composition uses predicted rotations.'}}


def font(size=28, bold=False, lang='zh'):
    name = ('msyhbd.ttc' if bold else 'msyh.ttc') if lang == 'zh' else ('arialbd.ttf' if bold else 'arial.ttf')
    return ImageFont.truetype('C:/Windows/Fonts/' + name, size)


def label(draw, xy, text, size=28, fill=INK, bold=False, lang='zh', maxw=None):
    f = font(size, bold, lang)
    if maxw is not None:
        while draw.textbbox((0, 0), text, font=f)[2] > maxw and size > 15:
            size -= 1
            f = font(size, bold, lang)
    draw.text(xy, text, font=f, fill=fill)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load():
    a = dict(np.load(DEST / 'head_outputs.npz'))
    record = json.loads((DEST / 'replay_verification.json').read_text(encoding='utf-8'))
    m = record['metadata']
    video = ROOT / f'outputs/sam2_g1_v6_v8_20261001/{m["clip"]}/input/rgb.mp4'
    mask_path = ROOT / f'outputs/sam2_near_people_20261002/{m["clip"]}/final/masks_rle.jsonl.gz'
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME0)
    frames = []
    for n in range(21):
        ok, bgr = cap.read()
        assert ok
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    masks = {}
    with gzip.open(mask_path, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['id'] != m['id'] or row['frame'] not in m['frames']:
                continue
            x1, y1, x2, y2 = map(int, row['rle'][-4:])
            counts = np.asarray(row['rle'][:-4], np.int64)
            bits = np.repeat(np.arange(len(counts)) % 2, counts).reshape(y2-y1+1, x2-x1+1)
            mask = np.zeros((1920, 2560), np.uint8)
            mask[y1:y2+1, x1:x2+1] = bits
            masks[row['frame']-FRAME0] = mask
    assert len(masks) == 21
    cache = {}
    for stage in STYLES:
        offsets = list(range(6)) if stage == 'history' else list(range(6,21))
        for j, offset in enumerate(offsets):
            rgb = frames[offset]
            mask = masks[offset]
            if stage != 'final':
                H = a['H_reference'][offset]
                # No image is fed back to any model. This is an evaluation canvas.
                rgb = cv2.warpPerspective(rgb, H, (2560,1920), flags=cv2.INTER_LINEAR)
                mask = cv2.warpPerspective(mask, H, (2560,1920), flags=cv2.INTER_NEAREST)
            rgb = rgb.copy()
            selected = mask.astype(bool)
            rgb[selected] = (rgb[selected] * .82 + np.array([47,178,153]) * .18).astype(np.uint8)
            x1,y1,x2,y2 = CROPS[stage]
            cache[(stage,j)] = Image.fromarray(rgb[y1:y2,x1:x2])
    reference = dict(history=a['stable_reference'][:6], future=a['stable_reference'][6:], final=a['xy'][6:])
    output = dict(history=a['history_pixel'], future=a['future_pixel'], final=a['final_pixel'])
    assert all(np.isfinite(v).all() for v in output.values())
    sources = {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [video,mask_path,DEST/'head_outputs.npz']}
    return a, cache, reference, output, frames, sources


def photo(stage, j, lang, state):
    a, cache, ref, out, frames, sources = state
    im = cache[(stage,j)].copy()
    d = ImageDraw.Draw(im)
    color = STYLES[stage][1]
    shift = np.array(CROPS[stage][:2])
    p = out[stage][j] - shift
    q = ref[stage][j] - shift
    assert (p > 15).all() and (q > 15).all() and (p < np.array(PHOTO_SIZE)-15).all() and (q < np.array(PHOTO_SIZE)-15).all()
    # Error connectors and distinct outlined symbols preserve small displacements.
    d.line([tuple(q),tuple(p)], fill='#202630', width=5)
    d.line([tuple(q),tuple(p)], fill='#E0E5EC', width=2)
    x,y = q
    for width,fill in [(6,'#17222D'),(3,WHITE)]:
        d.line([(x-9,y),(x+9,y)],fill=fill,width=width)
        d.line([(x,y-9),(x,y+9)],fill=fill,width=width)
    x,y = p
    d.ellipse([x-11,y-11,x+11,y+11], outline='#17222D', width=6)
    d.ellipse([x-10,y-10,x+10,y+10], outline=color, width=3)
    t = float(a['seconds'][j if stage == 'history' else j+6])
    time_text = f't {t:+.1f} s' if abs(t) > .01 else 't = 0.0 s'
    d.rounded_rectangle((12,12,202,57), radius=7, fill='#1D2A38')
    label(d,(23,16),time_text,size=27,fill=WHITE,lang='en')
    error = float(np.linalg.norm(out[stage][j]-ref[stage][j]))
    d.rounded_rectangle((12,451,PHOTO_SIZE[0]-12,489),radius=5,fill='#1D2A38')
    p0 = out[stage][j]
    s = (f'输出 ({p0[0]:.1f}, {p0[1]:.1f})  |  误差 {error:.1f} px' if lang == 'zh' else
         f'Output ({p0[0]:.1f}, {p0[1]:.1f}) | error {error:.1f} px')
    label(d,(22,453),s,size=21,fill=WHITE,lang=lang,maxw=PHOTO_SIZE[0]-45)
    return im


def make_card(stage, lang, indices, state, all_steps=False):
    cols = (3 if stage == 'history' else 5) if all_steps else 3
    rows = math.ceil(len(indices)/cols)
    gap, margin = 25, 35
    w = 2*margin + cols*580 + (cols-1)*gap
    h = 145 + rows*500 + (rows-1)*28 + 140
    im = Image.new('RGB',(w,h),WHITE)
    d = ImageDraw.Draw(im)
    color = STYLES[stage][0]
    d.rounded_rectangle((2,2,w-3,h-3),radius=16,outline=color,width=3)
    label(d,(margin,17),TITLES[lang][stage],size=40,bold=True,fill=color,lang=lang)
    label(d,(margin,77),DESC[lang][stage],size=29,fill=MUTED,lang=lang,maxw=w-2*margin)
    for n,j in enumerate(indices):
        tile = photo(stage,j,lang,state)
        im.paste(tile,(margin+(n%cols)*(580+gap),135+(n//cols)*528))
    last = 135+rows*500+(rows-1)*28
    legend = ('白色 +：SAM2 标注中心     彩色 ○：模型输出     绿色：目标 mask' if lang == 'zh' else
              'White +: SAM2 annotation center     Colored circle: model output     Green: target mask')
    label(d,(margin,last+20),legend,size=27,lang=lang,maxw=w-2*margin)
    label(d,(margin,last+67),NOTES[lang][stage],size=25,fill=MUTED,lang=lang,maxw=w-2*margin)
    return im


def overview(lang,cards,state):
    w = cards['history'].width
    h = sum(im.height for im in cards.values())+200
    im = Image.new('RGB',(w,h),WHITE)
    d = ImageDraw.Draw(im)
    title = '两个头的实际输出 → 最终几何合成' if lang == 'zh' else 'Two decoder readouts → final geometric composition'
    label(d,(32,15),title,40,bold=True,lang=lang,maxw=w-64)
    label(d,(32,77),CASE+'  |  fold 0 / seed 17  |  '+('同一已保存测试窗口' if lang=='zh' else 'Same saved test window'),26,fill=MUTED,lang=lang)
    y = 137
    for key in ['history','future','final']:
        im.paste(cards[key],(0,y));y+=cards[key].height+15
    im.save(DEST/f'head_outputs_overview_{lang}.png')
    return im


def architecture(lang,cards,state):
    cn = lang == 'zh'
    d = Architecture(lang)
    fig = d.fig
    fig.set_size_inches(31,18)
    d.ax.set_position([.002,.430,.581,.555])
    d.draw()
    # Exact readout points in the original implemented architecture.
    for x,y,num,kind in [(105,24,'1','training'),(145,63,'2','person'),(218,64,'3','geometry')]:
        d.ax.add_patch(Circle((x,y),1.85,facecolor=COLORS[kind],edgecolor='white',lw=.9,zorder=12))
        d.text(x,y,num,size=12,bold=True,color='white').set_zorder(13)
    font_body, font_bold = d.body, d.bold
    extra = []
    def text(x,y,s,size=15,color=INK,bold=False,ha='left'):
        t = fig.text(x,y,s,fontproperties=font_bold if bold else font_body,fontsize=size,
                     color=color,ha=ha,va='center',linespacing=1.55)
        extra.append(t)
        return t
    text(.601,.976,'按输出张量逐步查看真实结果' if cn else 'Actual readouts, shown on video frames',18,bold=True)
    text(.601,.950,CASE+'  |  fold 0 / seed 17',12,color=MUTED)
    for stage,y in [('history',.642),('future',.337),('final',.032)]:
        ax = fig.add_axes([.593,y,.399,.292])
        ax.imshow(cards[stage]);ax.axis('off')
    # One overview frame retains scene context; all enlarged crops use exact
    # original-image pixel coordinates rather than a person-following crop.
    rgb = state[4][5]
    ax = fig.add_axes([.033,.084,.29,.286])
    ax.imshow(rgb);ax.set_xlim(850,2180);ax.set_ylim(1610,750);ax.axis('off')
    p = state[0]['xy'][5]
    ax.scatter(*p,s=110,facecolors='none',edgecolors='#FFD06B',linewidths=2.2)
    ax.annotate('N01',p,xytext=(-35,45),textcoords='offset points',color=WHITE,fontsize=12,
                bbox=dict(boxstyle='round,pad=.2',fc='#17222D',alpha=.8),
                arrowprops=dict(arrowstyle='->',color=WHITE,lw=1.5))
    text(.036,.399,'同一对象 · 同一测试窗口' if cn else 'One pedestrian, one held-out window',18,bold=True)
    text(.036,.063,'t 帧全景与目标 N01（黑上衣、粉裤）' if cn else 'Frame t; target N01 in black top and pink trousers',12,color=MUTED)
    text(.346,.372,'① 重建过去 6 点' if cn else '1  Reconstruct six past points',17,COLORS['training'],True)
    text(.346,.337,'辅助约束编码器；不参与最终合成。' if cn else 'Auxiliary training; not part of composition.',12.5,MUTED)
    text(.346,.282,'② 预测未来 15 点' if cn else '2  Predict 15 future points',17,COLORS['person'],True)
    text(.346,.247,'输出位于 t 时刻朝向的参考图像平面。' if cn else 'Output in the image plane oriented at t.',12.5,MUTED)
    text(.346,.192,'③ 加入未来相机旋转' if cn else '3  Compose with future camera rotation',17,COLORS['geometry'],True)
    text(.346,.149,('用预测旋转重投影到各未来帧，\n得到最终 15 个图像坐标。' if cn else
                     'Reproject using predicted rotations\nto obtain 15 future image coordinates.'),12.5,MUTED)
    text(.036,.025,('彩色圈均为真实网络输出；白色十字为 SAM2 参考标注。未来画面及其参考旋转仅用于展示与核对。' if cn else
                   'Colored circles are actual model outputs; white crosses are SAM2 references. Future video / reference rotations are display-only.'),11,MUTED)
    d.validate()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for artist in extra:
        bb = artist.get_window_extent(renderer)
        assert bb.x0 >= 0 and bb.x1 <= canvas.x1 and bb.y0 >= 0 and bb.y1 <= canvas.y1, artist.get_text()
    for suffix in ['png','pdf','svg']:
        fig.savefig(DEST/f'architecture_head_outputs_{lang}.{suffix}',dpi=200,facecolor='white')
    plt.close(fig)
    preview = Image.open(DEST/f'architecture_head_outputs_{lang}.png')
    preview.thumbnail((3100,1800))
    preview.save(DEST/f'architecture_head_outputs_{lang}_preview.png')


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    state = load()
    (DEST/'frames').mkdir(exist_ok=True)
    selection = {'history':[0,3,5],'future':[4,9,14],'final':[4,9,14]}
    for lang in ['zh','en']:
        cards = {}
        for stage in ['history','future','final']:
            count = 6 if stage == 'history' else 15
            cards[stage] = make_card(stage,lang,selection[stage],state)
            cards[stage].save(DEST/f'{stage}_selected_{lang}.png')
            make_card(stage,lang,list(range(count)),state,all_steps=True).save(DEST/f'{stage}_all_steps_{lang}.png')
            if lang == 'zh':
                for j in range(count):
                    photo(stage,j,lang,state).save(DEST/f'frames/{stage}_{j+1:02d}.png')
        overview(lang,cards,state)
        architecture(lang,cards,state)
    a = state[0]
    verification = dict(passed=True, case=CASE, sources=state[5], selection=selection,
        past_frames=6, future_frames=15, crops_xyxy=CROPS,
        display_coordinate_rule='History and future heads: frames warped by reference H into t orientation. Final: actual future frames.',
        future_reference_used_for_inference=False,
        future_reference_used_for_display=True,
        history_head_enters_composition=False,
        sampling='Exact replay; mean of five fixed starts, ten steps, without retraining.',
        errors=dict(history_mean_px=float(np.linalg.norm(a['history_pixel']-a['stable_reference'][:6],axis=1).mean()),
                    future_component_ADE=float(np.linalg.norm(a['future_pixel']-a['stable_reference'][6:],axis=1).mean()),
                    final_ADE=float(np.linalg.norm(a['final_pixel']-a['xy'][6:],axis=1).mean())))
    (DEST/'render_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification['errors']))
    print(DEST)


if __name__ == '__main__':
    main()
