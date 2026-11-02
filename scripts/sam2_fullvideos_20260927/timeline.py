"""Plot mask presence as an audit aid; this is not an accuracy chart."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/sam2_fullvideos_20260927'
IDS=['162538','164557','151153','193709','201129']
NAMES=['食品卖场','白天街道至商场','站台至站厅','傍晚过街','夜间进站']
COLORS=['#D8AA00','#2997D8','#BD50CD','#319547','#DC772C']

def main():
    fontpath='/mnt/c/Windows/Fonts/msyh.ttc' if str(ROOT).startswith('/mnt/') else 'C:/Windows/Fonts/msyh.ttc'
    title=ImageFont.truetype(fontpath,29);font=ImageFont.truetype(fontpath,21);small=ImageFont.truetype(fontpath,17)
    img=Image.new('RGB',(1280,1450),'white');d=ImageDraw.Draw(img)
    d.text((34,20),'五段完整视频：SAM2 输出 mask 的时间分布',font=title,fill='#182B42')
    d.text((34,64),'彩色＝非空预测（可能跟错）　浅灰＝空 mask　白色＝尚未给提示　黑线＝首次提示',font=font,fill='#465468')
    d.text((34,99),'每个ID的预定对象不变，实际传播可能换人；空 mask 可能是出画或漏跟。此图不能当作准确率。',font=small,fill='#465468')
    for i,(sid,name) in enumerate(zip(IDS,NAMES)):
        base=OUT/sid;meta=json.loads((base/'manifest.json').read_text());tracks=json.loads((base/'sam2_raw_v1/tracks.json').read_text())['tracks']
        n=meta['frames'];y0=150+i*248;x0=100;x1=1190;scale=(x1-x0)/n
        d.text((34,y0),f'{sid}  {name}  |  {n/10:.1f}秒',font=font,fill='#182B42')
        for j,o in enumerate(tracks):
            y=y0+42+j*30;start=o['prompt_frame'];d.text((36,y-2),f'S{o["id"]}',font=small,fill='#182B42')
            d.rectangle((x0,y,x1,y+19),outline='#DCE1E7',fill='white')
            d.rectangle((x0+start*scale,y,x1,y+19),fill='#EBEDF1')
            spans=[]
            for r in o['frames']:
                if r['bbox_xyxy'] is not None:
                    f=r['frame']
                    if spans and spans[-1][1]+1==f:spans[-1][1]=f
                    else:spans.append([f,f])
            for a,b in spans:d.rectangle((x0+a*scale,y,x0+(b+1)*scale,y+19),fill=COLORS[j])
            d.line((x0+start*scale,y-3,x0+start*scale,y+22),fill='#182B42',width=2)
        for t in range(0,int(n/10)+1,30):
            if n/10-t<12:continue
            x=x0+t*10*scale;d.text((x-9,y0+200),str(t),font=small,fill='#465468')
        d.text((x1-65,y0+200),f'{n/10:.1f}s',font=small,fill='#465468')
    img.save(OUT/'mask_presence_timeline.png')

if __name__=='__main__':main()
