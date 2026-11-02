"""One temporally middle selected window per contributing clip, masks and center."""
from qa import *
from common import encoder,finish_encoder
def main():
 dest=OUT/'presentation';dest.mkdir(exist_ok=True);meta=read_json(OUT/'dataset/clear_samples.json')
 W,H=1280,1060;enc,log=encoder(dest/'SELECTED_G1_WINDOWS.mp4',W,H,10,20);cases=[];cards=[]
 for cid in sorted({m['clip'] for m in meta}):
  mm=sorted([m for m in meta if m['clip']==cid],key=lambda m:(m['cutoff'],m['id']));m=mm[len(mm)//2];lookup={(r['id'],r['frame']):r for r in v.load(OUT/cid/'final/masks_rle.jsonl.gz')}
  cap=cv2.VideoCapture(str(OUT/cid/'input/rgb.mp4'));frames=[]
  for f in m['frames']:
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read();assert ok;r=lookup[m['id'],f];im,_=v.overlay(frame,[r],{m['id']:(85,215,110)},{},f)
   canvas=np.full((H,W,3),22,np.uint8);canvas[65:1025]=im;b=np.array(r['box']);center=(b[:2]+b[2:])/4;cv2.circle(canvas,(round(center[0]),round(center[1])+65),5,(40,60,245),-1)
   canvas=v.text(canvas,[(15,5,f'新 G1 等量批次  |  片段 {cid}  ·  {m["id"]}'),(15,35,f'{m["description"]}  |  时间 {m["source_start_s"]+f/10:.1f}s'),(15,1029,'实际入选窗口 · 绿色 SAM mask / 红色外接框中心 · 离线抽查标签，非独立人工GT')],21)
   frames.append(canvas)
  cap.release()
  for repeat in range(2):
   for canvas in frames:enc.stdin.write(canvas.tobytes())
  preview=frames[5];cv2.imwrite(str(dest/f'{cid}_selected_preview.jpg'),preview);cards.append(cv2.resize(preview,(640,530)));cases.append(dict(key=m['key'],clip=cid,repeat=2,frames=42))
 finish_encoder(enc,log);contact=np.zeros((math.ceil(len(cards)/2)*530,1280,3),np.uint8)
 for j,im in enumerate(cards):contact[j//2*530:(j//2+1)*530,j%2*640:(j%2+1)*640]=im
 cv2.imwrite(str(dest/'SELECTED_CONTACT.jpg'),contact);write_json(dest/'selected_media.json',dict(cases=cases,fps=10,frames=len(cases)*42,selection='Middle selected window by cutoff and identity for each contributing clip; no forecast-score selection.'))
 print('Selected montage',len(cases),'cases',flush=True)
if __name__=='__main__':main()
