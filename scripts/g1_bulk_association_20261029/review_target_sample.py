from bulk_common import *
import cv2
ww=read(OUT/'forecast_window_candidates.json')['windows'];chosen={}
for w in ww:chosen.setdefault(w['recording'],w['key'])
keys=list(chosen.values());out=OUT/'review_target_sample';out.mkdir(exist_ok=True)
for i in range(0,len(keys),6):
 canvas=np.zeros((900,1800,3),np.uint8)
 for j,key in enumerate(keys[i:i+6]):
  im=cv2.imread(str(OUT/'presentation/target_strips'/f'{key}.jpg'));assert im is not None;canvas[(j//2)*300:(j//2+1)*300,(j%2)*900:(j%2+1)*900]=im
 cv2.imwrite(str(out/f'page_{i//6+1:02d}.jpg'),canvas)
write(out/'index.json',dict(rule='First continuous window per recording in frozen ordering; no outcome selection',keys=keys,recordings=list(chosen)))
