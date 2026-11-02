from census import *
from PIL import Image
ds=OUT/'refined_review';ds.mkdir(exist_ok=True)
clips=[c['clip'] for c in read(OUT/'selected_clips.json') if (OUT/c['clip']/'refined/complete.json').exists()]
excluded=read(OUT/'initial_review.json')['exclude_clips'];tracks=[];index=[]
for cid in clips:
 if cid in excluded:continue
 tracks.extend(sorted((OUT/cid/'refined').glob('track_*.jpg')))
for j in range(0,len(tracks),8):
 rr=tracks[j:j+8];im=Image.new('RGB',(1400,210*len(rr)),'white')
 for k,q in enumerate(rr):im.paste(Image.open(q),(0,210*k))
 name=f'tracks_{j//8+1:03d}.jpg';im.save(ds/name,quality=93)
 index.append(dict(page=name,tracks=[f'{q.parent.parent.name}:{int(q.stem.split("_")[1])}' for q in rr]))
write(ds/'index.json',index)
print('pages',len(index),'tracks',len(tracks))
