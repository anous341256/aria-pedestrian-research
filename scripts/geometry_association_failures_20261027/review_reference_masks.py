from render_cases import *
def main():
 inv=read(BASE/'inventory.json');ref=read(BASE/'source_identity_reference.json');rowsout=[]
 requests=[('07',['N06','N07'],list(range(14,24)),'D_reference'),('04',['N01','N03'],list(range(23,29)),'C_reference'),('03',['N07','N05'],[14,15,16,17,20,21,22],'B_reference'),('09',['N04','N03'],[17,18,19,20,25,26,27,28],'G_reference')]
 for cid,ids,fs,name in requests:
  entries={e['frame']:e for e in inv['entries'] if e['clip']==cid};cap=cv2.VideoCapture(str(ROOT/entries[0]['input']/'rgb.mp4'));images={}
  for f in range(max(fs)+1):
   ok,im=cap.read();assert ok
   if f in fs:images[f]=im
  cap.release();sheet=np.full((760,len(fs)*240,3),20,np.uint8);txt(sheet,f'{name}: actual SAM foreground; gray outside mask. Inspect source IDs, not tracker output.',10,30,.60)
  for k,sid in enumerate(ids):
   for j,f in enumerate(fs):
    y=55+k*350;txt(sheet,f'{sid} / f{f}',j*240+8,y+20,.55);target=next((t for t in entries[f]['targets'] if ref[t['target_key']]['source_id']==sid),None)
    if target is None:txt(sheet,'NO CANDIDATE',j*240+8,y+180,.5);continue
    mask=decode(target['rle']).astype(bool);im=images[f].copy();im[~mask]=127;sheet[y+30:y+330,j*240:(j+1)*240]=fit(crop(im,target['box']),240,300)
  cv2.imwrite(str(PRES/f'{name}_masks.jpg'),sheet)
  for f in fs:
   targets=[t for t in entries[f]['targets'] if ref[t['target_key']]['source_id'] in ids]
   if len(targets)==2:
    a,b=[decode(t['rle']).astype(bool) for t in targets];rowsout.append(dict(clip=cid,frame=f,source_ids=[ref[t['target_key']]['source_id'] for t in targets],mask_iou=float((a&b).sum()/max(1,(a|b).sum()))))
 write(OUT/'reference_overlap_checks.json',rowsout)
if __name__=='__main__':main()
