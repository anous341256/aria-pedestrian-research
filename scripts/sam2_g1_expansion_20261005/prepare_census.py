"""Persist the visually reviewed census, proposal aliases, and visible segments.

Review is annotation assistance, not a fully automatic association benchmark.
All coordinates refer to the rectified 2560 x 1920 working videos.
"""
from census import *

from reviewed_people import PEOPLE,MANUAL,REJECT,REVIEW_ONLY,REJECTED_CLIPS

def main():
    write_json(OUT/'rectified_selection_review.json',dict(accepted=list(PEOPLE),rejected=REJECTED_CLIPS,annotated_seconds=70))
    totals=[]
    for cid,people in PEOPLE.items():
        if (OUT/cid/'census.json').exists():continue
        props=read_json(OLD/cid/'proposal_tracklets.json');byid={t['proposal_id']:t for t in props};near=read_json(OUT/cid/'near_proposals.json')
        cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));frames=[]
        while True:
            ok,im=cap.read()
            if not ok:break
            frames.append(im)
        cap.release();entries=[];used=set()
        for i,(description,parts) in enumerate(people,1):
            rows=[];ids=[]
            for part in parts:
                pid,lo,hi=part if isinstance(part,tuple) else (part,0,10000)
                ids.append(pid);used.add(pid)
                rows.extend(dict(r,proposal_id=pid) for r in byid[pid]['rows'] if lo<=r['frame']<=hi)
            man=MANUAL.get((cid,i))
            if man:
                rows=[dict(frame=f,box=b,confidence=1.,source='visual_review_box',detection_id=None,proposal_id=None) for f,b in man['rows']]
            # Prefer full-frame whole-body boxes to highly confident tile head/leg crops.
            byframe={}
            for r in rows:
                f=r['frame'];score=r['confidence']*math.sqrt(max(1,r['box'][3]-r['box'][1]))
                if f not in byframe or score>byframe[f][0]:byframe[f]=(score,r)
            rows=[byframe[f][1] for f in sorted(byframe)]
            # Admit by 12% height at least once; retain surrounding visible 8% interval.
            # Reviewed border fragments are admitted even at one frame.
            rows=[r for r in rows if r['box'][3]-r['box'][1]>=.08*1920 or man or (cid=='08' and i==3)]
            chunks=[]
            for r in rows:
                if not chunks or r['frame']-chunks[-1][-1]['frame']>5:chunks.append([])
                chunks[-1].append(r)
            if man:chunks=[rows]
            for s,chunk in enumerate(chunks,1):
                start,end=chunk[0]['frame'],chunk[-1]['frame'];dense=[]
                for f in range(start,end+1):
                    exact=next((r for r in chunk if r['frame']==f),None)
                    if exact:dense.append(exact);continue
                    left=max((r for r in chunk if r['frame']<f),key=lambda r:r['frame']);right=min((r for r in chunk if r['frame']>f),key=lambda r:r['frame']);t=(f-left['frame'])/(right['frame']-left['frame'])
                    dense.append(dict(frame=f,box=((1-t)*np.array(left['box'])+t*np.array(right['box'])).tolist(),confidence=min(left['confidence'],right['confidence']),source='short_gap_box_interpolation',proposal_id=None,detection_id=None))
                def anchor_score(r):
                    x1,y1,x2,y2=np.array(r['box']).astype(int);roi=frames[r['frame']][max(0,y1):min(1920,y2),max(0,x1):min(2560,x2)]
                    sharp=cv2.Laplacian(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() if roi.size else 0
                    edge=.7 if min(x1,y1)<3 or x2>2556 or y2>1916 else 1.
                    return r['confidence']*math.sqrt(max(1,y2-y1))*edge*(.7+.3*min(1,sharp/100))
                anchor=max(chunk,key=anchor_score)
                entry=dict(id=f'N{i:02d}',segment=f'N{i:02d}_S{s:02d}',description=description,proposal_ids=ids,start=start,end=end,anchor_frame=anchor['frame'],anchor_box=anchor['box'],rows=dense,
                           review='Agent visual census review: identity/duplicate grouping and near admission; pixel masks not yet reviewed.',manual=man,
                           end_reason='clip_end' if end==len(frames)-1 else 'visible_proposal_interval_end_pending_boundary_review')
                entry['review_only_reason']=REVIEW_ONLY.get((cid,i))
                entries.append(entry)
        dispositions=[]
        for t in near:
            pid=t['proposal_id'];status='registered_person_proposal' if pid in used else 'reviewed_duplicate_or_partial'
            if pid in REJECT.get(cid,{}):status='rejected_'+REJECT[cid][pid]
            dispositions.append(dict(proposal_id=pid,status=status))
        entries.sort(key=lambda e:(-max(r['box'][3]-r['box'][1] for r in e['rows']),e['start']))
        for j,e in enumerate(entries):e['batch']=j//3+1
        write_json(OUT/cid/'census.json',dict(people=len(people),segments=entries,dispositions=dispositions,total_cap=None,batch_size=3))
        # Contact strips expose detector identity switches and review boundaries.
        for page in range(math.ceil(len(entries)/8)):
            sheet=Image.new('RGB',(1400,8*200),(242,242,242));draw=ImageDraw.Draw(sheet)
            for j,e in enumerate(entries[page*8:(page+1)*8]):
                y=j*200;draw.text((4,y+3),f"{e['segment']} {e['start']}-{e['end']}\n{e['description']}",font=FONT,fill=(0,0,0))
                sample=np.unique(np.linspace(0,len(e['rows'])-1,6).astype(int))
                for h,idx in enumerate(sample):
                    r=e['rows'][idx];x1,y1,x2,y2=np.array(r['box']).astype(int);pad=15
                    crop=Image.fromarray(cv2.cvtColor(frames[r['frame']][max(0,y1-pad):min(1920,y2+pad),max(0,x1-pad):min(2560,x2+pad)],cv2.COLOR_BGR2RGB));crop.thumbnail((180,168));x=270+h*185;sheet.paste(crop,(x,y));draw.text((x,y+170),f"f{r['frame']}",fill=(0,0,0),font=FONT)
            sheet.save(OUT/cid/f'identity_review_{page+1}.jpg',quality=94)
        totals.append(dict(clip=cid,people=len(people),segments=len(entries),requested_frames=sum(len(e['rows']) for e in entries)))
    write_json(OUT/'census_summary.json',totals);print(totals)
if __name__=='__main__':main()
