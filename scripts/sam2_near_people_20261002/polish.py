"""Two visually verified local negative-point corrections, retaining pass one."""
from run import *
import shutil

def main():
    specs=[('08',38,'N05','N02',[[1750,660],[1800,600],[1770,625]],[[1769,532]]),('09',7,'N04','N01',[[1740,1680],[1730,1670]],[])]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda')
        for cid,f,identity,donor_id,neg,positive in specs:
            dest=OUT/cid/'final';backup=OUT/cid/'review_pass1'
            if (dest/'polish.json').exists():continue
            backup.mkdir(exist_ok=True)
            for file in dest.iterdir():
                if file.is_file():shutil.copy2(file,backup/file.name)
            rows=inc.loadgz(dest/'masks_rle.jsonl.gz');logs=inc.loadgz(dest/'decisions.jsonl.gz');census=read_json(dest/'census.json')
            donor=next(r for r in rows if r['id']==donor_id and r['frame']==f);dm=decode(donor['rle'])
            assert all(dm[y,x] for x,y in neg)
            log=next(d for d in logs if d['id']==identity and d['frame']==f);box=log['prompt_box']
            cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read();cap.release();assert ok
            predictor=SAM2ImagePredictor(tracker._predictor);predictor.set_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            mm,ss,_=predictor.predict(box=np.array(box),point_coords=np.array(positive+neg,np.float32),point_labels=np.array([1]*len(positive)+[0]*len(neg)),multimask_output=True)
            choices=[]
            for i,m in enumerate(mm):
                m=m.astype(bool);q=quality(m,box);overlap=float((m&dm).sum()/max(1,min(m.sum(),dm.sum())));accepted=q['valid'] and ss[i]>.45 and overlap<.15 and not any(m[y,x] for x,y in neg) and all(m[y,x] for x,y in positive)
                choices.append(dict(index=i,sam_score=float(ss[i]),quality=q,donor_overlap=overlap,accepted=bool(accepted)))
            good=[c for c in choices if c['accepted']];chosen=max(good,key=lambda c:c['sam_score']) if good else None
            rows=[r for r in rows if not(r['id']==identity and r['frame']==f)]
            log.update(status='REVIEW_PROMPT' if chosen else 'REVIEW_PENDING',revision=2,positive_points=positive,negative_points=[dict(donor_id=donor_id,frame=f,xy=p,source='visually_verified_local_point_in_current_donor_mask') for p in neg],choices=choices,reason='Local review negative points in verified foreground donor; automatic interior points did not address overlap')
            if chosen:
                q=chosen['quality'];e=next(e for e in census['segments'] if e['id']==identity and e['start']<=f<=e['end']);rows.append(dict(id=identity,frame=f,segment=e['segment'],batch=e['batch'],rle=encode_mask(mm[chosen['index']].astype(bool)),box=q['box'],quality=q,donor_eligible=False,revision=2));log.update(quality=q,sam_score=chosen['sam_score'])
            inc.dumpgz(dest/'masks_rle.jsonl.gz',sorted(rows,key=lambda r:(r['frame'],r['id'])));inc.dumpgz(dest/'decisions.jsonl.gz',logs)
            write_json(dest/'polish.json',dict(id=identity,frame=f,choices=choices,accepted=bool(chosen),negative_points=log['negative_points'],preserved_previous='review_pass1',code_sha256=digest(Path(__file__))))
            print(cid,choices,flush=True)
if __name__=='__main__':main()
