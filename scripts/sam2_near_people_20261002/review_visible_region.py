"""Choose the reviewed visible-body hypothesis rather than the detached arm."""
from run import *
import shutil

def main():
    dest=OUT/'09/final';backup=OUT/'09/review_pass2';backup.mkdir(exist_ok=True)
    for name in ['masks_rle.jsonl.gz','decisions.jsonl.gz','polish.json']:shutil.copy2(dest/name,backup/name)
    rows=inc.loadgz(dest/'masks_rle.jsonl.gz');logs=inc.loadgz(dest/'decisions.jsonl.gz');log=next(d for d in logs if d['id']=='N04' and d['frame']==7)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        tracker=load_adapter().create(model_id='facebook/sam2.1-hiera-large',device='cuda');predictor=SAM2ImagePredictor(tracker._predictor)
        c=cv2.VideoCapture(str(SRC/'09/input/rgb.mp4'));c.set(cv2.CAP_PROP_POS_FRAMES,7);ok,frame=c.read();c.release();assert ok
        predictor.set_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        neg=log['negative_points'];mm,ss,_=predictor.predict(box=np.array(log['prompt_box']),point_coords=np.array([n['xy'] for n in neg],np.float32),point_labels=np.zeros(len(neg),np.int32),multimask_output=True)
        mask=mm[0].astype(bool);q=quality(mask,log['prompt_box']);assert q['valid'] and q['box'][3]<1560
        row=next(r for r in rows if r['id']=='N04' and r['frame']==7);row.update(rle=encode_mask(mask),box=q['box'],quality=q,revision=3)
        log.update(quality=q,sam_score=float(ss[0]),revision=3,selected_multimask_index=0,selection_reason='Visual review: upper-body hypothesis is the visible target; higher-score hypotheses include another persons detached forearm below the occluder')
        inc.dumpgz(dest/'masks_rle.jsonl.gz',rows);inc.dumpgz(dest/'decisions.jsonl.gz',logs)
        write_json(dest/'visible_region_review.json',dict(id='N04',frame=7,selected_index=0,quality=q,reason=log['selection_reason'],preserved_previous='review_pass2',code_sha256=digest(Path(__file__))))
        print(q,flush=True)
if __name__=='__main__':main()
