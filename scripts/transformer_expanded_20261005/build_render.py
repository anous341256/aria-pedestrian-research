from pathlib import Path
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
s=(ROOT/'scripts/transformer_fusion_20261005/render.py').read_text(encoding='utf8')
s=s.replace("np.load(OLD/'dataset.npz')","np.load(DATA/'dataset.npz')").replace("read(OLD/'samples.json')","read(DATA/'samples.json')")
s=s.replace("fixed=read(OLD/'presentation/selection.json')['fixed'];selected=[];reasons={}","cases=read(OUT/'fixed_cases.json');fixed=cases['old']+cases['new'];selected=[];reasons={}")
s=s.replace("key=func(scores,key=lambda k:scores[k][a]-scores[k][b])","key=func([k for k in scores if meta[lookup[k]]['source_version']=='new82'],key=lambda k:scores[k][a]-scores[k][b])")
s=s.replace("fixed_rule='Same original nine cases; first eligible person and median cutoff per clip.'","fixed_rule='Original nine plus nine new cases frozen before fitting; first eligible identity and median cutoff per clip.'")
s=s.replace("selected_rule='Post-score seed-17 gain/loss extrema; not representative average performance.'","selected_rule='Post-score seed-17 new-cohort gain/loss extrema; not representative average performance.'")
s=s.replace("cap=cv2.VideoCapture(str(ROOT/f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4'));ff=[]","video=ROOT/(f'outputs/sam2_g1_v6_v8_20261001/{cid}/input/rgb.mp4' if int(cid)<=10 else f'outputs/sam2_g1_expansion_20261005/{cid}/input/rgb.mp4');cap=cv2.VideoCapture(str(video));ff=[]")
s=s.replace("masks[cid]={(r['id'],r['frame']):r for r in gz(ROOT/f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz')}","labelpath=ROOT/(f'outputs/sam2_near_people_20261002/{cid}/final/masks_rle.jsonl.gz' if int(cid)<=10 else f'outputs/sam2_g1_expansion_20261005/{cid}/final/masks_rle.jsonl.gz');masks[cid]={(r['id'],r['frame']):r for r in gz(labelpath)}")
s=s.replace("[('fixed_9_six_groups',fixed),('selected_gain_loss_six_groups',selected)]","[('fixed_old9_six_groups',cases['old']),('fixed_new9_six_groups',cases['new']),('selected_new_gain_loss_six_groups',selected)]")
(HERE/'render.py').write_text(s,encoding='utf8');print('Rendering script prepared.')
