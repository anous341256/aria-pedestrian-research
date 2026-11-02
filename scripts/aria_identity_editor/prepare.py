"""Immutable inputs and a read-only snapshot of the prior human audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from engine import CONFIG,key,pair,Engine

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/tracktrack_reproduction/aria_164557_first60s_v1'
REVIEW=ROOT/'outputs/aria_rotation_match_review/v1'
ROT=ROOT/'outputs/aria_rotation_background/rectified_v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare(dest):
    if dest.exists():raise FileExistsError('Refusing to overwrite an editor project')
    nodes=[];features=[];cache_digest=hashlib.sha256()
    for f in range(600):
        p=BASE/f'tracking/detections/{f+1:06d}.npz';cache_digest.update(f'{f+1}:{sha(p)}\n'.encode())
        with np.load(p,allow_pickle=False) as data:
            assert int(data['frame_id'])==f+1
            arr=data['tracker_input80'];assert arr.ndim==2 and arr.shape[1]==2054 and np.isfinite(arr).all()
            for i,row in enumerate(arr):
                if not np.all(row[2:4]>row[:2]):raise ValueError('Invalid box')
                nodes.append({'key':key(f,i),'frame':f,'index':i,'box':row[:4].astype(float).tolist(),'score':float(row[4])})
                features.append(row[6:].astype(np.float32))
    with np.load(ROT/'relative_rotations.npz') as r:R=r['R_current_from_previous'];K=r['K_rect']
    metadata=json.loads((BASE/'input/manifest.json').read_text(encoding='utf-8'))
    assert np.allclose(K,metadata['geometry']['target']['K'])
    source_data=json.loads((REVIEW/'dataset.json').read_text(encoding='utf-8'));queries={q['id']:q for q in source_data['queries']}
    node_map={n['key']:n for n in nodes}
    with sqlite3.connect(f'file:{(REVIEW/"review.sqlite3").as_posix()}?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        dataset_hash=db.execute("SELECT value FROM meta WHERE key='dataset_sha256'").fetchone()[0]
        if dataset_hash!=sha(REVIEW/'dataset.json'):raise ValueError('Source review dataset mismatch')
        imported=list(db.execute('SELECT query_id,candidate,value FROM labels'))
        rev=db.execute('SELECT COALESCE(MAX(id),0) FROM events').fetchone()[0]
    labels={};records=[]
    for qid,candidate,value in imported:
        q=queries[qid];a=key(q['previous_frame'],q['source_index']);b=key(q['frame'],candidate)
        if not np.allclose(node_map[a]['box'],q['source_box']) or not np.allclose(node_map[b]['box'],q['candidates'][candidate]['box']):raise ValueError('Imported detection coordinate mismatch')
        labels[pair(a,b)]=value;records.append({'a':a,'b':b,'value':value,'original_query':qid,'original_candidate':candidate})
    data={'version':1,'nodes':nodes,'config':CONFIG,'video':str(BASE/'input/aria_rgb_rectified_first60s.mp4'),
          'width':2560,'height':1920,'fps':10,'frames':600,'cache_sha256_digest':cache_digest.hexdigest(),
          'video_sha256':sha(BASE/'input/aria_rgb_rectified_first60s.mp4'),'rotation_sha256':sha(ROT/'relative_rotations.npz'),
          'review_source':str(REVIEW),'review_dataset_sha256':dataset_hash,'review_revision':rev,
          'initial_labels':labels,'imported_records':records,'readme':'Rule-based appearance + Aria prototype, no training; imported corrections are hard identity constraints.'}
    engine=Engine(nodes,np.array(features),R,K);baseline=engine.solve({});corrected=engine.solve(labels)
    dest.mkdir(parents=True);(dest/'frame_cache').mkdir()
    (dest/'input.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    np.savez_compressed(dest/'features.npz',features=np.array(features),rotations=R,K=K)
    manifest={'input_sha256':sha(dest/'input.json'),'features_sha256':sha(dest/'features.npz'),
              'prepare_sha256':sha(__file__),'engine_sha256':sha(Path(__file__).with_name('engine.py')),
              'imported_review_revision':rev,'imported_decisions':len(labels),'automatic_baseline':baseline['stats'],
              'after_import':corrected['stats'],'duplicate_resolution':corrected['duplicates'],
              'scope':'All outputs are editable preannotations, not ground truth; no CVAT upload, original files unchanged.'}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'automatic_baseline.json').write_text(json.dumps(baseline,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();prepare(args.output)
