"""Check saved annotation identities, lifecycle, geometry, and donor provenance."""
from render import *
from common import digest

def main():
    checked=[];failures=[];source_hashes={};total_points=0
    expected_core='a9691d1291c92647061f44a220f6bfd70f947264b12e9e5a503d9b5e6afde527'
    actual=digest(ROOT/'scripts/sam2_g1_v6_v8_20261001/core.py')
    if actual!=expected_core:failures.append('V6 core was modified')
    for k in range(1,11):
        cid=f'{k:02d}';folder=OUT/cid/'final';entries=read_json(folder/'census.json')['segments'];rows=v.load(folder/'masks_rle.jsonl.gz');logs=v.load(folder/'decisions.jsonl.gz')
        lookup={(r['id'],r['frame']):r for r in rows};assert len(lookup)==len(rows)
        membership={(e['id'],r['frame']):e for e in entries for r in e['rows']}
        cap=cv2.VideoCapture(str(SRC/cid/'input/rgb.mp4'));N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        for r in rows:
            assert 0<=r['frame']<N
            e=membership[r['id'],r['frame']];assert r['segment']==e['segment']
            mask=v.decode(r['rle']).astype(bool);assert mask.shape==(1920,2560) and mask.any()
            yy,xx=np.where(mask);assert r['box']==[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],(cid,r['id'],r['frame'],r['box'])
        point_count=0
        for d in logs:
            for n in d.get('negative_points',[]):
                assert n['frame']==d['frame'] and n['donor_id']!=d['id']
                donor=lookup[n['donor_id'],n['frame']];mask=v.decode(donor['rle']);x,y=n['xy'];assert mask[y,x],(cid,d['id'],d['frame'],n)
                point_count+=1
        # A reversed traversal still advances virtual time and recomposes rotations.
        cal=np.load(SRC/cid/'input/reference.npz');R,K,t=cal['R'],cal['K'],cal['times_ns']/1e9;ki=np.linalg.inv(K)
        assert len(R)==N and np.all(np.diff(t)>0)
        max_roundtrip=0.
        for e in entries:
            a=e['anchor_frame']
            for direction in [1,-1]:
                fs=list(range(a,e['end']+1 if direction==1 else e['start']-1,direction));virtual=np.abs(t[fs]-t[a]);assert len(virtual)<2 or np.all(np.diff(virtual)>0)
                for f in fs:
                    to=K@R[f].T@R[a]@ki;back=K@R[a].T@R[f]@ki;error=np.max(np.abs(to@back-np.eye(3)));max_roundtrip=max(max_roundtrip,float(error));assert error<1e-4
        review=read_json(folder/'review.json')
        for name,oldhash in review['source_hashes'].items():assert digest(OUT/cid/name)==oldhash
        for path in [SRC/cid/'input/rgb.mp4',SRC/cid/'input/reference.npz',folder/'masks_rle.jsonl.gz',folder/'decisions.jsonl.gz',folder/'census.json']:
            source_hashes[str(path.relative_to(ROOT))]=digest(path)
        checked.append(dict(clip=cid,frames=N,people=len({e['id'] for e in entries}),segments=len(entries),masks=len(rows),negative_points_validated=point_count,max_rotation_roundtrip_error=max_roundtrip))
        total_points+=point_count
    assert ('N03',0) in {(r['id'],r['frame']) for r in v.load(OUT/'08/final/masks_rle.jsonl.gz')}
    ten=v.load(OUT/'10/final/masks_rle.jsonl.gz');assert [r['frame'] for r in ten if r['id']=='N06']==[0,1]
    result=dict(passed=not failures,failures=failures,clips=checked,total_masks=sum(x['masks'] for x in checked),negative_points_validated=total_points,core_sha256=actual,source_hashes=source_hashes,interpretation='Data integrity and provenance only; not ground-truth segmentation or identity accuracy.')
    write_json(OUT/'verification.json',result);print({k:v for k,v in result.items() if k not in ['source_hashes','clips']})
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
