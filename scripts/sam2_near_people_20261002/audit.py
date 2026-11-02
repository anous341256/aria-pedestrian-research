"""Flag mask conflicts and gaps; diagnostics are not ground-truth accuracy."""
from render import *
import json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',default='raw');args=parser.parse_args()
    allstats=[];alerts=[]
    for k in range(1,11):
        cid=f'{k:02d}';folder=OUT/cid if args.revision=='raw' else OUT/cid/args.revision
        if not (folder/'complete.json').exists():continue
        rows=v.load(folder/'masks_rle.jsonl.gz');ds=v.load(folder/'decisions.jsonl.gz');entries=read_json(folder/'census.json')['segments'];byf=collections.defaultdict(list);conflicts=[]
        for r in rows:byf[r['frame']].append(r)
        for f,rr in byf.items():
            masks=[cv2.resize(v.decode(r['rle']),(640,480),interpolation=cv2.INTER_NEAREST).astype(bool) for r in rr]
            for i,a in enumerate(masks):
                for j in range(i):
                    b=masks[j];inter=np.count_nonzero(a&b);ratio=inter/max(1,min(a.sum(),b.sum()))
                    if ratio>.15 and inter>=10:conflicts.append(dict(clip=cid,frame=f,ids=[rr[i]['id'],rr[j]['id']],containment=float(ratio)))
        emitted={(r['id'],r['frame']) for r in rows}
        pending=[dict(frame=r['frame'],id=e['id'],segment=e['segment']) for e in entries for r in e['rows'] if (e['id'],r['frame']) not in emitted]
        stat=dict(clip=cid,people=len({e['id'] for e in entries}),segments=len(entries),requested=sum(len(e['rows']) for e in entries),masks=len(rows),pending=pending,conflicts=conflicts,statuses=dict(collections.Counter(d['status'] for d in ds)),negative_points=sum(len(d.get('negative_points',[])) for d in ds))
        write_json(folder/'audit.json',stat);allstats.append(stat);alerts+=conflicts
    write_json(OUT/('audit.json' if args.revision=='raw' else f'audit_{args.revision}.json'),allstats);print(json.dumps(allstats,ensure_ascii=False))
if __name__=='__main__':main()
