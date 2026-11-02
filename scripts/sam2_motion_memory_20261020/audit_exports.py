"""Read-only checks of mask serialization, target/frame alignment and shared seeds."""
import argparse, collections, gzip, json
from pathlib import Path
import numpy as np
from selection import OUT, read, write

ARMS = ['visual', 'external_box', 'camera', 'camera_person']

def rows(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f]

def validate_rle(encoded):
    a = np.asarray(encoded)
    assert len(a) >= 5 and np.isfinite(a).all()
    assert np.equal(a, np.floor(a)).all()
    lengths = a[:-4].astype(np.int64)
    x1, y1, x2, y2 = a[-4:].astype(int)
    assert (lengths >= 0).all()
    assert 0 <= x1 <= x2 < 2560 and 0 <= y1 <= y2 < 1920
    assert lengths.sum() == (x2-x1+1)*(y2-y1+1)
    return [int(x1), int(y1), int(x2+1), int(y2+1)], int(lengths[1::2].sum())

def main(partial=False):
    selected = read(OUT/'selection.json')
    complete = [c for c in selected if (OUT/c['id']/'complete.json').exists()]
    if not partial:
        assert len(complete) == len(selected)
    counts = collections.Counter()
    flags = []
    for c in complete:
        entries = {e['segment']: e for e in c['entries']}
        seeds = {r['segment']: r for r in rows(OUT/c['id']/'seeds.jsonl.gz')}
        assert set(seeds) == set(entries)
        for s in seeds.values():
            validate_rle(s['rle'])
        for arm in ARMS:
            seen = set()
            logs = rows(OUT/c['id']/arm/'decisions.jsonl.gz')
            for r in logs:
                key = (r['segment'], r['frame'])
                assert key not in seen, (c['id'], arm, key)
                seen.add(key)
                e = entries[r['segment']]
                assert e['start'] <= r['frame'] <= e['end']
                assert r.get('id') == e['id']
                if 'raw_rle' in r:
                    _, pixels = validate_rle(r['raw_rle'])
                    source = next(x for x in r['raw_candidates'] if x['source'] == r['raw_source'])
                    assert pixels == source['pixels']
                    counts['raw_masks_checked'] += 1
            seen = set()
            for r in rows(OUT/c['id']/arm/'masks_rle.jsonl.gz'):
                key = (r['segment'], r['frame'])
                assert key not in seen, (c['id'], arm, key)
                seen.add(key)
                e = entries[r['segment']]
                assert e['start'] <= r['frame'] <= e['end']
                assert r['id'] == e['id'] and r['batch'] == e['batch']
                box, pixels = validate_rle(r['rle'])
                assert box == r['box'] == r['quality']['box']
                assert pixels == r['quality']['pixels']
                if r['frame'] == e['anchor_frame']:
                    assert r['rle'] == seeds[r['segment']]['rle']
                    counts['identical_seed_exports'] += 1
                    if arm == 'visual' and (not r['quality']['valid'] or seeds[r['segment']]['score'] < .6 or box[3]-box[1] < .08*1920):
                        flags.append(dict(clip=c['id'], segment=r['segment'], frame=r['frame'],
                            quality=r['quality'], sam_score=seeds[r['segment']]['score'],
                            below_retention=box[3]-box[1] < .08*1920))
                else:
                    assert box[3]-box[1] >= .08*1920
                counts['final_masks_checked'] += 1
    result = dict(clips=len(complete), total_clips=len(selected), partial=partial,
        counts=dict(counts), seed_quality_flags=flags,
        note='Shared seeds are identical across arms. Seed flags are review warnings, not removals or proof of identity error.')
    write(OUT/('export_audit_partial.json' if partial else 'export_audit.json'), result)
    print(json.dumps(dict(clips=len(complete), counts=dict(counts), seed_quality_flags=len(flags))))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true')
    main(p.parse_args().partial)
