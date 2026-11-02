from pathlib import Path
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
source=(ROOT/'scripts/transformer_fusion_20261005/evaluate.py').read_text(encoding='utf8')
source=source.replace("np.load(OLD/'dataset.npz')","np.load(DATA/'dataset.npz')").replace("read(OLD/'samples.json')","read(DATA/'samples.json')")
start=source.index("                z=np.load(OLD/f'fold_");end=source.index('                for v in VARIANTS:',start)
source=source[:start]+source[end:]
source=source.replace("seed=seed,mode=mode,method=method,nonoverlap=", "seed=seed,mode=mode,method=method,source_version=m['source_version'],static_control=m['static_control'],nonoverlap=")
source=source.replace("VARIANTS+['hold_last','old_comp_dual','previous_separated']","VARIANTS+['hold_last']")
needle="    summary['paired']=paired;write(OUT/'summary.json',summary)"
extra='''    summary['paired']=paired
    subsets={'old117':lambda r:r['source_version']=='old117','new82':lambda r:r['source_version']=='new82',
             'new79_nonstatic':lambda r:r['source_version']=='new82' and not r['static_control'],
             'static3':lambda r:r['static_control'],'all196_nonstatic':lambda r:not r['static_control']}
    summary['cohorts']={name:{m:reduce([r for r in reviewed if r['method']==m and predicate(r)]) for m in allmethods} for name,predicate in subsets.items()}
    # Historical experiment comparison on exactly the old test keys and seeds.
    before=[r for r in read(BASELINE/'metrics.json') if r['mode']=='reviewed' and r['method'] in VARIANTS]
    after=[r for r in reviewed if r['source_version']=='old117' and r['method'] in VARIANTS]
    baseline_comparison={}
    for v in VARIANTS:
        aa={(r['key'],r['seed']):r for r in after if r['method']==v};bb={(r['key'],r['seed']):r for r in before if r['method']==v};assert aa.keys()==bb.keys()
        delta=[dict(key=k[0],seed=k[1],recording=aa[k]['recording'],ADE=aa[k]['metrics']['ADE15']-bb[k]['metrics']['ADE15']) for k in aa]
        baseline_comparison[v]=dict(before=reduce(list(bb.values())),after=reduce(list(aa.values())),delta_ADE=float(np.mean([r['ADE'] for r in delta])),per_seed_delta={str(s):float(np.mean([r['ADE'] for r in delta if r['seed']==s])) for s in cfg['seeds']},per_recording_delta={rec:float(np.mean([r['ADE'] for r in delta if r['recording']==rec])) for rec in sorted({r['recording'] for r in delta})})
    summary['old_test_before_after']=baseline_comparison
    write(OUT/'summary.json',summary)'''
assert needle in source;source=source.replace(needle,extra)
(HERE/'evaluate.py').write_text(source,encoding='utf8')
print('Expanded cohort evaluation prepared.')
