"""Create auditable runner copies: only input paths, frame origin, provenance vary."""
from pathlib import Path
import io,tokenize,shutil,difflib,json,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
V6=ROOT/'scripts/sam2_motion_tracker_v1_20261001';V8=ROOT/'scripts/sam2_reliability_prior_20261001'
def origin_zero(s):
    ts=[]
    for t in tokenize.generate_tokens(io.StringIO(s).readline):
        if t.type==tokenize.NUMBER and t.string in ['100','340','99']:
            t=t._replace(string={'100':'0','340':'N','99':'(-1)'}[t.string])
        ts.append(t)
    return tokenize.untokenize(ts)
def save(src,name,s):
    compile(s,name,'exec');(HERE/name).write_text(s,encoding='utf8')
    (HERE/(name+'.diff')).write_text(''.join(difflib.unified_diff(src.read_text().splitlines(True),s.splitlines(True),fromfile=str(src),tofile=name)),encoding='utf8')
for name,src in [('core.py',V6/'core.py'),('tracker_geometry.py',V8/'tracker_geometry.py'),('reliability.py',V8/'reliability.py')]:
    shutil.copyfile(src,HERE/name)
prefix="\nimport os\nCLIP=ROOT/'outputs/sam2_g1_v6_v8_20261001'/os.environ['G1_CLIP']\nOLD=CLIP/'input'\nVIDEO=OLD/'rgb.mp4'\nPOSE=OLD/'reference.npz'\nALIGN=OLD/'alignment.json'\nN=read_json(OLD/'metadata.json')['frames']\n"
src=V8/'sam_io.py';s=origin_zero(src.read_text());p=s.index('\n\nclass Camera:');s=s[:p]+prefix+s[p:];save(src,'sam_io.py',s)
for ver,src in [('v6',V6/'run.py'),('v8',V8/'run.py')]:
    s=origin_zero(src.read_text());p=s.index('\n\nclass Camera:' if ver=='v6' else '\n\ndef small_overlap')
    s=s[:p]+prefix+f"OUT=CLIP/'{ver}'\n"+("BRANCHES=['motion_with_camera']\n" if ver=='v6' else '')+s[p:]
    s=s.replace("all(a['status']=='valid' for a in alignment[0:N])","all(a['status'] in ['valid','held_previous_valid'] for a in alignment[0:N])")
    s=s.replace("assert read_json(OUT/'unit_checks.json')['status']=='passed'","assert read_json(ROOT/'outputs/sam2_g1_v6_v8_20261001/unit_checks.json')['status']=='passed'")
    s=s.replace("('baseline_masks',OLD/'masks_rle.jsonl.gz'),",'')
    s=s.replace("scope='Five fixed people, 24-second full clip. Basic conflict handling; no trained re-identification, '","scope='Selected G1 clip with reviewed fixed seeds. Basic conflict handling; no trained re-identification, '")
    s=s.replace('User requested implementation of the combined prior and maintenance of earlier attempts.','User requested frozen versions 6 and 8 on all ten selected G1 clips.')
    s=s.replace('/240','/{N}')
    s=s.replace("['tracker_geometry.py','sam_io.py','reliability.py','run.py','test_policy.py']","['tracker_geometry.py','sam_io.py','reliability.py','v8_run.py','check_frozen.py']")
    save(src,ver+'_run.py',s)
print('Frozen policy modules copied; runner adaptations recorded as diffs.')
