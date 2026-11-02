"""Assemble local research briefing and exact artifact manifests; no source edits."""
from pathlib import Path
import hashlib,html,json,re,subprocess

ROOT=Path('E:/New_research')
OUT=ROOT/'outputs/research_branches_20261029'
BRANCHES={
 'scene':Path('C:/Users/nytha/.codex/worktrees/scene-pedestrian/New_research'),
 'gaze':Path('C:/Users/nytha/.codex/worktrees/gaze-motion/New_research')}
def read(p):return p.read_text(encoding='utf-8-sig')
def rich(s):
    return re.sub(r'(https?://[^\s<>]+)',r'<a href="\1">\1</a>',html.escape(s))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    names={'scene':'scene_pedestrian_20261029','gaze':'gaze_motion_20261029'}
    overview=read(ROOT/'reports/research_branches_20261029/OVERVIEW.txt')
    sections=[f'<article><pre>{rich(overview)}</pre></article>']
    registry={}
    for label,work in BRANCHES.items():
        version=names[label];body=read(work/f'reports/{version}/FEASIBILITY.txt')
        (OUT/f'{label}_FEASIBILITY.txt').write_text(body,encoding='utf8')
        sections.append(f'<details id="{label}"><summary>{"场景与行人运动" if label=="scene" else "眼动与自身、周围行人运动"}：完整调查报告</summary><pre>{rich(body)}</pre></details>')
        registry[label]={'worktree':str(work),'branch':subprocess.check_output(['git','-C',str(work),'branch','--show-current'],text=True).strip(),
                         'commit':subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip(),
                         'report':str(work/f'reports/{version}/FEASIBILITY.txt')}
    rows=json.loads(read(OUT/'inventory/inventory.json'))['recordings']
    table='<table><tr><th>录制</th><th>原录制场景描述（非局部标签）</th><th>眼动跨度/s</th><th>短时窗口</th><th>长时窗口</th></tr>'
    for r in rows:
        table+='<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [r['recording'][-6:],r['scene_note'].get('scene','未知'),r.get('gaze_span_s',0),r['short_expanded_windows'],r['long_expanded_windows']])+'</tr>'
    table+='</table>'
    sections.append('<details><summary>29段录制的可用数据清单</summary>'+table+'</details>')
    sections.append('<details><summary>21段固定抽样场景图（仅检查设施可见性）</summary>'+''.join(f'<img src="scenes/scenes_{i:02}.jpg" alt="固定抽样场景第{i}页">' for i in range(1,4))+'</details>')
    source='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>场景与眼动：两个研究分支</title><style>body{max-width:1060px;margin:36px auto;padding:0 24px;background:#f5f7fa;color:#192b3c;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}article,details{background:white;border:1px solid #dbe3eb;border-radius:10px;padding:22px;margin:18px 0}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}summary{cursor:pointer;font-weight:650;color:#12476a}a{color:#176796}img{max-width:100%;display:block;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #dae1e8;padding:7px;text-align:left}h1{font-size:29px}nav{display:flex;gap:22px;flex-wrap:wrap}</style><h1>场景、眼动与运动：前期调查</h1><nav><a href="#scene" onclick="document.getElementById('scene').open=true">场景分支</a><a href="#gaze" onclick="document.getElementById('gaze').open=true">眼动分支</a></nav>'''+''.join(sections)+'</html>'
    (OUT/'index.html').write_text(source,encoding='utf8')
    reg=ROOT/'experiment_registry/experiments/research_branches_20261029';reg.mkdir(parents=True,exist_ok=True)
    artifacts={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.rglob('*')) if p.is_file()}
    data={'status':'preliminary investigation complete; proposed experiments not trained','base_commit':'7d42c06534833f202e5de5e1c836a004a6e7ade6','branches':registry,'artifacts':artifacts}
    (reg/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    for name in ['verification.json','inventory/inventory.json','gaze/audit.json','scenes/review_manifest.json']:
        (reg/Path(name).name).write_text(read(OUT/name),encoding='utf8',newline='\n')
    print('PACKAGED',len(artifacts),'artifacts',registry)
if __name__=='__main__':main()
