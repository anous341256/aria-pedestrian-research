from pathlib import Path
import json, re, subprocess, zipfile, sys
from pypdf import PdfReader
from verify_package import check_html,sha,NODE
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
BASE=Path('E:/New_research/outputs/reading_guide_20261021')
PACK=BASE/'CV_Trajectory_Reading_Guide_JA'
AUDIT=BASE/'audit_ja'
SOURCE=BASE/'CV_Trajectory_Reading_Guide'
catalog=json.loads((PACK/'catalog.json').read_text(encoding='utf-8'))
assert len(catalog)==27 and sum(c['level']=='main' for c in catalog)==14
for c in catalog:
 assert sha(PACK/c['file'])==c['sha256']==sha(SOURCE/c['file'])
 assert len(PdfReader(PACK/c['file']).pages)==c['pages']
 for key in ('short','why','flow','read','question','limit'):assert c[key]
for rel,h in json.loads((AUDIT/'original_zh_hashes.json').read_text()).items():
 assert sha(SOURCE/rel)==h,rel
# Common untranslated Chinese labels and simplified glyphs absent from Japanese UI.
for f in PACK.iterdir():
 if f.suffix in ('.html','.txt','.json'):
  text=f.read_text(encoding='utf-8-sig')
  assert not re.search(r'论文|阅读|数据流|本地|主读|选读|收录|导出|导入|先读|[这们读图轨预测视页签录输时网备查错载阶关译据仅获讲难观]',text),f
files=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='checksums.sha256')
(PACK/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),encoding='utf-8')
dom,links=check_html(PACK)
(AUDIT/'dom_fixture.json').write_text(json.dumps(dom.elements,ensure_ascii=False),encoding='utf-8')
html=(PACK/'index.html').read_text(encoding='utf-8')
assert '<html lang="ja">' in html
script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
(AUDIT/'guide.js').write_text(script,encoding='utf-8')
subprocess.run([str(NODE),'--check',str(AUDIT/'guide.js')],check=True)
# Reuse the existing in-memory unit tests, localizing only the expected UI strings.
test=Path(__file__).with_name('test_logic.cjs').read_text(encoding='utf-8')
test=test.replace('我的阅读笔记','読書メモ').replace('导入失败','読み込み失敗').replace('过大','大きすぎ')
insert="""
 x.byId('search').value='カメラ';await x.byId('search').fire('input');
 assert(x.document.querySelectorAll('.paper').filter(e=>!e.hidden).length>1);
 assert(x.byId('search-status').textContent.includes('本の論文'));
 await x.byId('clear-search').fire('click');results.push('Japanese keyword search and status message');
 assert(source.includes("localStorage.getItem(storageKey)"));
 assert(source.includes("const storageKey='trajectory-reading-guide-ja-v1'"));
 assert(source.includes("const key='trajectory-reading-guide-v1'"));
 results.push('separate autosave key with compatible export format');
"""
test=test.replace(' const output={passed:',insert+' const output={passed:')
(AUDIT/'test_logic_ja.cjs').write_text(test,encoding='utf-8')
subprocess.run([str(NODE),str(AUDIT/'test_logic_ja.cjs'),str(AUDIT)],check=True)
archive=BASE/'CV_Trajectory_Reading_Guide_JA.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(PACK.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(BASE).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert all(n.startswith(PACK.name+'/') and '..' not in Path(n).parts for n in z.namelist())
 z.extractall(AUDIT/'portable_roundtrip')
relocated=AUDIT/'portable_roundtrip'/PACK.name
for f in PACK.rglob('*'):
 if f.is_file():assert sha(f)==sha(relocated/f.relative_to(PACK))
check_html(relocated)
result=dict(language='ja',papers=27,main_readings=14,**links,all_pdfs_identical_to_chinese_edition=True,
 chinese_package_unchanged=True,translation_label_check=True,archive_integrity=True,relocation_check=True,
 archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),logic_tests=json.loads((AUDIT/'logic_results.json').read_text()),
 browser_visual_review='Not performed; previous local-file browser policy rejection respected. No browser workaround attempted.')
(AUDIT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
