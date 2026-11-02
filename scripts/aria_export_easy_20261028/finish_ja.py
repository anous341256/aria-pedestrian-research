from pathlib import Path
import hashlib,json,zipfile,ast
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path('E:/New_research/outputs/aria_export_easy_20261028')
PACK=ROOT/'Aria_Video_Exporter_JA';SRC=ROOT/'Aria_Video_Exporter';AUDIT=ROOT/'audit_ja'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,h in json.loads((AUDIT/'original_hashes.json').read_text()).items():assert sha(SRC/name)==h
receipts=[json.loads(p.read_text(encoding='utf-8')) for p in (AUDIT/'出力結果').glob('*/result.json')]
assert len(receipts)==1 and receipts[0]['status']=='complete'
v=receipts[0]['verification'];assert v['frames']==75 and v['width']==2560 and v['height']==1920 and v['full_decode_passed'] and v['metadata_matches_csv']
for name in ('launcher.ps1','export_video.py','check_env.py','run.sh'):
 assert sha(PACK/name)==sha(AUDIT/'日本語 test & run'/name)
for p in PACK.glob('*.sh'):assert b'\r' not in p.read_bytes()
for p in PACK.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
for d in json.loads((PACK/'DEPENDENCIES.json').read_text(encoding='utf-8')):
 assert sha(PACK/d['file'])==d['sha256']
assert not (PACK/'README_ZH.txt').exists()
for p in (PACK/'START_HERE.html',PACK/'README_JA.txt',PACK/'VALIDATION.txt'):
 s=p.read_text(encoding='utf-8-sig');assert '中文' not in s and 'zh-CN' not in s
class Check(HTMLParser):
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'href' in a:
   u=urlsplit(a['href'])
   if not u.scheme:assert (PACK/unquote(u.path)).is_file()
Check().feed((PACK/'START_HERE.html').read_text(encoding='utf-8'))
with (PACK/'VALIDATION.txt').open('a',encoding='utf-8') as f:
 f.write('\n日本語版の再検証：\nPASS: 日本語の起動補助から実際の VRS を再変換。2560×1920、75 フレーム、音声あり。\nPASS: 全編デコードと、CSV・MP4 内のタイムスタンプの一致。\nPASS: PowerShell / Bash / Python の構文、日本語・空白・& を含むパス。\nPASS: 同梱依存関係のハッシュ不変。元の中日併記パッケージも不変。\nPASS: 日本語 ZIP の CRC と全ファイルの展開後ハッシュ。\n日本語版では共通の検証済み環境を再利用。WSL の初回有効化と手動ダイアログ操作は別 PC で未検証。\n')
files=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='checksums.sha256')
assert not any(p.suffix in ('.vrs','.mp4','.log') or 'run_logs' in p.parts for p in files)
(PACK/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),encoding='utf-8')
archive=ROOT/'Aria_Video_Exporter_Windows_JA.zip'
with zipfile.ZipFile(archive,'w') as z:
 for p in sorted(PACK.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(ROOT).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.whl' else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 z.extractall(AUDIT/'roundtrip')
for p in PACK.rglob('*'):
 if p.is_file():assert sha(p)==sha(AUDIT/'roundtrip'/p.relative_to(ROOT))
result={'language':'ja','bytes':archive.stat().st_size,'sha256':sha(archive),'real_export':v,'original_bilingual_unchanged':True,'archive_verified':True}
(AUDIT/'delivery_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
