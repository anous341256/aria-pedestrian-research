from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
from pypdf import PdfReader
import json, hashlib, zipfile, re, subprocess, sys
from prepare import PACK, OUT
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
AUDIT=OUT/'audit'
NODE=Path('C:/Users/nytha/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Inspect(HTMLParser):
 def __init__(self):
  super().__init__();self.elements=[];self.ids=set();self.stack=[];self.links=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:
   assert a['id'] not in self.ids,('duplicate id',a['id'])
   self.ids.add(a['id'])
  self.elements.append(dict(tag=tag,attrs=a,parent=self.stack[-1][1] if self.stack else None))
  if tag not in ('meta','link','input','img','br','hr','source','wbr','area','base','embed','param','track'):
   self.stack.append((tag,len(self.elements)-1))
  for k in ('href','src'):
   if k in a:self.links.append((tag,k,a[k]))
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,-1,-1):
   if self.stack[i][0]==tag:self.stack=self.stack[:i];break
def check_html(root):
 html=(root/'index.html').read_text(encoding='utf-8');p=Inspect();p.feed(html)
 assert '{{' not in html
 local=external=0
 for tag,attr,link in p.links:
  u=urlsplit(link)
  if u.scheme:
   assert tag=='a' and attr=='href' and u.scheme=='https',('unexpected resource',link)
   external+=1
  elif u.path:
   target=(root/unquote(u.path)).resolve()
   assert target.is_relative_to(root.resolve()) and target.is_file(),link
   local+=1
  elif u.fragment:assert u.fragment in p.ids,link
 for f in root.iterdir():
  if f.suffix in ('.html','.json','.txt'):
   txt=f.read_text(encoding='utf-8')
   assert not re.search(r'(?i)(C:[\\/]|E:[\\/]|nytha|localhost|127\.0\.0\.1)',txt),f
 return p,dict(local_links=local,explicit_external_links=external)
def main():
 catalog=json.loads((PACK/'catalog.json').read_text(encoding='utf-8'))
 manifest=json.loads((AUDIT/'acquisition_manifest.json').read_text(encoding='utf-8'))
 assert len(catalog)==27 and sum(c['level']=='main' for c in catalog)==14
 local_sources=0
 for c in manifest:
  p=PACK/c['file'];assert sha(p)==c['sha256']
  assert len(PdfReader(p).pages)==c['pages']
  if c['source']:
   assert sha(Path(c['source']))==c['sha256'];local_sources+=1
 files=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='checksums.sha256')
 (PACK/'checksums.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(PACK).as_posix()}\n' for p in files),encoding='utf-8')
 p,links=check_html(PACK)
 (AUDIT/'dom_fixture.json').write_text(json.dumps(p.elements,ensure_ascii=False),encoding='utf-8')
 script=re.search(r'<script>(.*?)</script>',(PACK/'index.html').read_text(encoding='utf-8'),re.S).group(1)
 (AUDIT/'guide.js').write_text(script,encoding='utf-8')
 subprocess.run([str(NODE),'--check',str(AUDIT/'guide.js')],check=True)
 subprocess.run([str(NODE),str(Path(__file__).with_name('test_logic.cjs')),str(AUDIT)],check=True)
 zip_path=OUT/'CV_Trajectory_Reading_Guide.zip'
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in sorted(PACK.rglob('*')):
   if f.is_file():z.write(f,f.relative_to(PACK.parent).as_posix())
 relocated=AUDIT/'portable_roundtrip'
 with zipfile.ZipFile(zip_path) as z:
  assert z.testzip() is None
  assert all(n.startswith(PACK.name+'/') and '..' not in Path(n).parts for n in z.namelist())
  z.extractall(relocated)
 recovered=relocated/PACK.name
 for f in PACK.rglob('*'):
  if f.is_file():assert sha(f)==sha(recovered/f.relative_to(PACK))
 check_html(recovered)
 result=dict(papers=len(catalog),main_readings=14,original_local_sources_unchanged=local_sources,
  pdf_pages=sum(c['pages'] for c in catalog),**links,archive_bytes=zip_path.stat().st_size,
  archive_sha256=sha(zip_path),archive_integrity=True,relocated_file_hashes_match=True,
  js_syntax=True,logic_tests=json.loads((AUDIT/'logic_results.json').read_text()),
  browser_visual_review='Not performed: browser policy rejected local file URL; no workaround attempted.')
 (AUDIT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
