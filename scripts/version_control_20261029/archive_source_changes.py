"""Save local changes to upstream repositories and older root code projects."""
from pathlib import Path
import os,json,subprocess,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'experiment_registry/history_import_20261029'
GIT='/mnt/c/Program Files/Git/cmd/git.exe'
def win(p):return 'E:/New_research/'+str(p.relative_to(ROOT))
def call(d,*args):return subprocess.run([GIT,'-c',f'safe.directory={win(d)}','-c','core.filemode=false','-C',win(d),*args],capture_output=True,check=True).stdout
def save(p,b):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
def main():
 repos=[];roots=[]
 for d in ROOT.iterdir():
  if d.is_dir() and d.name!='.git' and (d/'.git').exists():roots.append(d)
 for base in ['external','third_party','tools']:
  if not (ROOT/base).exists():continue
  for f,dirs,files in os.walk(ROOT/base):
   dirs[:]=[n for n in dirs if n not in {'node_modules','__pycache__','.venv','data','datasets','models'}]
   if '.git' in dirs or '.git' in files:roots.append(Path(f));dirs[:]=[]
 suffix={'.py','.ps1','.sh','.cmd','.bat','.yaml','.yml','.toml','.cpp','.h','.c','.json','.ini','.cfg','.md','.txt','.ipynb'}
 saved=[];skipped=[]
 for d in sorted(set(roots)):
  rel=d.relative_to(ROOT);label=str(rel).replace('/','__');head=call(d,'rev-parse','HEAD').decode().strip();patch=call(d,'diff','HEAD','--binary');pp=DEST/'upstream_patches'/f'{label}.patch';save(pp,patch)
  untracked=call(d,'ls-files','--others','--exclude-standard','-z').decode('utf-8').split('\0');files=[]
  for x in filter(None,untracked):
   p=d/x
   if not p.is_file():continue
   if p.suffix in suffix and p.stat().st_size<2*1024*1024:
    target=DEST/'upstream_untracked'/rel/x;save(target,p.read_bytes());files.append(str(p.relative_to(ROOT)))
   else:
    item={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size}
    if p.suffix in {'.pt','.pth','.ckpt','.safetensors'}:
     h=hashlib.sha256()
     with p.open('rb') as stream:
      for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
     item['sha256']=h.hexdigest()
    skipped.append(item)
  repos.append({'path':str(rel),'HEAD':head,'patch':str(pp.relative_to(ROOT)),'patch_bytes':len(patch),'untracked_sources':files})
 for base in ['deploy','sgnet_deploy','SGNet','TrajMamba-R0','WeatherFeatureAlign','WeatherFeatureAlign+','work','baselines']:
  d=ROOT/base
  if not d.exists() or (d/'.git').exists():continue
  for f,dirs,files in os.walk(d):
   dirs[:]=[n for n in dirs if n not in {'.git','__pycache__','node_modules','data','datasets','outputs','models','runs','.venv'}]
   for x in files:
    p=Path(f)/x
    if p.suffix in suffix and p.stat().st_size<2*1024*1024 and not any(t in x.lower() for t in ['credential','password','secret','token']):
     target=DEST/'legacy_source'/p.relative_to(ROOT);b=p.read_bytes();save(target,b);saved.append({'source':str(p.relative_to(ROOT)),'snapshot':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest()})
 obj={'upstream_repositories':repos,'legacy_source_files':saved,'untracked_non_source_external_files':skipped,'note':'Upstream repositories are not modified. Restore their recorded HEAD and apply archived patches; untracked source copies are separate. Large untracked assets remain external.'};save(DEST/'additional_source_manifest.json',json.dumps(obj,ensure_ascii=False,indent=2).encode());print({'upstream_repos':len(repos),'legacy_source_files':len(saved),'untracked_sources':sum(len(x['untracked_sources']) for x in repos),'patch_bytes':sum(x['patch_bytes'] for x in repos),'external_non_source':len(skipped)},flush=True)
if __name__=='__main__':main()
