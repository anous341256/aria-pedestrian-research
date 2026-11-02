"""Import existing history as today's snapshot, not fabricated old commits."""
from pathlib import Path
import json, hashlib, subprocess, shutil, time, os
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'experiment_registry/history_import_20261029'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 DEST.mkdir(parents=True,exist_ok=True);versions=[];weights=[];files=[];start=time.monotonic()
 # Source and result files remain in their existing tracked paths. Copy small
 # protocols from ignored output trees, preserving provenance and exact bytes.
 names=sorted({p.name for base in ['scripts','plans','reports','outputs'] for p in (ROOT/base).iterdir() if p.is_dir() and p.name!='__pycache__'})
 for name in names:
  versions.append({'version':name,'locations':[str((ROOT/base/name).relative_to(ROOT)) for base in ['scripts','reports','outputs'] if (ROOT/base/name).exists()]})
 for base in ['outputs','models','runs','output','baselines']:
  d=ROOT/base
  if not d.exists():continue
  if shutil.which('rg'):
   listing=subprocess.run(['rg','--files','--hidden','--no-ignore',str(d)],capture_output=True,text=True,check=False).stdout.splitlines()
  else:
   listing=[]
   for folder,dirs,fnames in os.walk(d):
    dirs[:]=[x for x in dirs if x not in {'.git','__pycache__','node_modules'}]
    listing.extend(str(Path(folder)/x) for x in fnames)
  for line in listing:
   p=Path(line);rel=p.relative_to(ROOT);size=p.stat().st_size
   if p.suffix in {'.pt','.pth','.ckpt','.safetensors'}:
    weights.append({'path':str(rel),'bytes':size,'sha256':digest(p)})
   copy_code=p.suffix in {'.py','.sh','.ps1','.yaml','.yml','.toml'} and '__pycache__' not in p.parts
   copy_protocol=p.suffix=='.json' and len(rel.parts)==3 and size<=10*1024*1024 and any(s in p.name.lower() for s in ['protocol','fold','sample','metadata','manifest','hash','inventory','selection','comparison','verification','complete','metric','config','environment','preflight'])
   if copy_code or copy_protocol:
    target=DEST/'snapshots'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    files.append({'source':str(rel),'snapshot':str(target.relative_to(ROOT)),'bytes':size,'sha256':digest(p)})
  print(base,'weights',len(weights),'snapshots',len(files),flush=True)
 write(DEST/'versions.json',versions);write(DEST/'weight_manifest.json',weights);write(DEST/'snapshot_manifest.json',files)
 upstream=[]
 for d in ROOT.iterdir():
  if d.is_dir() and d.name!='.git' and (d/'.git').exists():
   head=subprocess.run(['git','-c',f'safe.directory={d}','-C',str(d),'rev-parse','HEAD'],capture_output=True,text=True)
   status=subprocess.run(['git','-c',f'safe.directory={d}','-C',str(d),'status','--porcelain'],capture_output=True,text=True)
   upstream.append({'path':str(d.relative_to(ROOT)),'head':head.stdout.strip(),'status':status.stdout.splitlines(),'error':head.stderr.strip()})
 write(DEST/'upstream_repositories.json',upstream)
 write(DEST/'import_summary.json',{'import_date':'2026-10-29','historical_versions':len(versions),'copied_small_artifacts':len(files),'weight_files_hashed':len(weights),'weight_bytes':sum(w['bytes'] for w in weights),'snapshot_bytes':sum(w['bytes'] for w in files),'seconds':time.monotonic()-start,'note':'Present-day import of existing historical directories. Binary weights remain external; manifest tracks exact bytes. Raw data/videos are external and are referenced by the archived experiment source/delivery manifests. This repository alone is not a full data backup.'})
 print(json.dumps(json.loads((DEST/'import_summary.json').read_text()),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
