"""Run inference and CPU rendering concurrently, then audit and package."""
import subprocess,sys,time,traceback
from pathlib import Path
from selection import ROOT,OUT,read,write,digest
import render
render.cv2.setNumThreads(2)
def main():
 code=[Path(__file__).parent/n for n in ['run.py','attention.py','selection.py']]
 lock=OUT/'frozen_code.json'
 hashes={str(p.relative_to(ROOT)):digest(p) for p in code}
 if lock.exists():assert read(lock)==hashes
 else:write(lock,hashes)
 selected=read(OUT/'selection.json');rendered=set();begin=time.time()
 parts=[selected[:96],selected[96:]]
 workers=[]
 for index,part in enumerate(parts):
  handle=(OUT/f'inference_worker_{index+1}.log').open('a',encoding='utf8')
  command=[sys.executable,'-u',str(Path(__file__).with_name('shard.py')),'--clips']+[c['id'] for c in part]
  workers.append((subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT),handle,part))
 try:
  while True:
   for c in selected:
    if c['id'] not in rendered and (OUT/c['id']/'complete.json').exists():
     render.render_clip(c);rendered.add(c['id'])
     write(OUT/'pipeline_progress.json',dict(rendered=len(rendered),total=len(selected),elapsed_seconds=time.time()-begin))
     print('Rendered',c['id'],len(rendered),'/',len(selected),flush=True)
   if all(w[0].poll() is not None for w in workers):
    failures=[w for w in workers if w[0].returncode]
    if failures:
     # Resource failures retry after parallel workers finish, using the original
     # serial runner and its unchanged computation, skipping completed arms.
     write(OUT/'parallel_retry.json',dict(exit_codes=[w[0].returncode for w in workers]))
     for _,handle,part in failures:
      command=[sys.executable,'-u',str(Path(__file__).with_name('run.py')),'--clips']+[c['id'] for c in part]
      subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
     workers=[]
    if len(rendered)==len(selected):break
   time.sleep(2)
  for _,handle,_ in workers:handle.close()
  write(OUT/'inference_complete.json',dict(clips=len(selected),arms=render.ARMS))
  import finish
  finish.main()
 except Exception:
  write(OUT/'pipeline_error.json',dict(error=traceback.format_exc(),workers=[dict(pid=w[0].pid,returncode=w[0].poll()) for w in workers]))
  raise
if __name__=='__main__':main()
