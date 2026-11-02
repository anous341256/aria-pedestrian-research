from bulk_common import *
import subprocess

def main():
 while not (OUT/'association_complete.json').exists():time.sleep(10)
 for name in ['export_bulk.py','final_audit.py','render_bulk.py']:
  write(OUT/'postprocess_progress.json',dict(stage=name));subprocess.run([sys.executable,str(Path(__file__).with_name(name))],check=True)
 while not (OUT/'source_audit.json').exists():time.sleep(10)
 subprocess.run([sys.executable,str(Path(__file__).with_name('finish_bulk.py'))],check=True)
if __name__=='__main__':main()
