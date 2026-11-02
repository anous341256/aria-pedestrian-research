"""Reproduce the complete pipeline, including exact physical finalization."""
from pathlib import Path
import subprocess,sys
folder=Path(__file__).resolve().parent
for script in ['preflight.py','experiment.py','physical_inference.py','audit.py','render.py','finalize_delivery.py']:
    subprocess.run([sys.executable,str(folder/script)],check=True,cwd=folder.parents[1])
