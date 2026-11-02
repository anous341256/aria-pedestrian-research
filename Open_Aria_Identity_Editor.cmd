@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
"E:\New_research\.conda-envs\phase1-py310\Scripts\python.exe" "%~dp0scripts\aria_identity_editor\server.py" --data "%~dp0outputs\aria_identity_editor\segments_v2" --port 8767 --open-browser
if errorlevel 1 pause
