@echo off
setlocal
cd /d "%~dp0"
"E:\New_research\.conda-envs\phase1-py310\Scripts\python.exe" "%~dp0scripts\rotation_match_review\serve.py" --data "%~dp0outputs\aria_rotation_match_review\v1" --open-browser
if errorlevel 1 pause
