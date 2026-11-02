@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_aria_full_viewer.ps1" %*
if errorlevel 1 pause
