@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_aria_mps_batch.ps1" %*
set "ARIA_BATCH_EXIT=%ERRORLEVEL%"
if /I "%~1"=="-CheckOnly" exit /b %ARIA_BATCH_EXIT%
echo.
echo Keep the official MPS window open until all requested results are downloaded.
echo Do not launch another eye-gaze batch while this one is running.
pause
exit /b %ARIA_BATCH_EXIT%
