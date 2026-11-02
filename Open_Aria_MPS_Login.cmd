@echo off
setlocal
title Project Aria MPS - Login only
if not "%~2"=="" goto USAGE
if "%~1"=="" goto LOGIN
if /I "%~1"=="--check" goto CHECK
goto USAGE

:CHECK
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_aria_mps_login.ps1" -CheckOnly
exit /b %ERRORLEVEL%

:LOGIN
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_aria_mps_login.ps1"
set "ARIA_MPS_LOGIN_EXIT=%ERRORLEVEL%"
echo.
echo This window is for login only. No recordings were submitted by this launcher.
pause
exit /b %ARIA_MPS_LOGIN_EXIT%

:USAGE
echo Usage: Open_Aria_MPS_Login.cmd [--check]
echo Do not pass an account name, password, or recording path to this launcher.
exit /b 2
