@echo off
chcp 65001 >nul
title Official CVAT SAM2 Tracker
wsl.exe -d Ubuntu-24.04 -- /opt/cvat-sam2/bin/python /mnt/e/New_research/scripts/cvat_sam2/connect.py
echo.
echo Tracker launcher exited. Existing CVAT annotations were not deleted.
pause
