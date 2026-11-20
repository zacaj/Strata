@echo off
title Strata darwin-r3-gyro-m
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-darwin-r3-gyro-m.json" "--port" "8080" "--host" "0.0.0.0"
if errorlevel 1 pause
