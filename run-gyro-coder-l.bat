@echo off
title Strata gyro-coder-l
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-gyro-coder-l.json" "--port" "8080" "--host" "0.0.0.0"
if errorlevel 1 pause
