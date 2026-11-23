@echo off
title Strata swift-gyro-s
cd /d "C:\programs\Strata"
"C:\programs\Strata\terminal\.venv\Scripts\python.exe" "C:\programs\Strata\terminal\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\terminal\strata-swift-gyro-s.json" "--port" "8080" "--host" "0.0.0.0"
if errorlevel 1 pause
