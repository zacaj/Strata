@echo off
title Strata coder-IQ1_M
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-coder-iq1_m.json" "--port" "8080" "--host" "0.0.0.0"
if errorlevel 1 pause
