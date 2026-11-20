@echo off

title Strata deepseek-v4.1-flash-IST-DAS-1.75bpw

cd /d "C:\programs\Strata"

"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-deepseek-v4.1-flash-ist-das-1.75bpw.json" "--port" "8080" "--host" "0.0.0.0"

if errorlevel 1 pause

