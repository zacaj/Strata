@echo off
title Strata qwen36-tqp
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-qwen36-tqp.json" "--port" "8080" 
if errorlevel 1 pause
