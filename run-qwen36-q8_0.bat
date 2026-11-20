@echo off
title Strata qwen36-Q8_0
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-qwen36-q8_0.json" "--port" "8080" "--open"
if errorlevel 1 pause
