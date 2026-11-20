@echo off
title Strata qwen36-opus-distill-q4kxl
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-qwen36-opus-distill-q4kxl.json" "--port" "8080" "--open"
if errorlevel 1 pause
