@echo off
title Strata qwen36-UD-Q6_K_XL
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-qwen36-ud-q6_k_xl.json" "--port" "8080" "--open"
if errorlevel 1 pause
