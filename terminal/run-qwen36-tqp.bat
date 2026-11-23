@echo off
title Strata qwen36-tqp
cd /d "C:\programs\Strata"
set STRATA_Q36_SPEC_SAMPLED=1
"C:\programs\Strata\terminal\.venv\Scripts\python.exe" "C:\programs\Strata\terminal\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\terminal\strata-qwen36-tqp.json" "--port" "8080" 
if errorlevel 1 pause
