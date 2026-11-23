@echo off
title Strata qwen36-Q8_0
cd /d "C:\programs\Strata"
<<<<<<< Updated upstream
=======
set STRATA_Q36_SPEC_SAMPLED=1
>>>>>>> Stashed changes
"C:\programs\Strata\terminal\.venv\Scripts\python.exe" "C:\programs\Strata\terminal\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\terminal\strata-qwen36-q8_0.json" "--port" "8080" "--open"
if errorlevel 1 pause
