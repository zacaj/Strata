@echo off
title Strata qwen36-UD-Q6_K_XL
cd /d "C:\programs\Strata"
<<<<<<< Updated upstream
=======
set STRATA_Q36_SPEC_SAMPLED=1
>>>>>>> Stashed changes
"C:\programs\Strata\terminal\.venv\Scripts\python.exe" "C:\programs\Strata\terminal\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\terminal\strata-qwen36-ud-q6_k_xl.json" "--port" "8080" "--open"
if errorlevel 1 pause
