@echo off
title Strata darwin-180b-rsi-r3-gsq-rco-iq3_xxs
cd /d "C:\programs\Strata"
"C:\programs\Strata\.venv\Scripts\python.exe" "C:\programs\Strata\serve\server.py" "--engine" "strata" "--config" "C:\programs\Strata\strata-darwin-180b-rsi-r3-gsq-rco-iq3_xxs.json" "--port" "8080" "--host" "0.0.0.0"
if errorlevel 1 pause
