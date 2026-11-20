@echo off

title Strata prompt-logger :3002 -^> :8080

cd /d "C:\programs\Strata\prompt-logger"

set LLAMA_BACKEND_HOST=127.0.0.1
set LLAMA_BACKEND_PORT=8080
set PROXY_PORT=3002
set LOG_FILTERED=true
set LOG_FULL_BODY=true
set LOG_RESPONSES=true
set PYTHONUTF8=1

"C:\programs\Strata\.venv\Scripts\python.exe" -u proxy.py

if errorlevel 1 pause
