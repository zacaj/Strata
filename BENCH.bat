@echo off
rem Quick Strata speed check (the old BENCH.bat), for the 0.2.16 layout. Run from this folder: your strata-*.json configs are here.
rem Options: --clients 4, --long, --compare <older results folder>, --config FILE, --url http://127.0.0.1:8080
rem With no --config or --url it uses the newest strata-*.json in this folder.
setlocal
set "PYTHONUTF8=1"
title Strata - benchmark
cd /d "%~dp0"
if not exist "terminal\.venv\Scripts\python.exe" (
  echo  Strata is not set up here yet: run START-HERE.bat first.
  pause
  exit /b 1
)
set "PICK="
echo %* | findstr /i /c:"--config" /c:"--url" >nul
if errorlevel 1 (
  for /f "delims=" %%F in ('dir /b /o-d "strata-*.json" ^| findstr /v /i "shared-settings"') do if not defined PICK set "PICK=%%F"
)
if defined PICK (
  echo  Using %PICK%
  "terminal\.venv\Scripts\python.exe" terminal\tools\strata_bench.py --config "%CD%\%PICK%" %*
) else (
  "terminal\.venv\Scripts\python.exe" terminal\tools\strata_bench.py %*
)
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
