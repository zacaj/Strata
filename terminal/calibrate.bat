@echo off
rem Tunes a config's engine settings on this PC and prints the result (changes nothing).
rem Stop the server first (llm stop): it needs the GPU.
rem   calibrate.bat strata-inkling-2-mini-ud-iq3_xxs.json
if "%~1"=="" (
  echo usage: calibrate.bat strata-NAME.json
  dir /b "%~dp0strata-*.json"
  exit /b 1
)
title Strata calibrate %~n1
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0tools\calibrate.py" "%~f1"
pause
