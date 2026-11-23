@echo off
rem Runs the Terminal's `strata` command from this source checkout: strata drafting fetch, strata launch --status, ...
rem (the installer's shim is not on the PATH in a checkout; the Terminal and its .venv live in terminal\)
setlocal
set "PYTHONUTF8=1"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo The Terminal's Python environment is not set up yet: run START-HERE.bat once.
  exit /b 1
)
"%PY%" "%~dp0strata.py" %*
exit /b %errorlevel%
