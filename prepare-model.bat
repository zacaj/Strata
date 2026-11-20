@echo off
rem Packs a GGUF, writes strata-<name>.json, then offers the expert profile and calibration.
rem Usage: prepare-model.bat "C:\models\model-00001-of-00002.gguf" [iq_pack options, e.g. --compat-bf16]
rem It asks for a short name (suggesting one). With no arguments it also asks for the path.
cd /d "C:\programs\Strata"
if "%~1"=="" (
  set /p GGUF=GGUF path ^(shard 1^): 
  call ".venv\Scripts\python.exe" prepare-model.py "%%GGUF:"=%%"
) else (
  ".venv\Scripts\python.exe" prepare-model.py %*
)
pause
