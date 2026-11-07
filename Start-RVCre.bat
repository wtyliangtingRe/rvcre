@echo off
setlocal
title RVCre - Local Voice Studio
cd /d "%~dp0"

rem Change this line if Miniconda is installed somewhere else.
set "RVCreCondaRoot=D:\EnglishOnly\miniconda"
call "%RVCreCondaRoot%\condabin\conda.bat" activate rvcre
if errorlevel 1 (
  echo Could not activate the rvcre Conda environment.
  echo Check RVCreCondaRoot in this file.
  pause
  exit /b 1
)

echo Starting RVCre. Keep this window open while using the app.
python -m rvcre ui %*
set "RVCreExitCode=%ERRORLEVEL%"
if not "%RVCreExitCode%"=="0" pause
exit /b %RVCreExitCode%
