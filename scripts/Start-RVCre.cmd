@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo RVCre's local Python environment is not ready.
  echo Run from the project folder: powershell -NoProfile -File .\scripts\Setup-RVCre.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import rvcre, gradio" >nul 2>&1
if errorlevel 1 (
  echo RVCre or the UI dependencies are missing. Run scripts\Setup-RVCre.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m rvcre ui --workspace data --config config\backends.json %*
set "RVCreExitCode=%ERRORLEVEL%"
if not "%RVCreExitCode%"=="0" pause
exit /b %RVCreExitCode%
