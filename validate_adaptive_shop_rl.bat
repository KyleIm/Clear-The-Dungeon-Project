@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project .venv is missing. See docs\adaptive_shop_v2.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m adaptive_shop.validate %*
set "rl_status=%ERRORLEVEL%"
if not "%rl_status%"=="0" pause
exit /b %rl_status%
