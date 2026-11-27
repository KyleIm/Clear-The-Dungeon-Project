@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project .venv is missing. See docs\alternating_combat_shop_rl.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m unittest discover -s tests -p test_alternating_rl_*.py -v %*
set "rl_status=%ERRORLEVEL%"
if not "%rl_status%"=="0" pause
exit /b %rl_status%
