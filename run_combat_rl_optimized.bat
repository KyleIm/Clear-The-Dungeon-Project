@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project .venv is missing. See docs\combat_reward_performance.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m combat_rl.runner train --config combat_rl\training_optimized.json %*
set "rl_status=%ERRORLEVEL%"
if not "%rl_status%"=="0" pause
exit /b %rl_status%
