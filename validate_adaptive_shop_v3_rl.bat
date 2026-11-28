@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project .venv is missing. See docs\adaptive_shop_v3.md.
    exit /b 1
)
".venv\Scripts\python.exe" -m adaptive_shop_v3.validate %*
exit /b %ERRORLEVEL%
