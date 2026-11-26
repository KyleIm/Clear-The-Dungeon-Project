@echo off
setlocal
pushd "%~dp0" || goto path_error

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_COMMAND=.venv\Scripts\python.exe"
) else (
    set "PYTHON_COMMAND=py -3.12"
)
%PYTHON_COMMAND% -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 with Matplotlib is required.
    echo Install baseline dependencies with: .venv\Scripts\python.exe -m pip install -r experiments\requirements-baseline.txt
    set "RESULT=1"
    goto finish
)

%PYTHON_COMMAND% -m experiments.random_baseline_large %*
set "RESULT=%ERRORLEVEL%"

:finish
popd
if not "%RESULT%"=="0" (
    echo Baseline runner ended with exit code %RESULT%.
    pause
) else if "%~1"=="" (
    pause
)
exit /b %RESULT%

:path_error
echo Could not open the project directory.
pause
exit /b 1
