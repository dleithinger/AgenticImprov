@echo off
setlocal
cd /d "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if not exist ".venv\Scripts\python.exe" goto failed
)

".venv\Scripts\python.exe" -c "from importlib.metadata import version; from packaging.version import Version; assert Version('1.52') <= Version(version('streamlit')) < Version('2')" >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)

echo Open http://localhost:8501 in your browser. Press Ctrl+C here to shut down.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo AgenticImprov could not start. Check the errors above and the setup steps in README.md.
pause
exit /b 1
