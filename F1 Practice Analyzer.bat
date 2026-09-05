@echo off
rem Double-click launcher for the F1 Practice Analyzer marimo app (port 2728).
rem If the server is already up it just opens the browser; otherwise it
rem starts the server in a minimized window (close that window to stop it).

netstat -ano | findstr ":2728" | findstr "LISTENING" >nul
if %errorlevel%==0 goto open

start "F1 Practice Analyzer (marimo) - close this window to stop the app" /min py -m marimo run "%~dp0f1_practice_app.py" --headless --port 2728 --no-token --watch
timeout /t 4 /nobreak >nul

:open
start http://localhost:2728
