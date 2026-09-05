#!/bin/bash
# Mac double-click launcher for the F1 Practice Analyzer marimo app (port 2728
# — do NOT use marimo's default 2718). Opens the browser if the server is
# already up, otherwise starts it first from the project's own .venv.
cd "$(dirname "$0")"
if ! lsof -i :2728 -sTCP:LISTEN >/dev/null 2>&1; then
    nohup .venv/bin/python -m marimo run f1_practice_app.py --headless \
        --port 2728 --no-token --watch >/dev/null 2>&1 &
    sleep 4
fi
open http://localhost:2728
