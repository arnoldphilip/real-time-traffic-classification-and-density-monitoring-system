@echo off
title REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
python launcher.py
pause
