@echo off
title ATCC Traffic Classification and Counting
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"
python launcher.py
pause
