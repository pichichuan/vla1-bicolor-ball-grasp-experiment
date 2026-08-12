@echo off
setlocal
cd /d "%~dp0"
python .\src\stage8_gui.py
if errorlevel 1 pause
