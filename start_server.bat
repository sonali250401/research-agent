@echo off
cd /d "%~dp0"
"C:\Users\SONALI\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8080 --reload
if errorlevel 1 pause
