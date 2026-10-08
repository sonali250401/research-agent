$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$python = "C:\Users\SONALI\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python 3.11 was not found at $python. Install Python 3.11 or update start_server.ps1 with the correct path."
}

& $python -m uvicorn api:app --host 0.0.0.0 --port 8000
