$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { python -m venv .venv }
if ($LASTEXITCODE -ne 0) { throw 'Python virtual environment setup failed.' }
& './.venv/Scripts/python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& './.venv/Scripts/python.exe' -m ml.fetch_assets
if ($LASTEXITCODE -ne 0) { throw 'Model download or manifest preparation failed.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
