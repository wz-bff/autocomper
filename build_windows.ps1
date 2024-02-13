$EnvPath="env_windows"

if (-not (Test-Path -Path $EnvPath -PathType Container)) {
    python -m venv $EnvPath
}

if (-not ($env:VIRTUAL_ENV)) {
    & .\$EnvPath\Scripts\Activate.ps1
}

pip install -r requirements.txt
python setup.py build
deactivate
Write-Host -NoNewLine 'Press any key to continue...';
[void][System.Console]::ReadKey($true)