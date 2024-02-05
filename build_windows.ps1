if (-not (Test-Path -Path ".env/" -PathType Container)) {
    python -m venv .env
    .\.env\Scripts\Activate.ps1
}
elseif (-not ($env:VIRTUAL_ENV)) {
    .\.env\Scripts\Activate.ps1
}

pip install -r requirements.txt
pyinstaller --add-data ".\bdetectionmodel_05_01_23.onnx;." --add-data "ffmpeg;ffmpeg" --collect-data sv_ttk --add-data "img;img" --noconfirm --paths .\.env\ --noconsole --icon ".\app.ico" --onedir --contents-directory . --name AutoComper autocomper.py
deactivate
Write-Host -NoNewLine 'Press any key to continue...';
[void][System.Console]::ReadKey($true)