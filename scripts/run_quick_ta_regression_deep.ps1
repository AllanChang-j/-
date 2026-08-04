param(
    [string]$ConfigPath = "config/quick_ta_regression_deep.yaml"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    powershell -ExecutionPolicy Bypass -File scripts/lab_one_click_level1.ps1 -ConfigPath $ConfigPath -RunTests 1
}
else {
    .\.venv\Scripts\python.exe main.py --config $ConfigPath
}
