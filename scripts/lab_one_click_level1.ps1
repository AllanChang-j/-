param(
    [string]$ConfigPath = "config/level1_hardware_full_comparison.yaml",
    [string]$DataPath = "data/raw/taiwan_daily_ohlcv_20240101_20260630.csv",
    [string]$DataStart = "2024-01-01",
    [string]$DataEnd = "2026-06-30",
    [string]$VenvDir = ".venv",
    [string]$CudaTorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$InstallCudaTorch = "auto",
    [string]$CollectIfMissing = "1",
    [string]$RunTests = "1"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

function Find-Python {
    $python = Get-Command py -ErrorAction SilentlyContinue
    if ($python) {
        return @("py", "-3")
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }
    $python = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python) {
        return @("python3")
    }
    return $null
}

function Invoke-Python {
    param(
        [string[]]$PythonCommand,
        [string[]]$Arguments
    )
    $extraArgs = @()
    if ($PythonCommand.Length -gt 1) {
        $extraArgs = $PythonCommand[1..($PythonCommand.Length - 1)]
    }
    & $PythonCommand[0] @extraArgs @Arguments
}

function Install-Python-WithWinget {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python was not found, and winget is not available. Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/ and rerun this script."
    }
    Write-Step "Python not found; installing Python 3.12 with winget"
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$LogDir = "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("level1_one_click_windows_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $LogPath -Append | Out-Null

try {
    Write-Step "Project root: $ProjectRoot"
    Write-Step "Log path: $LogPath"
    Write-Step "Config: $ConfigPath"
    Write-Step "Data: $DataPath"

    $pythonCommand = Find-Python
    if (-not $pythonCommand) {
        Install-Python-WithWinget
        $pythonCommand = Find-Python
    }
    if (-not $pythonCommand) {
        throw "Python installation finished but Python still cannot be found in this PowerShell session. Open a new PowerShell window and rerun the command."
    }

    Write-Step "Creating virtual environment"
    Invoke-Python $pythonCommand @("-m", "venv", $VenvDir)

    $VenvPython = Join-Path $ProjectRoot "$VenvDir\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment Python not found: $VenvPython"
    }

    New-Item -ItemType Directory -Force -Path "data/raw" | Out-Null
    New-Item -ItemType Directory -Force -Path "experiments" | Out-Null

    Write-Step "Upgrading pip tooling"
    & $VenvPython -m pip install --upgrade pip setuptools wheel

    $HasNvidia = $false
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $HasNvidia = $true
    }
    if (($InstallCudaTorch -eq "1") -or (($InstallCudaTorch -eq "auto") -and $HasNvidia)) {
        Write-Step "Installing CUDA PyTorch from $CudaTorchIndexUrl"
        & $VenvPython -m pip install --upgrade torch torchvision torchaudio --index-url $CudaTorchIndexUrl
    }
    else {
        Write-Step "Skipping explicit CUDA PyTorch install"
    }

    Write-Step "Installing research dependencies"
    & $VenvPython -m pip install -r requirements-research.txt

    Write-Step "Environment report"
    & $VenvPython scripts/verify_lab_environment.py

    if (-not (Test-Path $DataPath)) {
        if ($CollectIfMissing -ne "1") {
            throw "Missing $DataPath. Copy the CSV into data/raw or rerun with -CollectIfMissing 1."
        }
        Write-Step "Data file missing; collecting official daily history $DataStart to $DataEnd"
        & $VenvPython data/collect_stage1_history.py `
            --start $DataStart `
            --end $DataEnd `
            --output $DataPath `
            --strict-network
    }
    else {
        Write-Step "Data file exists; skipping collection"
    }

    Write-Step "Validating daily data file"
    $env:DATA_PATH = $DataPath
    & $VenvPython -c "import os; from data.loaders import load_daily_csv; df=load_daily_csv(os.environ['DATA_PATH']); print('rows', len(df)); print('date_min', df['date'].min().date()); print('date_max', df['date'].max().date()); print('symbols', df['symbol'].nunique()); print('markets', df['market'].value_counts(dropna=False).to_dict())"

    if ($RunTests -eq "1") {
        Write-Step "Running unit safety tests"
        & $VenvPython -m pytest tests
    }

    Write-Step "Starting Level 1 training and final test"
    & $VenvPython main.py --config $ConfigPath

    Write-Step "Level 1 run completed"
    Write-Step "Key outputs"
    $env:CONFIG_PATH = $ConfigPath
    & $VenvPython -c "import os; from pathlib import Path; from utils.config import load_config; cfg=load_config(os.environ['CONFIG_PATH']); out=Path(cfg['experiment']['output_dir']); paths=['reports/final_comparison_report.xlsx','reports/test_comparison.csv','reports/backtest_comparison.csv','reports/cross_validation_summary.csv','reports/best_model_summary.json']; [print(out / p, 'exists' if (out / p).exists() else 'missing') for p in paths]"
}
finally {
    Stop-Transcript | Out-Null
}
