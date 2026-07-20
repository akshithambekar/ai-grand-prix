param(
    [string]$SimulatorPath = "",
    [switch]$SkipSimulator,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$venvRoot = Join-Path $repoRoot ".venv_win"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

function Find-NativePython {
    $py = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($py) {
        return [pscustomobject]@{
            Executable = $py.Source
            Arguments = @("-3")
        }
    }

    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($python) {
        return [pscustomobject]@{
            Executable = $python.Source
            Arguments = @()
        }
    }

    throw "Python 3.11+ was not found. Install the Windows build from python.org and rerun this script."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = Find-NativePython
    $pythonExe = $launcher.Executable
    $pythonArgs = @($launcher.Arguments)
    $pythonArgs += @("-m", "venv", $venvRoot)
    & $pythonExe @pythonArgs

    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "The selected Python created a non-native environment. Install the Windows build from python.org; do not use MSYS/Git Bash Python."
    }
}

& $venvPython -c "import aigp_pilot, numpy, pymavlink" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing pilot dependencies..."
    & $venvPython -m pip install -e $repoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
}

if (-not $SkipSimulator) {
    if (-not $SimulatorPath) {
        $candidates = @(
            (Join-Path (Split-Path $repoRoot -Parent) "AIGP_VQ1_3385\FlightSim.exe"),
            (Join-Path $repoRoot "AIGP_VQ1_3385\FlightSim.exe"),
            (Join-Path $repoRoot "FlightSim.exe")
        )
        $SimulatorPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    }

    if (-not $SimulatorPath -or -not (Test-Path -LiteralPath $SimulatorPath)) {
        throw "VQ1 FlightSim.exe was not found. Extract the simulator beside this repo or pass -SimulatorPath."
    }

    $running = Get-Process -Name "DCGame-Win64-Shipping" -ErrorAction SilentlyContinue
    if (-not $running) {
        Write-Host "Starting VQ1 simulator..."
        Start-Process -FilePath $SimulatorPath -WorkingDirectory (Split-Path $SimulatorPath -Parent)
    } else {
        Write-Host "VQ1 simulator is already running."
    }
}

if ($CheckOnly) {
    Write-Host "VQ1 pilot environment is ready."
    exit 0
}

Write-Host "Starting AI-GP pilot. Enter VQ1 Round 1 and start the countdown when ready."
Push-Location $repoRoot
try {
    & $venvPython "fly.py"
} finally {
    Pop-Location
}
