[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$UseDocker,
    [string]$ImageName = "nophigene:latest",
    [string]$ContainerName = "nophigene-ui",
    [switch]$SkipBuild,
    [switch]$EnableLocalExtraction,
    [switch]$NoOpenBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$dockerStartScript = Join-Path $scriptDir "start_nophigene_ui.ps1"
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$dataDir = Join-Path $repoRoot "data"
$referenceDir = Join-Path $dataDir "reference\hg38"
$extractedDir = Join-Path $dataDir "extracted"
$resultsDir = Join-Path $repoRoot "results"
$pidFile = Join-Path $repoRoot ".nophigene-ui.pid"
$stdoutLog = Join-Path $repoRoot ".nophigene-ui.log"
$stderrLog = Join-Path $repoRoot ".nophigene-ui.err.log"

if ($UseDocker) {
    Write-Host ""
    Write-Host "NophiGene local launcher is delegating to Docker for Extraction support." -ForegroundColor Green
    Write-Host "Docker image will include samtools and bcftools when rebuilt from the current Dockerfile."
    Write-Host ""

    $dockerParams = @{
        Port = $Port
        ImageName = $ImageName
        ContainerName = $ContainerName
    }
    if ($SkipBuild) { $dockerParams.SkipBuild = $true }
    if ($NoOpenBrowser) { $dockerParams.NoOpenBrowser = $true }
    if ($DryRun) { $dockerParams.DryRun = $true }

    & $dockerStartScript @dockerParams
    exit $LASTEXITCODE
}

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Directories {
    foreach ($dir in @($dataDir, $referenceDir, $extractedDir, $resultsDir)) {
        if (-not (Test-Path $dir)) {
            Write-Step "Creating $dir"
            if (-not $DryRun) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
        }
    }
}

function Test-CommandAvailable {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Configure-LocalExtraction {
    $envOverrideEnabled = $env:NOPHIGENE_ENABLE_LOCAL_EXTRACTION -eq "1"
    if (-not $EnableLocalExtraction -and -not $envOverrideEnabled) {
        Write-Host "Extraction  : disabled in local mode; use Docker for BAM extraction." -ForegroundColor Yellow
        return
    }

    $missingTools = @()
    foreach ($toolName in @("samtools", "bcftools")) {
        if (-not (Test-CommandAvailable -Name $toolName)) {
            $missingTools += $toolName
        }
    }

    if ($missingTools.Count -gt 0) {
        throw "Local extraction was requested, but these tools were not found on PATH: $($missingTools -join ', '). Use the Docker launcher or install the tools locally."
    }

    $env:NOPHIGENE_ENABLE_LOCAL_EXTRACTION = "1"
    Write-Host "Extraction  : enabled for this local process via NOPHIGENE_ENABLE_LOCAL_EXTRACTION=1." -ForegroundColor Green
}

function Test-WebReady {
    param([int]$WebPort)

    if ($DryRun) {
        return $true
    }

    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$WebPort" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    }
    catch {
        return $false
    }
}

function Wait-ForWebApp {
    param([int]$WebPort)

    if ($DryRun) {
        Write-Step "Would wait for http://127.0.0.1:$WebPort to respond"
        return
    }

    Write-Step "Waiting for the local web UI to respond on port $WebPort"
    $deadline = (Get-Date).AddMinutes(2)
    while ((Get-Date) -lt $deadline) {
        if (Test-WebReady -WebPort $WebPort) {
            return
        }
        Start-Sleep -Seconds 2
    }

    throw "The local UI did not respond on http://127.0.0.1:$WebPort within 2 minutes. Check $stdoutLog and $stderrLog."
}

function Get-TrackedProcess {
    if (-not (Test-Path $pidFile)) {
        return $null
    }

    $rawPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $rawPid) {
        return $null
    }

    try {
        return Get-Process -Id ([int]$rawPid) -ErrorAction Stop
    }
    catch {
        return $null
    }
}

Write-Host ""
Write-Host "NophiGene local launcher" -ForegroundColor Green
Write-Host "Repository : $repoRoot"
Write-Host "Python     : $pythonPath"
Write-Host "Port       : $Port"
Write-Host "Reference  : $referenceDir"
Write-Host "Extracted  : $extractedDir"
Write-Host ""

if (-not (Test-Path $pythonPath)) {
    throw "No local Python environment was found at $pythonPath. Recreate .venv first."
}

Ensure-Directories
Configure-LocalExtraction

$existing = Get-TrackedProcess
if ($existing) {
    if (Test-WebReady -WebPort $Port) {
        Write-Step "A tracked UI process is already running with PID $($existing.Id)"
        if (-not $NoOpenBrowser) {
            $url = "http://127.0.0.1:$Port"
            Write-Step "Opening $url"
            if (-not $DryRun) {
                Start-Process $url | Out-Null
            }
        }
        exit 0
    }

    Write-Step "A stale tracked UI process was found. Restarting it."
    if (-not $DryRun) {
        Stop-Process -Id $existing.Id -Force -ErrorAction SilentlyContinue
    }
}
elseif (Test-Path $pidFile) {
    Remove-Item -LiteralPath $pidFile -Force
}

Write-Step "Checking local app dependencies"
if (-not $DryRun) {
    & $pythonPath -c "import flask, allel, pandas, methylprep"
    if ($LASTEXITCODE -ne 0) {
        throw "The local environment is missing app dependencies. Install requirements-app.txt into .venv first."
    }
}

Write-Step "Starting the local web UI"
if (-not $DryRun) {
    if (Test-Path $stdoutLog) { Remove-Item -LiteralPath $stdoutLog -Force }
    if (Test-Path $stderrLog) { Remove-Item -LiteralPath $stderrLog -Force }

    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("src/app.py", "web", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru

    Set-Content -LiteralPath $pidFile -Value $process.Id -NoNewline
}

Wait-ForWebApp -WebPort $Port

if (-not $NoOpenBrowser) {
    $url = "http://127.0.0.1:$Port"
    Write-Step "Opening $url"
    if (-not $DryRun) {
        Start-Process $url | Out-Null
    }
}

Write-Host ""
Write-Host "Local UI is ready." -ForegroundColor Green
Write-Host "Open http://127.0.0.1:$Port if the browser did not appear automatically."
Write-Host "BAM extraction remains Docker-first unless local samtools/bcftools are explicitly enabled."
Write-Host "Use 'Stop NophiGene UI.cmd' to stop the local server later."
