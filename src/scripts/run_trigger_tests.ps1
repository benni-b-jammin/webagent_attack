$ErrorActionPreference = "Stop"

# Move to repo root (script stored in src/scripts/)
Set-Location (Join-Path $PSScriptRoot "..\..")

$demoDir = "src/config/demo_runs"
$triggerDir = "src/data/triggers/3-results"
$promptDir = "src/data/test_prompts"

# Optional logging
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile = ".\results\run_trigger_tests_$timestamp.txt"

function Run-And-Log {
    param(
        [string]$Command
    )

    Write-Host ">> $Command"
    ">> $Command" | Out-File -FilePath $logFile -Append

    $tempFile = [System.IO.Path]::GetTempFileName()
    try {
        cmd /c "$Command > `"$tempFile`" 2>&1"
        Get-Content $tempFile | Tee-Object -FilePath $logFile -Append

        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Remove-Item $tempFile -ErrorAction SilentlyContinue
    }
}

# Get all demo YAMLs except ones containing "default"
$yamlFiles = Get-ChildItem -Path $demoDir -File |
    Where-Object {
        $_.Extension -in ".yaml", ".yml" -and $_.BaseName -notmatch "default"
    }

foreach ($yaml in $yamlFiles) {
    # Example: demo_google_translate.yaml -> google_translate
    $rootName = $yaml.BaseName -replace '^demo_', ''

    Write-Host "`n=== Processing $rootName ===" | Tee-Object -FilePath $logFile -Append

    $promptFile = Join-Path $promptDir "$rootName.txt"
    if (-not (Test-Path $promptFile)) {
        Write-Host "Missing prompt file: $promptFile" | Tee-Object -FilePath $logFile -Append
        continue
    }

    # Find most recent trigger JSON beginning with rootName_
    $triggerFile = Get-ChildItem -Path $triggerDir -File -Filter "$rootName*.json" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $triggerFile) {
        Write-Host "Missing trigger file for root: $rootName" | Tee-Object -FilePath $logFile -Append
        continue
    }

    $prompts = Get-Content $promptFile | Where-Object { $_.Trim() -ne "" }

    foreach ($prompt in $prompts) {
        Write-Host "Prompt: $prompt" | Tee-Object -FilePath $logFile -Append

        Run-And-Log "python3 -m src.attacks.run_demo --config `"$($yaml.FullName)`" --trigger_path `"$($triggerFile.FullName)`" --goal `"$prompt`""
    }
}

Write-Host "Done." | Tee-Object -FilePath $logFile -Append