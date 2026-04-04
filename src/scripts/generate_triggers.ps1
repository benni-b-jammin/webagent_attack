$ErrorActionPreference = "Stop"

# Move to repo root (script is stored in src/)
Set-Location (Join-Path $PSScriptRoot "..\..\.")

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
Start-Transcript -Path ".\results\generate_triggers_log_$timestamp.txt"

Write-Host "Running dataset capture..."
python3 -m src.attacks.capture_dataset

Write-Host "Running trigger generation..."

$files = Get-ChildItem -Path "src/config/narrow_triggers" -File |
    Where-Object { $_.Extension -in ".yaml", ".yml", ".json" }

Write-Host "Found $($files.Count) config files"
$files | ForEach-Object { Write-Host $_.FullName }

foreach ($file in $files) {
    if ($file.Name -notmatch "default") {
        Write-Host "Using config: $($file.FullName)"

        1..3 | ForEach-Object {
            python3 -m src.attacks.make_trigger --algo gcg --config "$($file.FullName)"
        }
    }
    else {
        Write-Host "Skipping default config: $($file.FullName)"
    }
}

Write-Host "Done."