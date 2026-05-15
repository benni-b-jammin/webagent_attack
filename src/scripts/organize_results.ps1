$ErrorActionPreference = "Stop"

# Move to repo root if script is stored in src/scripts/
Set-Location (Join-Path $PSScriptRoot "..\..")

$thirdResultsDir = "results\3-results"
$baselineDir = "results\0-baseline"

New-Item -ItemType Directory -Force -Path $thirdResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $baselineDir | Out-Null

$triggerDirs = @(
    "results\2026-04-17_11-37-55_demo_run",
    "results\2026-04-17_11-38-50_demo_run",
    "results\2026-04-17_11-39-35_demo_run",
    "results\2026-04-17_11-40-21_demo_run",
    "results\2026-04-17_11-41-10_demo_run",
    "results\2026-04-17_11-41-55_demo_run",
    "results\2026-04-17_11-42-56_demo_run",
    "results\2026-04-17_11-43-41_demo_run",
    "results\2026-04-17_11-44-40_demo_run",
    "results\2026-04-17_11-45-24_demo_run",

    "results\2026-04-17_11-22-51_demo_run",
    "results\2026-04-17_11-24-09_demo_run",
    "results\2026-04-17_11-24-56_demo_run",
    "results\2026-04-17_11-25-42_demo_run",
    "results\2026-04-17_11-26-33_demo_run",
    "results\2026-04-17_11-33-53_demo_run",
    "results\2026-04-17_11-34-36_demo_run",
    "results\2026-04-17_11-35-26_demo_run",
    "results\2026-04-17_11-36-15_demo_run",
    "results\2026-04-17_11-37-05_demo_run",

    "results\2026-04-17_12-16-03_demo_run",
    "results\2026-04-17_12-19-56_demo_run",
    "results\2026-04-17_12-22-02_demo_run",
    "results\2026-04-17_12-23-45_demo_run",
    "results\2026-04-17_12-25-15_demo_run",
    "results\2026-04-17_12-29-38_demo_run",
    "results\2026-04-17_12-31-21_demo_run",
    "results\2026-04-17_12-36-00_demo_run",
    "results\2026-04-17_12-38-29_demo_run",
    "results\2026-04-17_12-40-05_demo_run",

    "results\2026-04-17_11-46-24_demo_run",
    "results\2026-04-17_11-47-24_demo_run",
    "results\2026-04-17_11-48-26_demo_run",
    "results\2026-04-17_11-49-22_demo_run",
    "results\2026-04-17_11-50-29_demo_run",
    "results\2026-04-17_11-51-26_demo_run",
    "results\2026-04-17_11-52-26_demo_run",
    "results\2026-04-17_11-53-32_demo_run",
    "results\2026-04-17_11-54-33_demo_run",
    "results\2026-04-17_11-55-34_demo_run",

    "results\2026-04-17_11-56-36_demo_run",
    "results\2026-04-17_11-57-29_demo_run",
    "results\2026-04-17_11-58-21_demo_run",
    "results\2026-04-17_11-59-04_demo_run",
    "results\2026-04-17_11-59-48_demo_run",
    "results\2026-04-17_12-00-41_demo_run",
    "results\2026-04-17_12-01-26_demo_run",
    "results\2026-04-17_12-02-12_demo_run",
    "results\2026-04-17_12-03-05_demo_run",
    "results\2026-04-17_12-03-56_demo_run"
)

$refDirs = @(
    "results\2026-04-16_19-53-00_demo_run",
    "results\2026-04-07_19-45-29_demo_run",
    "results\2026-04-07_20-05-48_demo_run",
    "results\2026-04-16_15-36-16_demo_run",
    "results\2026-04-17_12-33-41_demo_run",
    "results\2026-04-16_20-19-28_demo_run",
    "results\2026-04-09_22-56-20_demo_run",
    "results\2026-04-16_14-56-54_demo_run",
    "results\2026-04-16_15-06-58_demo_run",
    "results\2026-04-16_15-14-40_demo_run",
    "results\2026-04-16_15-17-21_demo_run"
)

Write-Host "Moving trigger result folders to 3-results..."
foreach ($dir in $triggerDirs) {
    if (Test-Path $dir) {
        Move-Item -Path $dir -Destination $thirdResultsDir
        Write-Host "Moved: $dir -> $thirdResultsDir"
    }
    else {
        Write-Host "Missing: $dir"
    }
}

Write-Host "Moving baseline/ref folders to 0-baseline..."
foreach ($dir in $refDirs) {
    if (Test-Path $dir) {
        Move-Item -Path $dir -Destination $baselineDir
        Write-Host "Moved: $dir -> $baselineDir"
    }
    else {
        Write-Host "Missing: $dir"
    }
}

Write-Host "Done."