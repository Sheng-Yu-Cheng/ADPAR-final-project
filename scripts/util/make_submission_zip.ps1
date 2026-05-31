param(
    [string]$OutputDir = "dist",
    [string]$SubmissionDirName = "submission",
    [string]$ZipName = "ADPAR-final-project-submission.zip"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..\..")
$OutputRoot = Join-Path $Root $OutputDir
$SubmissionDir = Join-Path $OutputRoot $SubmissionDirName
$LuaDir = Join-Path $SubmissionDir "lua"
$ZipPath = Join-Path $OutputRoot $ZipName

$MainSource = Join-Path $Root "scripts\tester\live_event_classifier_v5b_packet_loss_safe.py"
$DroneSource = Join-Path $Root "scripts\tester\drone-control.py"
$ReadmeSource = Join-Path $Root "README.md"
$RequirementSource = Join-Path $Root "requirement.txt"
$SlidesSource = Join-Path $Root "slides.pdf"
$LuaSource = Join-Path $Root "scripts\radar"

$RequiredPaths = @(
    $MainSource,
    $DroneSource,
    $ReadmeSource,
    $RequirementSource,
    $SlidesSource,
    $LuaSource
)

foreach ($path in $RequiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required source path: $path"
    }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

if (Test-Path -LiteralPath $SubmissionDir) {
    Remove-Item -LiteralPath $SubmissionDir -Recurse -Force
}

New-Item -ItemType Directory -Path $LuaDir -Force | Out-Null

Copy-Item -LiteralPath $MainSource -Destination (Join-Path $SubmissionDir "main.py") -Force
Copy-Item -LiteralPath $DroneSource -Destination (Join-Path $SubmissionDir "drone-control.py") -Force
Copy-Item -LiteralPath $ReadmeSource -Destination (Join-Path $SubmissionDir "README.md") -Force
Copy-Item -LiteralPath $RequirementSource -Destination (Join-Path $SubmissionDir "requirement.txt") -Force
Copy-Item -LiteralPath $SlidesSource -Destination (Join-Path $SubmissionDir "slides.pdf") -Force
Copy-Item -Path (Join-Path $LuaSource "*.lua") -Destination $LuaDir -Force

$MainPath = Join-Path $SubmissionDir "main.py"
$MainText = Get-Content -LiteralPath $MainPath -Raw
$MainText = [regex]::Replace(
    $MainText,
    'OUT_PATH = Path\(\s*r?"[^"]*Raw_0\.bin"\s*\)',
    "OUT_PATH = Path(`r`n    `"adc/live_event_classifier_v5b_packet_loss_safe_Raw_0.bin`"`r`n)"
)
Set-Content -LiteralPath $MainPath -Value $MainText -NoNewline

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

Compress-Archive -Path (Join-Path $SubmissionDir "*") -DestinationPath $ZipPath -Force

Write-Host "Refreshed submission folder:"
Write-Host $SubmissionDir
Write-Host ""
Write-Host "Created submission zip:"
Write-Host $ZipPath
Write-Host ""
Write-Host "Included:"
Write-Host " - main.py"
Write-Host " - README.md"
Write-Host " - drone-control.py"
Write-Host " - requirement.txt"
Write-Host " - slides.pdf"
Write-Host " - lua/"
