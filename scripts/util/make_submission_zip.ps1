param(
    [string]$OutputDir = "dist",
    [string]$PackageName = "ADPAR-final-project-submission",
    [switch]$IncludeAdc
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outputRoot = Join-Path $Root $OutputDir
$stagingName = "$PackageName-$timestamp"
$stagingDir = Join-Path $outputRoot $stagingName
$zipPath = "$stagingDir.zip"

if (Test-Path $stagingDir) {
    Remove-Item -LiteralPath $stagingDir -Recurse -Force
}

New-Item -ItemType Directory -Path $stagingDir | Out-Null

$excludeDirs = @(".git", ".venv", "dist", "__pycache__")
if (-not $IncludeAdc) {
    $excludeDirs += "adc"
}

$excludeFileNames = @('~$slides.pptx')
$excludeExtensions = @(".pyc", ".pyo", ".zip")

function Should-SkipDirectory {
    param([System.IO.DirectoryInfo]$Dir)
    return $excludeDirs -contains $Dir.Name
}

function Should-SkipFile {
    param([System.IO.FileInfo]$File)

    if ($excludeFileNames -contains $File.Name) {
        return $true
    }

    if ($excludeExtensions -contains $File.Extension.ToLowerInvariant()) {
        return $true
    }

    return $false
}

function Copy-ProjectItem {
    param(
        [string]$Source,
        [string]$Destination
    )

    $item = Get-Item -LiteralPath $Source -Force

    if ($item.PSIsContainer) {
        if (Should-SkipDirectory $item) {
            return
        }

        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
        foreach ($child in Get-ChildItem -LiteralPath $item.FullName -Force) {
            Copy-ProjectItem -Source $child.FullName -Destination (Join-Path $Destination $child.Name)
        }
    }
    else {
        if (Should-SkipFile $item) {
            return
        }

        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Force
    }
}

foreach ($item in Get-ChildItem -LiteralPath $Root -Force) {
    Copy-ProjectItem -Source $item.FullName -Destination (Join-Path $stagingDir $item.Name)
}

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath -Force
Remove-Item -LiteralPath $stagingDir -Recurse -Force

Write-Host "Created submission package:"
Write-Host $zipPath

if (-not $IncludeAdc) {
    Write-Host "Note: adc/ was excluded. Run with -IncludeAdc to include raw captures."
}
