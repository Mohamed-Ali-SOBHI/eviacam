Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildOutput = Join-Path $repoRoot "win32\Release\bin"
$artifactRoot = Join-Path $repoRoot "artifacts\windows-x64"
$artifactZip = Join-Path $repoRoot "artifacts\eviacam-windows-x64.zip"

if (-not (Test-Path (Join-Path $buildOutput "eviacam.exe"))) {
    throw "Expected build output not found: $buildOutput\eviacam.exe"
}

if (Test-Path $artifactRoot) {
    Remove-Item -Path $artifactRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
Copy-Item -Path (Join-Path $buildOutput "*") -Destination $artifactRoot -Recurse -Force

if (Test-Path $artifactZip) {
    Remove-Item -Path $artifactZip -Force
}

Compress-Archive -Path (Join-Path $artifactRoot "*") -DestinationPath $artifactZip

Write-Host "Packaged artifact: $artifactZip"
"ARTIFACT_ZIP=$artifactZip" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
