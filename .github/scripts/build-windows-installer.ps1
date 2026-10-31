Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$version = if ($env:EVIACAM_VERSION) { $env:EVIACAM_VERSION } else { "2.1.4.1" }
$packageRoot = $env:PACKAGE_ROOT
$artifactsDir = $env:ARTIFACT_DIR

if (-not $packageRoot -or -not (Test-Path $packageRoot)) {
    throw "Portable package directory not found: $packageRoot"
}

$iscc = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
if (-not $iscc) {
    $knownPaths = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $isccPath = $knownPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $isccPath) {
        throw "Inno Setup 6 compiler was not found."
    }
}
else {
    $isccPath = $iscc.Source
}

$installerScript = Join-Path $repoRoot "win32\installer-windows-x64.iss"
& $isccPath `
    "/DAppVersion=$version" `
    "/DPackageRoot=$packageRoot" `
    "/DOutputDir=$artifactsDir" `
    $installerScript

$installer = Join-Path $artifactsDir "eviacam-$version-windows-x64-setup.exe"
if (-not (Test-Path $installer)) {
    throw "Expected installer was not generated: $installer"
}

$installerChecksum = "$installer.sha256"
$installerHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
"$installerHash *$(Split-Path $installer -Leaf)" |
    Set-Content -LiteralPath $installerChecksum -Encoding ascii

Write-Host "Packaged installer: $installer"
"INSTALLER_EXE=$installer" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
"INSTALLER_SHA256=$installerChecksum" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
