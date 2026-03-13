Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildOutput = Join-Path $repoRoot "win32\Release\bin"
$artifactRoot = Join-Path $repoRoot "artifacts\windows-x64"
$artifactZip = Join-Path $repoRoot "artifacts\eviacam-windows-x64.zip"
$wxDllRoot = if ($env:WXWIN) { Join-Path $env:WXWIN "lib\vc14x_x64_dll" } else { $null }
$opencvBinRoot = if ($env:CVPATH) { Join-Path $env:CVPATH "build\x64\vc15\bin" } else { $null }

if (-not (Test-Path (Join-Path $buildOutput "eviacam.exe"))) {
    throw "Expected build output not found: $buildOutput\eviacam.exe"
}

if (Test-Path $artifactRoot) {
    Remove-Item -Path $artifactRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
Copy-Item -Path (Join-Path $buildOutput "*") -Destination $artifactRoot -Recurse -Force

if ($wxDllRoot -and (Test-Path $wxDllRoot)) {
    Copy-Item -Path (Join-Path $wxDllRoot "wx*.dll") -Destination $artifactRoot -Force
}

if ($opencvBinRoot -and (Test-Path $opencvBinRoot)) {
    Copy-Item -Path (Join-Path $opencvBinRoot "opencv_world460.dll") -Destination $artifactRoot -Force -ErrorAction SilentlyContinue
}

$requiredFiles = @(
    "eviacam.exe",
    "wxbase32u_vc14x_x64.dll",
    "wxmsw32u_core_vc14x_x64.dll",
    "opencv_world460.dll",
    "haarcascade_frontalface_default.xml",
    "face_detection_yunet_2023mar.onnx",
    "mediapipe_face_mesh_backend.py"
)

$missingFiles = @($requiredFiles | Where-Object { -not (Test-Path (Join-Path $artifactRoot $_)) })
if ($missingFiles.Count -gt 0) {
    throw "Packaged Windows artifact is missing required files: $($missingFiles -join ', ')"
}

if (Test-Path $artifactZip) {
    Remove-Item -Path $artifactZip -Force
}

Compress-Archive -Path (Join-Path $artifactRoot "*") -DestinationPath $artifactZip

Write-Host "Packaged artifact: $artifactZip"
"ARTIFACT_ZIP=$artifactZip" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
