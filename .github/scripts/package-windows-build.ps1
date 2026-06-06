Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$buildOutput = Join-Path $repoRoot "win32\Release\bin"
$version = if ($env:EVIACAM_VERSION) { $env:EVIACAM_VERSION } else { "2.1.4.1" }
$packageName = "eviacam-$version-windows-x64"
$artifactsDir = Join-Path $repoRoot "artifacts"
$packageRoot = Join-Path $artifactsDir $packageName
$artifactZip = Join-Path $artifactsDir "$packageName.zip"
$artifactChecksum = "$artifactZip.sha256"
$wxDllRoot = if ($env:WXWIN) { Join-Path $env:WXWIN "lib\vc14x_x64_dll" } else { $null }
$opencvBinRoot = if ($env:CVPATH) { Join-Path $env:CVPATH "build\x64\vc15\bin" } else { $null }
$ortBinRoot = if ($env:ORT_ROOT) { Join-Path $env:ORT_ROOT "lib" } else { $null }

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string] $Source,
        [Parameter(Mandatory = $true)][string] $Destination
    )

    if (-not (Test-Path $Source)) {
        throw "Required distribution file not found: $Source"
    }

    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Find-VcRuntimeRoot {
    if ($env:VCToolsRedistDir) {
        $candidate = Join-Path $env:VCToolsRedistDir "x64\Microsoft.VC143.CRT"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return $null
    }

    $installationPath = & $vswhere -latest -products * -property installationPath
    if (-not $installationPath) {
        return $null
    }

    $redistRoot = Join-Path $installationPath "VC\Redist\MSVC"
    $versionDir = Get-ChildItem -Path $redistRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if (-not $versionDir) {
        return $null
    }

    $candidate = Join-Path $versionDir.FullName "x64\Microsoft.VC143.CRT"
    if (Test-Path $candidate) {
        return $candidate
    }

    return $null
}

if (-not (Test-Path (Join-Path $buildOutput "eviacam.exe"))) {
    throw "Expected build output not found: $buildOutput\eviacam.exe"
}

New-Item -ItemType Directory -Force -Path $artifactsDir | Out-Null
if (Test-Path $packageRoot) {
    Remove-Item -LiteralPath $packageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

Copy-Item -Path (Join-Path $buildOutput "*") -Destination $packageRoot -Recurse -Force
Get-ChildItem -Path $packageRoot -File |
    Where-Object {
        $_.Extension -in ".exp", ".lib", ".pdb" -or
        $_.Name -match "^wx.*ud_.*\.dll$"
    } |
    Remove-Item -Force

if (-not $wxDllRoot -or -not (Test-Path $wxDllRoot)) {
    throw "wxWidgets runtime directory not found."
}
Get-ChildItem -Path $wxDllRoot -Filter "wx*.dll" |
    Where-Object { $_.Name -notmatch "^wx.*ud_.*\.dll$" } |
    Copy-Item -Destination $packageRoot -Force

Copy-RequiredFile -Source (Join-Path $opencvBinRoot "opencv_world460.dll") -Destination $packageRoot
Copy-RequiredFile -Source (Join-Path $ortBinRoot "onnxruntime.dll") -Destination $packageRoot
Copy-RequiredFile -Source (Join-Path $ortBinRoot "onnxruntime_providers_shared.dll") -Destination $packageRoot

$vcRuntimeRoot = Find-VcRuntimeRoot
if (-not $vcRuntimeRoot) {
    throw "Microsoft Visual C++ x64 application-local runtime was not found."
}
$vcRuntimeFiles = @(
    "concrt140.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll"
)
foreach ($runtimeFile in $vcRuntimeFiles) {
    Copy-RequiredFile -Source (Join-Path $vcRuntimeRoot $runtimeFile) -Destination $packageRoot
}

Copy-RequiredFile -Source (Join-Path $repoRoot "COPYING") -Destination (Join-Path $packageRoot "LICENSE.txt")
Copy-RequiredFile -Source (Join-Path $repoRoot "win32\RELEASE-NOTES-WINDOWS.txt") -Destination (Join-Path $packageRoot "RELEASE-NOTES.txt")
Copy-RequiredFile -Source (Join-Path $repoRoot "win32\README-WINDOWS.txt") -Destination $packageRoot

$thirdPartyDir = Join-Path $packageRoot "THIRD-PARTY-LICENSES"
New-Item -ItemType Directory -Force -Path $thirdPartyDir | Out-Null

$opencvLicense = Join-Path $env:CVPATH "LICENSE"
if (-not (Test-Path $opencvLicense)) {
    $opencvLicense = Join-Path $repoRoot "doc\opencvlicense.txt"
}
Copy-RequiredFile -Source $opencvLicense -Destination (Join-Path $thirdPartyDir "OpenCV.txt")

$wxLicense = Join-Path $env:WXWIN "docs\licence.txt"
Copy-RequiredFile -Source $wxLicense -Destination (Join-Path $thirdPartyDir "wxWidgets.txt")

$ortLicense = Join-Path $env:ORT_ROOT "LICENSE"
Copy-RequiredFile -Source $ortLicense -Destination (Join-Path $thirdPartyDir "ONNX-Runtime.txt")

$requiredFiles = @(
    "eviacam.exe",
    "wxbase32u_vc14x_x64.dll",
    "wxmsw32u_core_vc14x_x64.dll",
    "opencv_world460.dll",
    "onnxruntime.dll",
    "vcruntime140.dll",
    "msvcp140.dll",
    "haarcascade_frontalface_default.xml",
    "face_detection_back_256x256.onnx",
    "face_mesh_192x192.onnx",
    "LICENSE.txt",
    "README-WINDOWS.txt"
)

$missingFiles = @($requiredFiles | Where-Object { -not (Test-Path (Join-Path $packageRoot $_)) })
if ($missingFiles.Count -gt 0) {
    throw "Packaged Windows artifact is missing required files: $($missingFiles -join ', ')"
}

$buildInfo = @(
    "Enable Viacam Windows distribution"
    "Version: $version"
    "Architecture: x64"
    "Git commit: $env:GITHUB_SHA"
    "GitHub run: $env:GITHUB_SERVER_URL/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"
)
$buildInfo | Set-Content -LiteralPath (Join-Path $packageRoot "BUILD-INFO.txt") -Encoding utf8

$manifestPath = Join-Path $packageRoot "SHA256SUMS.txt"
Get-ChildItem -Path $packageRoot -Recurse -File |
    Where-Object { $_.FullName -ne $manifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        $relativePath = $_.FullName.Substring($packageRoot.Length + 1).Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash *$relativePath"
    } |
    Set-Content -LiteralPath $manifestPath -Encoding ascii

if (Test-Path $artifactZip) {
    Remove-Item -LiteralPath $artifactZip -Force
}
Compress-Archive -Path $packageRoot -DestinationPath $artifactZip -CompressionLevel Optimal

$zipHash = (Get-FileHash -LiteralPath $artifactZip -Algorithm SHA256).Hash.ToLowerInvariant()
"$zipHash *$(Split-Path $artifactZip -Leaf)" |
    Set-Content -LiteralPath $artifactChecksum -Encoding ascii

Write-Host "Packaged portable distribution: $artifactZip"
"ARTIFACT_DIR=$artifactsDir" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
"PACKAGE_ROOT=$packageRoot" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
"PORTABLE_ZIP=$artifactZip" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
"PORTABLE_SHA256=$artifactChecksum" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
