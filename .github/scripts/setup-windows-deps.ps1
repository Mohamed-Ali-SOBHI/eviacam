Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$depsRoot = Join-Path $repoRoot "deps"
$downloadsRoot = Join-Path $depsRoot "downloads"
$extractRoot = Join-Path $depsRoot "extract"
$wxExtractRoot = Join-Path $extractRoot "wxWidgets"

New-Item -ItemType Directory -Force -Path $downloadsRoot, $extractRoot | Out-Null

$sevenZip = Get-Command 7z -ErrorAction Stop

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string] $Url,
        [Parameter(Mandatory = $true)][string] $Destination
    )

    if (Test-Path $Destination) {
        Write-Host "Using cached download: $Destination"
        return
    }

    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination
}

function Find-OpenCvRoot {
    param(
        [Parameter(Mandatory = $true)][string] $SearchRoot
    )

    if (Test-Path (Join-Path $SearchRoot "build\include\opencv2")) {
        return $SearchRoot
    }

    $candidate = Get-ChildItem -Path $SearchRoot -Directory -Recurse |
        Where-Object { Test-Path (Join-Path $_.FullName "build\include\opencv2") } |
        Select-Object -First 1

    if (-not $candidate) {
        throw "Unable to locate an extracted OpenCV root under $SearchRoot"
    }

    return $candidate.FullName
}

function Find-WxRoot {
    param(
        [Parameter(Mandatory = $true)][string] $SearchRoot
    )

    $candidate = Get-ChildItem -Path $SearchRoot -Directory -Recurse |
        Where-Object {
            (Test-Path (Join-Path $_.FullName "include\wx\version.h")) -and
            (Test-Path (Join-Path $_.FullName "lib\vc_x64_dll")) -and
            (Test-Path (Join-Path $_.FullName "locale"))
        } |
        Select-Object -First 1

    if (-not $candidate) {
        throw "Unable to locate an extracted wxWidgets root under $SearchRoot"
    }

    return $candidate.FullName
}

$wxVersion = "3.2.10"
$wxTag = "v3.2.10"
$wxSourceArchive = Join-Path $downloadsRoot "wxWidgets-${wxVersion}.zip"
$wxDevArchive = Join-Path $downloadsRoot "wxMSW-${wxVersion}_vc14x_x64_Dev.7z"
$wxReleaseDllArchive = Join-Path $downloadsRoot "wxMSW-${wxVersion}_vc14x_x64_ReleaseDLL.7z"

Download-File -Url "https://github.com/wxWidgets/wxWidgets/releases/download/$wxTag/wxWidgets-${wxVersion}.zip" -Destination $wxSourceArchive
Download-File -Url "https://github.com/wxWidgets/wxWidgets/releases/download/$wxTag/wxMSW-${wxVersion}_vc14x_x64_Dev.7z" -Destination $wxDevArchive
Download-File -Url "https://github.com/wxWidgets/wxWidgets/releases/download/$wxTag/wxMSW-${wxVersion}_vc14x_x64_ReleaseDLL.7z" -Destination $wxReleaseDllArchive

if (Test-Path $wxExtractRoot) {
    Remove-Item -Path $wxExtractRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $wxExtractRoot | Out-Null

Expand-Archive -Path $wxSourceArchive -DestinationPath $wxExtractRoot -Force
& $sevenZip.Source x $wxDevArchive "-o$wxExtractRoot" -y | Out-Null
& $sevenZip.Source x $wxReleaseDllArchive "-o$wxExtractRoot" -y | Out-Null

$wxRoot = Find-WxRoot -SearchRoot $wxExtractRoot

$opencvVersion = "4.6.0"
$opencvArchive = Join-Path $downloadsRoot "opencv-${opencvVersion}-vc14_vc15.exe"
$opencvExtractRoot = Join-Path $extractRoot "opencv"

Download-File -Url "https://github.com/opencv/opencv/releases/download/$opencvVersion/opencv-${opencvVersion}-vc14_vc15.exe" -Destination $opencvArchive

if (Test-Path $opencvExtractRoot) {
    Remove-Item -Path $opencvExtractRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $opencvExtractRoot | Out-Null
& $sevenZip.Source x $opencvArchive "-o$opencvExtractRoot" -y | Out-Null

$opencvRoot = Find-OpenCvRoot -SearchRoot $opencvExtractRoot

Write-Host "WXWIN=$wxRoot"
Write-Host "CVPATH=$opencvRoot"

"WXWIN=$wxRoot" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
"CVPATH=$opencvRoot" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
