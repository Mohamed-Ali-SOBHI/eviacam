param(
    [string]$Toolchain = (Join-Path $env:USERPROFILE 'Documents\eviacam-msys64\ucrt64'),
    [string]$Destination = (Join-Path $env:USERPROFILE 'Documents\eViacam-YuNet')
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$bin = Join-Path $Toolchain 'bin'
New-Item -ItemType Directory -Force $Destination | Out-Null
$Destination = (Resolve-Path $Destination).Path
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($name in 'eviacam.exe','detector_tests.exe','pipeline_tests.exe','face_benchmark.exe') {
    Copy-Item -LiteralPath (Join-Path $repo "build\$name") -Destination $Destination -Force
    $queue.Enqueue((Join-Path $Destination $name))
}
$seen = @{}
while ($queue.Count) {
    $file = $queue.Dequeue()
    $imports = & (Join-Path $bin 'objdump.exe') -p $file
    foreach ($line in $imports) {
        if ($line -match 'DLL Name:\s*(\S+)') {
            $name = $Matches[1]
            if ($seen.ContainsKey($name.ToLowerInvariant())) { continue }
            $seen[$name.ToLowerInvariant()] = $true
            $source = Join-Path $repo ".deps\opencv-install\x64\mingw\bin\$name"
            if (-not (Test-Path -LiteralPath $source)) { $source = Join-Path $bin $name }
            if (Test-Path -LiteralPath $source) {
                Copy-Item -LiteralPath $source -Destination $Destination -Force
                $queue.Enqueue((Join-Path $Destination $name))
            } elseif (-not (Test-Path -LiteralPath (Join-Path $env:SystemRoot "System32\$name")) -and $name -notlike 'api-ms-*') {
                throw "Missing dependency: $name ($file)"
            }
        }
    }
}
Copy-Item -Path (Join-Path $repo 'src\data\*') -Destination $Destination -Force
Copy-Item -Path (Join-Path $repo 'doc\*.png'),(Join-Path $repo 'doc\*.wav') -Destination $Destination -Force
Copy-Item -LiteralPath (Join-Path $repo 'THANKS') -Destination $Destination -Force
Copy-Item -LiteralPath (Join-Path $repo 'src\resources') -Destination $Destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $repo 'models\face_detection_yunet_2023mar.onnx') -Destination $Destination -Force
Copy-Item -LiteralPath (Join-Path $repo 'models\LICENSE-YuNet') -Destination $Destination -Force
Copy-Item -LiteralPath (Join-Path $repo '.deps\opencv-4.10.0\LICENSE') -Destination (Join-Path $Destination 'LICENSE-OpenCV.txt') -Force
Copy-Item -LiteralPath (Join-Path $repo 'COPYING') -Destination (Join-Path $Destination 'LICENSE-eViacam.txt') -Force
Copy-Item -LiteralPath (Join-Path $repo 'tests\generated\replay.avi') -Destination $Destination -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launch-trial.ps1') -Destination $Destination -Force
Copy-Item -LiteralPath (Join-Path $Toolchain 'share\licenses') -Destination $Destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $repo '.deps\opencv-install\etc\licenses') -Destination (Join-Path $Destination 'licenses-opencv') -Recurse -Force
# Reuse installed translation/help files, read-only. No installed executable or DLL is copied.
$installed = Join-Path ${env:ProgramFiles(x86)} 'Enable Viacam\bin'
foreach ($name in 'fr','help') {
    if (Test-Path -LiteralPath (Join-Path $installed $name)) {
        Copy-Item -LiteralPath (Join-Path $installed $name) -Destination $Destination -Recurse -Force
    }
}

# Snapshot the current user's settings once; the original registry is never written.
$configPath = Join-Path $Destination 'portable.ini'
if (-not (Test-Path -LiteralPath $configPath)) {
    $lines = [System.Collections.Generic.List[string]]::new()
    $registryRoot = 'HKCU:\Software\eviacam'
    if (Test-Path -LiteralPath "$registryRoot\settings") {
        $keys = @(Get-Item -LiteralPath "$registryRoot\settings") + @(Get-ChildItem -LiteralPath "$registryRoot\settings" -Recurse)
        foreach ($key in $keys) {
            $section = $key.Name.Substring('HKEY_CURRENT_USER\Software\eviacam\'.Length).Replace('\','/')
            $lines.Add("[$section]")
            foreach ($name in $key.GetValueNames()) {
                if (-not $name) { continue }
                $value = [string]$key.GetValue($name)
                if ($name -eq 'checkUpdatesAtStartup' -or $name -eq 'runWizardAtStartup') { $value = '0' }
                if ($name -eq 'enabledAtStartup') { $value = '1' }
                $value = $value.Replace('\','\\').Replace("`r",'\r').Replace("`n",'\n')
                $lines.Add("$name=$value")
            }
            $lines.Add('')
        }
    } else {
        $lines.Add("[settings]`ncurrentProfile=default`ncheckUpdatesAtStartup=0`n[settings/default]`nenabledAtStartup=1`nrunWizardAtStartup=0")
    }
    [IO.File]::WriteAllLines($configPath, $lines, [Text.UTF8Encoding]::new($false))
    Copy-Item -LiteralPath $configPath -Destination (Join-Path $Destination 'settings-before-test.ini')
}
[IO.File]::WriteAllText((Join-Path $Destination 'demo.ini'), @'
[settings]
currentProfile=default
cameraName=Offline video (no camera)
checkUpdatesAtStartup=0
[settings/default]
enabledAtStartup=0
runWizardAtStartup=0
minimisedAtStartup=0
[settings/default/motionTracker]
trackFace=1
enableWhenFaceDetected=1
locateFaceTimeout=1000
threadPeriod=100
'@, [Text.UTF8Encoding]::new($false))
& (Join-Path $PSScriptRoot 'write-launchers.ps1') -Destination $Destination
Write-Output "Portable package: $Destination"
Write-Output "Bundled $($seen.Count) resolved import names (including system DLLs)."
