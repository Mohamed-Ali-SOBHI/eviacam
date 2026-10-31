param([string]$Toolchain = (Join-Path $env:USERPROFILE 'Documents\eviacam-msys64\ucrt64'))
$ErrorActionPreference = 'Stop'
$bin = Join-Path $Toolchain 'bin'
$env:PATH = "$bin;$PSScriptRoot/../.deps/opencv-install/x64/mingw/bin;$env:PATH"
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo
& (Join-Path $bin 'cmake.exe') -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=$Toolchain" "-DOpenCV_DIR=$repo/.deps/opencv-install"
if ($LASTEXITCODE) { throw 'CMake configuration failed' }
& (Join-Path $bin 'cmake.exe') --build build -j 4
if ($LASTEXITCODE) { throw 'Build failed' }
& (Join-Path $bin 'ctest.exe') --test-dir build --output-on-failure
if ($LASTEXITCODE) { throw 'Tests failed' }
