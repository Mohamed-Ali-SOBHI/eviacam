param([string]$Toolchain = (Join-Path $env:USERPROFILE 'Documents\eviacam-msys64\ucrt64'))
$ErrorActionPreference='Stop'
$env:PATH = (Join-Path $Toolchain 'bin')+';'+$env:PATH
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo
$options=@('--fresh','-S','.deps/opencv-4.10.0','-B','.deps/opencv-build','-G','Ninja',
 "-DCMAKE_C_COMPILER=$Toolchain/bin/gcc.exe","-DCMAKE_CXX_COMPILER=$Toolchain/bin/g++.exe",'-DCMAKE_BUILD_TYPE=Release',"-DCMAKE_INSTALL_PREFIX=$repo/.deps/opencv-install",
 '-DBUILD_LIST=core,imgproc,imgcodecs,video,videoio,objdetect,dnn',
 '-DBUILD_TESTS=OFF','-DBUILD_PERF_TESTS=OFF','-DBUILD_EXAMPLES=OFF','-DBUILD_opencv_apps=OFF',
 '-DBUILD_JAVA=OFF','-DBUILD_opencv_python3=OFF','-DWITH_IPP=OFF','-DWITH_OPENCL=OFF',
 '-DWITH_LAPACK=OFF','-DWITH_FFMPEG=OFF','-DWITH_GSTREAMER=OFF','-DWITH_MSMF=OFF',
 '-DWITH_DSHOW=ON','-DWITH_PROTOBUF=ON','-DBUILD_PROTOBUF=ON','-DWITH_TIFF=OFF',
 '-DWITH_OPENEXR=OFF','-DWITH_WEBP=OFF','-DWITH_JASPER=OFF','-DWITH_AVIF=OFF',
 '-DBUILD_ZLIB=ON','-DBUILD_PNG=ON','-DBUILD_JPEG=ON','-DBUILD_SHARED_LIBS=ON',
 '-DCPU_DISPATCH=','-DCMAKE_POLICY_VERSION_MINIMUM=3.5')
& cmake @options > .deps/configure-opencv.log 2>&1
if($LASTEXITCODE) { Get-Content .deps/configure-opencv.log -Tail 40; throw 'OpenCV configure failed' }
& cmake --build .deps/opencv-build -j 4 > .deps/build-opencv.log 2>&1
if($LASTEXITCODE) { Get-Content .deps/build-opencv.log -Tail 40; throw 'OpenCV build failed' }
& cmake --install .deps/opencv-build > .deps/install-opencv.log 2>&1
if($LASTEXITCODE) { throw 'OpenCV install failed' }
Write-Output 'OpenCV 4.10.0 built and installed locally.'
