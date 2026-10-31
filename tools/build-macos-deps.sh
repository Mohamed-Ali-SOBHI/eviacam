#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
prefix="$root/.deps/macos"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
mkdir -p .deps
if [[ ! -d .deps/wxWidgets-3.2.11 ]]; then
  curl --fail --location --retry 3 https://github.com/wxWidgets/wxWidgets/releases/download/v3.2.11/wxWidgets-3.2.11.tar.bz2 -o .deps/wx.tar.bz2
  tar -xjf .deps/wx.tar.bz2 -C .deps
fi
cmake -S .deps/wxWidgets-3.2.11 -B .deps/wx-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
  -DwxBUILD_SHARED=OFF -DwxBUILD_INSTALL=ON -DwxBUILD_SAMPLES=OFF \
  -DwxBUILD_TESTS=OFF -DwxBUILD_DEBUG_LEVEL=0 \
  -DwxUSE_WEBVIEW=OFF -DwxUSE_MEDIACTRL=OFF -DwxUSE_WEBREQUEST=OFF \
  -DwxUSE_LIBPNG=builtin -DwxUSE_LIBJPEG=builtin -DwxUSE_ZLIB=builtin \
  -DwxUSE_EXPAT=builtin -DwxUSE_LIBTIFF=OFF -DwxUSE_LIBLZMA=OFF
cmake --build .deps/wx-build --parallel 3
cmake --install .deps/wx-build
if [[ ! -d .deps/opencv-4.10.0 ]]; then
  curl --fail --location --retry 3 https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz -o .deps/opencv.tar.gz
  tar -xzf .deps/opencv.tar.gz -C .deps
fi
cmake -S .deps/opencv-4.10.0 -B .deps/opencv-macos-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_LIST=core,imgproc,imgcodecs,video,videoio,objdetect,dnn \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_LAPACK=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=ON -DWITH_OBSENSOR=OFF \
  -DWITH_PROTOBUF=ON -DBUILD_PROTOBUF=ON -DWITH_ADE=OFF -DWITH_TIFF=OFF \
  -DWITH_OPENEXR=OFF -DWITH_WEBP=OFF -DWITH_JASPER=OFF -DWITH_AVIF=OFF \
  -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DCPU_DISPATCH=
cmake --build .deps/opencv-macos-build --parallel 3
cmake --install .deps/opencv-macos-build
