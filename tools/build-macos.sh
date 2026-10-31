#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build-macos -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DCMAKE_PREFIX_PATH="$PWD/.deps/macos"
cmake --build build-macos --parallel 3
ctest --test-dir build-macos --output-on-failure
