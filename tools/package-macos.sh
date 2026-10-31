#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
app="build-macos/eViacam-YuNet.app"
resources="$app/Contents/Resources"
mkdir -p "$resources/licenses" dist
cp COPYING "$resources/licenses/eViacam-GPL.txt"
cp models/LICENSE-YuNet "$resources/licenses/"
cp .deps/opencv-4.10.0/LICENSE "$resources/licenses/OpenCV-Apache.txt"
cp .deps/wxWidgets-3.2.11/docs/licence.txt "$resources/licenses/wxWidgets.txt"
cp -R .deps/macos/share/licenses "$resources/licenses/dependencies" 2>/dev/null || true
cp README-MACOS.md "$resources/"
# Ad-hoc signature satisfies local code integrity on Apple Silicon; it is not
# an Apple Developer ID signature or notarization. No security settings change.
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
python tools/test-macos-bundle.py
archive="dist/eViacam-YuNet-macos-$(uname -m).zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
shasum -a 256 "$archive" > "$archive.sha256"
