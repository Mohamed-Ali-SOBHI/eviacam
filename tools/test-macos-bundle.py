"""Run the packaged Cocoa app on a replay: no camera, no input permissions/events."""
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
app = root / "build-macos/eViacam-YuNet.app"
exe = app / "Contents/MacOS/eViacam-YuNet"
# Check all load commands: a portable bundle must not need Homebrew/build paths.
deps = subprocess.check_output(["otool", "-L", str(exe)], text=True)
for line in deps.splitlines()[1:]:
    name = line.strip().split(" (", 1)[0]
    assert name.startswith(("/System/Library/", "/usr/lib/")), f"Non-system dependency: {name}"
with tempfile.TemporaryDirectory() as folder:
    config = Path(folder) / "smoke.ini"
    config.write_text("[settings]\ncurrentProfile=default\ncheckUpdatesAtStartup=0\n"
                      "[settings/default]\nenabledAtStartup=0\nrunWizardAtStartup=0\n"
                      "minimisedAtStartup=0\n", encoding="utf-8")
    env = dict(os.environ, EVIACAM_REPLAY=str(root / "tests/generated/replay.avi"),
               EVIACAM_SMOKE_TEST="1", EVIACAM_TIMED_TRIAL="1", EVIACAM_FACE_DETECTOR="yunet")
    run = subprocess.run([str(exe), "-c", str(config)], env=env, cwd=folder,
                         capture_output=True, text=True, timeout=60)
    print(run.stdout)
    print(run.stderr)
    assert run.returncode == 0, f"App exited with {run.returncode}"
    assert "MACOS_REPLAY_OK frames=90" in run.stdout, "App did not render/detect 90 frames"
print("PASS: standalone Cocoa bundle, resources, YuNet, video replay and clean exit")
