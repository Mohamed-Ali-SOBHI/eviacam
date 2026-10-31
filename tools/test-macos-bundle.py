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
    with subprocess.Popen([str(exe), "-d", "-c", str(config)], env=env, cwd=folder,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            report = root / "build-macos/Testing/Temporary/bundle-hang.log"
            subprocess.run(["sample", str(process.pid), "1", "-file", str(report)], timeout=15)
            process.kill()
            stdout, stderr = process.communicate(timeout=10)
            print(stdout, flush=True)
            print(stderr, flush=True)
            print(report.read_text(errors="replace") if report.exists() else "No stack report", flush=True)
            raise AssertionError("Cocoa app did not finish the replay within 60 seconds")
        print(stdout)
        print(stderr)
        assert process.returncode == 0, f"App exited with {process.returncode}"
        assert "MACOS_REPLAY_OK frames=90" in stdout, "App did not render/detect 90 frames"
print("PASS: standalone Cocoa bundle, resources, YuNet, video replay and clean exit")
