"""Deploy firmware/current to the board, with the pin-map audit as a gate.

    python tools/deploy.py --label "yaw travel limit"      # upload changed files
    python tools/deploy.py --dry-run

Every upload:
  1. refuses unless verify_against_kicad.py passes (pinmap vs PCB copper),
  2. snapshots what is being sent into firmware/backups/iterNNN_<ts>_<label>/,
  3. increments firmware/ITERATION,
  4. uploads only the files whose contents actually changed,
  5. resets the board and confirms the console comes back.

Rollback is `python tools/deploy.py --from-backup <dir>`.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import send as sendmod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT = os.path.join(ROOT, "firmware", "current")
BACKUPS = os.path.join(ROOT, "firmware", "backups")
ITERATION_FILE = os.path.join(ROOT, "firmware", "ITERATION")
PYTHON = sys.executable


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def latest_backup():
    dirs = [d for d in os.listdir(BACKUPS) if os.path.isdir(os.path.join(BACKUPS, d))]
    return os.path.join(BACKUPS, sorted(dirs)[-1]) if dirs else None


def read_iteration():
    try:
        return int(open(ITERATION_FILE).read().strip())
    except (OSError, ValueError):
        return 0


def run_audit():
    r = subprocess.run([PYTHON, os.path.join(ROOT, "tools", "verify_against_kicad.py")],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr)
    return r.returncode == 0


def changed_files(src):
    """Files in src differing from the newest backup (i.e. from what is on the board)."""
    prev = latest_backup()
    names = sorted(f for f in os.listdir(src) if f.endswith(".py"))
    if prev is None:
        return names
    out = []
    for n in names:
        old = os.path.join(prev, n)
        if not os.path.exists(old) or sha(old) != sha(os.path.join(src, n)):
            out.append(n)
    return out


def mpremote(port, *args):
    cmd = [PYTHON, "-m", "mpremote", "connect", port] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="update", help="short name for this iteration")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-backup", help="roll back: deploy this backup dir instead")
    ap.add_argument("--all", action="store_true", help="upload every file, not just changed")
    args = ap.parse_args()

    src = args.from_backup or CURRENT
    if not os.path.isdir(src):
        raise SystemExit("no such source dir: %s" % src)

    print("=== pin-map audit (gate) ===")
    if not run_audit():
        raise SystemExit("AUDIT FAILED - refusing to deploy")
    print()

    todo = sorted(f for f in os.listdir(src) if f.endswith(".py")) if args.all \
        else changed_files(src)
    if not todo:
        print("nothing changed - board already matches firmware/current")
        return 0

    iteration = read_iteration() + 1
    print("iteration %d  <- %s" % (iteration, args.label))
    for f in todo:
        print("   will upload  %s" % f)
    if args.dry_run:
        print("\n(dry run - nothing sent)")
        return 0

    port = sendmod.find_port()
    if not port:
        raise SystemExit("no MicroPython board (2e8a:0005) found")

    # Snapshot exactly what we are about to put on the board.
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = os.path.join(BACKUPS, "iter%03d_%s_%s" % (iteration, ts, args.label.replace(" ", "-")))
    os.makedirs(dest, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))

    # The console owns stdin; drop to the REPL so mpremote can use the port.
    print("\nreleasing console...")
    sendmod.send(["quit"], timeout=4, quiet=True)
    time.sleep(0.6)

    failed = []
    for f in todo:
        r = mpremote(port, "fs", "cp", os.path.join(src, f), ":" + f)
        ok = r.returncode == 0
        print("   %s %s" % ("ok  " if ok else "FAIL", f))
        if not ok:
            failed.append(f)
            sys.stdout.write(r.stderr)

    print("\nresetting into the console...")
    port = sendmod.reset_into_console(port, quiet=True)

    json.dump({
        "iteration": iteration,
        "label": args.label,
        "uploaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "uploaded": not failed,
        "files_sent": todo,
        "failed": failed,
        "source": os.path.relpath(src, ROOT),
        "files": {f: {"bytes": os.path.getsize(os.path.join(dest, f)), "sha256": sha(os.path.join(dest, f))[:16]}
                  for f in sorted(os.listdir(dest)) if f.endswith(".py")},
    }, open(os.path.join(dest, "MANIFEST.json"), "w"), indent=2)

    if failed:
        raise SystemExit("upload incomplete: %s" % ", ".join(failed))

    open(ITERATION_FILE, "w").write("%d\n" % iteration)
    print("\niteration %d deployed. snapshot: %s" % (iteration, os.path.relpath(dest, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
