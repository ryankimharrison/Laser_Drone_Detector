"""Does locking exposure restore the C270's frame rate?

The camera delivers ~19 fps in a dim room at every resolution, which points at
auto-exposure extending integration time rather than at USB bandwidth. AGENT_HANDOFF
(5.2) requires >= 25 fps and notes MSMF exposure control is unverified on this bench.

Read-only with respect to the turret: this only touches cameras.
"""
import time

import cv2

WIDTH, HEIGHT = 1280, 720
WARMUP = 10
MEASURE = 60

# OpenCV's manual/auto convention differs per backend. On MSMF and DSHOW the
# DirectShow-era encoding is 0.25 = manual, 0.75 = auto.
MANUAL, AUTO = 0.25, 0.75


def open_cam(index, backend):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def measure(cap, n=MEASURE):
    for _ in range(WARMUP):
        if not cap.read()[0]:
            return None
    t0 = time.perf_counter()
    ok_count = 0
    for _ in range(n):
        if cap.read()[0]:
            ok_count += 1
    dt = time.perf_counter() - t0
    return ok_count / dt if dt > 0 else None


def report(label, cap):
    fps = measure(cap)
    ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    ex = cap.get(cv2.CAP_PROP_EXPOSURE)
    flag = "OK " if fps and fps >= 25 else "LOW"
    print(
        "  %-34s %s %5.1f fps   auto_exposure=%-6s exposure=%s"
        % (label, flag, fps or 0.0, ae, ex)
    )
    return fps


def trial(index, backend, backend_name):
    print("\n=== %s, %dx%d MJPG ===" % (backend_name, WIDTH, HEIGHT))
    cap = open_cam(index, backend)
    if cap is None:
        print("  could not open")
        return
    try:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("  negotiated %dx%d" % (w, h))

        report("baseline (as opened)", cap)

        ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL)
        print("  set(AUTO_EXPOSURE, %.2f) -> %s" % (MANUAL, ok))
        report("after manual-exposure request", cap)

        for ev in (-4, -5, -6, -7):
            ok = cap.set(cv2.CAP_PROP_EXPOSURE, ev)
            if not ok:
                print("  set(EXPOSURE, %d) -> False" % ev)
                continue
            report("manual, exposure=%d" % ev, cap)
    finally:
        cap.release()


def find_index_by_size(target_w):
    """Identify a camera by the resolution it accepts -- indices are not stable."""
    found = {}
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        found[i] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    print("index -> max width:", found)
    for i, w in found.items():
        if w == target_w:
            return i
    return None


if __name__ == "__main__":
    print("OpenCV", cv2.__version__)
    # The C270 tops out at 1280; the wide module reaches 1920.
    idx = find_index_by_size(1280)
    if idx is None:
        raise SystemExit("C270 not found by max-width fingerprint")
    print("C270 at index %d" % idx)
    trial(idx, cv2.CAP_MSMF, "CAP_MSMF")
    trial(idx, cv2.CAP_DSHOW, "CAP_DSHOW")
