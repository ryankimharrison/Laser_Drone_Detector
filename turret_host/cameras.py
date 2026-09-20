"""Camera identification, capture threads and exposure lock.

Three facts drive every decision in this file, all of them measured on this
machine (see config.py CAMERAS and BUILD_SPEC "Hard-won constraints"):

  1. MSMF delivers full frame rate but silently ignores exposure writes.
     DSHOW honours exposure but collapses the frame rate. So streaming happens
     on MSMF and exposure is set through a short DSHOW open beforehand -- UVC
     controls live on the device, not on the handle, so they persist.
  2. CAP_PROP_BUFFERSIZE is not settable on Windows. The driver's hidden 2-5
     frame buffer can only be defeated by a tight grab()/retrieve() thread.
  3. Indices are not stable and no fingerprint distinguishes the C270 from the
     built-in HP webcam -- both cap at 1280. The narrow camera is resolved by
     its USB serial or not at all.

Nothing here touches hardware at import time: every open lives inside a
function or a start() call, so this module imports on a bare machine.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
#   sys.path repair -- MUST run before any other import, and may use only
#   `os`/`sys`, which are both loaded before user code runs.
#
#   Running this file by path puts turret_host/ at sys.path[0], where types.py
#   SHADOWS THE STDLIB `types` MODULE. The next lazy stdlib import chain
#   (`threading -> functools -> from types import GenericAlias`, or
#   `re -> enum -> from types import MappingProxyType`) then picks up ours and
#   dies with a circular-import error that names this package and looks like
#   our bug. REPLACE the script directory with the project root: merely
#   inserting the root is not enough, because sys.path[0] still wins.
#
#   A no-op under `python -m turret_host.<mod>` and under a normal import.
# --------------------------------------------------------------------------
import os as _os
import sys as _sys

if __package__ in (None, ""):
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _sys.path[:] = [p for p in _sys.path
                    if _os.path.abspath(p or _os.getcwd()) != _here]
    _sys.path.insert(0, _os.path.dirname(_here))

import json
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# The path repair that makes `python turret_host/cameras.py` work is at the top
# of this file: it has to run before `import re`, not here.
from turret_host import config
from turret_host.types import Frame, Slot


# ==========================================================================
#   MODULE POLICY
# ==========================================================================

# MSMF's source reader cannot be raced: a second open that overlaps the first
# never becomes ready, and the failure mode is a camera that reports
# isOpened() True while every read() fails. Every VideoCapture construction in
# this process goes through this lock, so callers may start capture threads
# from anywhere without coordinating.
_OPEN_LOCK = threading.Lock()

# On MSMF and DSHOW the DirectShow-era encoding applies: 0.25 = manual,
# 0.75 = auto. This is not a fraction of anything, it is an enum in disguise.
_EXPOSURE_MANUAL = 0.25
_EXPOSURE_AUTO = 0.75

# Frames thrown away before timing. MSMF's first frames after an open arrive
# late while the source reader spins up; including them reads as a low fps on
# a camera that is in fact fine.
_FPS_WARMUP_FRAMES = 10

# A dead handle grabs False forever without blocking, so an unguarded loop
# spins at 100% CPU publishing nothing. At 30 fps this is ~1 s of silence.
_MAX_CONSECUTIVE_GRAB_FAILURES = 30

# Resolution asked for when fingerprinting: larger than any UVC mode, so the
# driver clamps to the device's real ceiling.
_FINGERPRINT_REQUEST = (4096, 2160)

# How many OpenCV indices to probe, and how many consecutive dead indices end
# the sweep. Indices are dense on Windows; a gap means the sweep is over.
_MAX_PROBE_INDEX = 8
_PROBE_STOP_AFTER_MISSES = 2

_PS_TIMEOUT_S = 25.0


class CameraError(RuntimeError):
    """A camera could not be opened, configured, or kept running."""


class CameraIdentificationError(CameraError):
    """Which OpenCV index is which physical camera could not be established.

    Raised in preference to returning a plausible guess: guessing here points
    the narrow tracker at the laptop lid camera, and the failure looks like a
    tracking bug rather than a wiring bug.
    """


# ==========================================================================
#   USB IDENTITY
# ==========================================================================

@dataclass(frozen=True)
class UsbVideoDevice:
    """One physical video capture device as Windows sees it.

    Local to this module on purpose: nothing downstream of identification
    needs it, so it does not belong in the shared types contract.
    """
    name: str
    instance_id: str          # the MI_00 capture interface, e.g. USB\VID_046D&PID_0825&MI_00\9&...
    parent_id: str            # the USB device node, whose last segment is the serial
    vid: int
    pid: int
    serial: Optional[str]     # None when the device reports no iSerialNumber

    @property
    def vid_pid(self) -> Tuple[int, int]:
        return (self.vid, self.pid)

    def describe(self) -> str:
        return "%s  VID:PID %04X:%04X  serial %s" % (
            self.name, self.vid, self.pid, self.serial or "<none>")


@dataclass(frozen=True)
class CameraIdentification:
    """Result of identify_cameras(): indices plus the evidence behind them."""
    narrow_index: int
    wide_index: int
    narrow_device: Optional[UsbVideoDevice]
    wide_device: Optional[UsbVideoDevice]
    method: str                                  # "winrt" | "pygrabber" | "elimination"
    max_widths: Dict[int, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["camera identification via %s" % self.method]
        for label, idx, dev in (("narrow", self.narrow_index, self.narrow_device),
                                ("wide", self.wide_index, self.wide_device)):
            lines.append("  %-6s index %d  max width %s  %s" % (
                label, idx, self.max_widths.get(idx, "?"),
                dev.describe() if dev else "<no USB record>"))
        lines.extend("  note: " + n for n in self.notes)
        return "\n".join(lines)


_PS_PARENT_HELPER = r"""
function Get-Parent([string]$id) {
    $p = Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_Parent' -ErrorAction SilentlyContinue
    if ($p) { return [string]$p.Data }
    return ''
}
"""

# Class Camera (and legacy Image) only. Class Media would also return the
# webcam's microphone, which shares the parent and the VID:PID and would show
# up as a second, phantom copy of the same camera.
_PS_PNP_CAMERAS = _PS_PARENT_HELPER + r"""
$ErrorActionPreference = 'Continue'
Get-PnpDevice -Class Camera,Image -PresentOnly | ForEach-Object {
    '{0}|{1}|{2}' -f $_.FriendlyName, $_.InstanceId, (Get-Parent $_.InstanceId)
}
"""

# WinRT's VideoCapture enumeration is the same KSCATEGORY_VIDEO_CAMERA
# interface list that Media Foundation walks, so its order is the order
# OpenCV's MSMF backend indexes. That equivalence is an assumption, not a
# guarantee -- identify_cameras() checks it against the resolution
# fingerprints and refuses if the two disagree.
_PS_WINRT_CAMERAS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Devices.Enumeration.DeviceInformation,Windows.Devices.Enumeration,ContentType=WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$op = [Windows.Devices.Enumeration.DeviceInformation]::FindAllAsync(
    [Windows.Devices.Enumeration.DeviceClass]::VideoCapture)
$task = $asTask.MakeGenericMethod(
    [Windows.Devices.Enumeration.DeviceInformationCollection]).Invoke($null, @($op))
if (-not $task.Wait(15000)) { throw 'DeviceInformation.FindAllAsync timed out' }
foreach ($d in $task.Result) { '{0}|{1}' -f $d.Name, $d.Id }
"""


def _powershell(script: str) -> str:
    """Run a PowerShell script and return stdout, or raise with its stderr."""
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=_PS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise CameraIdentificationError(
            "PowerShell enumeration failed (exit %d): %s"
            % (proc.returncode, (proc.stderr or "").strip()))
    return proc.stdout


def _serial_from_parent(parent_id: str) -> Optional[str]:
    """Pull the USB iSerialNumber out of a parent instance id.

    `USB\\VID_046D&PID_0825\\C8258920` -> "C8258920". When a device reports no
    serial Windows synthesises a location-based id instead, which always
    contains '&' -- a real iSerialNumber never does. That test is why the
    HP webcam's "0001" is kept (it is a genuine, if useless, serial) while
    "6&2000b69f&1&0000" is correctly rejected as not-a-serial.
    """
    if not parent_id:
        return None
    tail = parent_id.rsplit("\\", 1)[-1].strip()
    if not tail or "&" in tail:
        return None
    return tail


def _vid_pid_from_instance(instance_id: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", instance_id)
    if m is None:
        return None
    return (int(m.group(1), 16), int(m.group(2), 16))


def _pnp_video_devices() -> List[UsbVideoDevice]:
    """Every present camera-class device, with VID:PID and serial resolved.

    Unordered -- this answers "what is plugged in", not "which index is it".
    """
    devices: List[UsbVideoDevice] = []
    seen_parents = set()
    for line in _powershell(_PS_PNP_CAMERAS).splitlines():
        line = line.strip()
        if not line or line.count("|") < 2:
            continue
        name, instance_id, parent_id = [p.strip() for p in line.split("|", 2)]
        vid_pid = _vid_pid_from_instance(instance_id)
        if vid_pid is None:
            continue                      # not a USB device (virtual camera, WIA scanner)
        # One physical camera can expose several interfaces under one parent.
        # Keep the first; they are the same piece of hardware.
        key = parent_id.lower() or instance_id.lower()
        if key in seen_parents:
            continue
        seen_parents.add(key)
        devices.append(UsbVideoDevice(
            name=name, instance_id=instance_id, parent_id=parent_id,
            vid=vid_pid[0], pid=vid_pid[1], serial=_serial_from_parent(parent_id)))
    return devices


def _instance_from_winrt_id(winrt_id: str) -> str:
    r"""`\\?\USB#VID_046D&PID_0825&MI_00#9&1de...#{guid}\GLOBAL` -> instance id."""
    body = winrt_id.lstrip("\\?").lstrip("\\")
    parts = body.split("#")
    if len(parts) < 3:
        return ""
    return "\\".join(parts[:3])


def _ordered_devices(pnp: Sequence[UsbVideoDevice]) -> Tuple[List[UsbVideoDevice], str, List[str]]:
    """Return the camera list in OpenCV index order, or ([], "", notes).

    Two independent ways to get the order; both are best-effort, and the
    caller cross-checks whichever one answers against the resolution
    fingerprints before trusting it.
    """
    notes: List[str] = []
    by_instance = {d.instance_id.lower(): d for d in pnp}
    by_name = {}
    for d in pnp:
        by_name.setdefault(d.name.strip().lower(), d)

    # --- 1. WinRT: gives order *and* the device interface id, so the mapping
    #        is by identity rather than by matching display names.
    try:
        out = _powershell(_PS_WINRT_CAMERAS)
    except (CameraIdentificationError, OSError, subprocess.SubprocessError) as exc:
        notes.append("WinRT enumeration unavailable: %s" % exc)
    else:
        ordered: List[UsbVideoDevice] = []
        unmatched = []
        for line in out.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, winrt_id = [p.strip() for p in line.split("|", 1)]
            dev = by_instance.get(_instance_from_winrt_id(winrt_id).lower())
            if dev is None:
                dev = by_name.get(name.lower())
            if dev is None:
                unmatched.append(name)
                continue
            ordered.append(dev)
        if unmatched:
            notes.append("WinRT listed cameras with no PnP record: %s" % ", ".join(unmatched))
        elif ordered:
            return ordered, "winrt", notes

    # --- 2. pygrabber: DirectShow's order, matched by friendly name only.
    #        Weaker on both counts (DSHOW's enumeration order is not by
    #        definition MSMF's), so it is the fallback, never the first try.
    try:
        from pygrabber.dshow_graph import FilterGraph  # noqa: PLC0415  (optional dep)
    except ImportError:
        notes.append("pygrabber not installed")
        return [], "", notes

    names = FilterGraph().get_input_devices()
    ordered = []
    for name in names:
        dev = by_name.get(name.strip().lower())
        if dev is None:
            notes.append("pygrabber name %r matches no PnP record" % name)
            return [], "", notes
        ordered.append(dev)
    if not ordered:
        notes.append("pygrabber returned no devices")
        return [], "", notes
    notes.append("order from DirectShow (pygrabber); verified against fingerprints")
    return ordered, "pygrabber", notes


# ==========================================================================
#   FINGERPRINTING
# ==========================================================================

def probe_max_widths(max_index: int = _MAX_PROBE_INDEX) -> Dict[int, int]:
    """index -> the widest mode the device at that index will accept.

    Opens one camera at a time and always releases. Slow (about a second per
    index) and only run at startup.
    """
    widths: Dict[int, int] = {}
    misses = 0
    for i in range(max_index):
        with _OPEN_LOCK:
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            try:
                if not cap.isOpened():
                    misses += 1
                    if misses >= _PROBE_STOP_AFTER_MISSES:
                        break
                    continue
                misses = 0
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, _FINGERPRINT_REQUEST[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _FINGERPRINT_REQUEST[1])
                widths[i] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            finally:
                # Unconditional: a leaked handle costs the *next* run, which
                # then sees isOpened() True and every read() failing.
                cap.release()
    return widths


_INDEX_CACHE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "calibration", "camera_indices.json")


def _usb_signature(pnp) -> str:
    """A cheap fingerprint of exactly which camera devices are attached.

    Built from the PnP inventory, which costs one PowerShell call -- not from
    opening cameras, which is the expensive part we are trying to avoid.
    Replug anything, add or remove a device, and this changes, so a stale cache
    cannot survive the one event that invalidates an index.
    """
    return "|".join(sorted("%04X:%04X/%s" % (d.vid_pid[0], d.vid_pid[1],
                                             d.serial or "-") for d in pnp))


def _load_index_cache(signature: str):
    try:
        with open(_INDEX_CACHE, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None
    if blob.get("signature") != signature:
        return None
    try:
        return int(blob["narrow_index"]), int(blob["wide_index"]), blob.get("max_widths", {})
    except (KeyError, TypeError, ValueError):
        return None


def _save_index_cache(signature: str, narrow: int, wide: int, widths) -> None:
    try:
        _os.makedirs(_os.path.dirname(_INDEX_CACHE), exist_ok=True)
        tmp = _INDEX_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"signature": signature, "narrow_index": narrow,
                       "wide_index": wide,
                       "max_widths": {str(k): v for k, v in (widths or {}).items()},
                       "note": "Indices cached to skip probe_max_widths, which "
                               "opens every index on MSMF and costs ~60 s. "
                               "Invalidated automatically when the attached "
                               "camera set changes. Delete this file to force "
                               "a full re-probe."}, fh, indent=2)
        _os.replace(tmp, _INDEX_CACHE)
    except OSError:
        pass                      # a cache that cannot be written is not fatal


def identify_cameras(max_index: int = _MAX_PROBE_INDEX,
                     use_cache: bool = True) -> CameraIdentification:
    """Resolve the narrow and wide cameras to OpenCV indices.

    Narrow is resolved by USB serial (config.NARROW_SERIAL) and never by
    resolution: the built-in HP webcam caps at 1280 exactly like the C270, so
    the max-width fingerprint cannot separate them. Wide has no serial, so it
    is resolved by VID:PID and confirmed by the 1920 fingerprint.

    Raises CameraIdentificationError rather than returning a guess.
    """
    pnp = _pnp_video_devices()
    if not pnp:
        raise CameraIdentificationError(
            "no camera-class USB devices present -- nothing is plugged in, or "
            "PowerShell's Get-PnpDevice returned nothing")

    narrow_devs = [d for d in pnp
                   if d.vid_pid == config.NARROW_VID_PID and d.serial == config.NARROW_SERIAL]
    wide_devs = [d for d in pnp if d.vid_pid == config.WIDE_VID_PID]

    inventory = "\n".join("    " + d.describe() for d in pnp)
    if len(narrow_devs) != 1:
        raise CameraIdentificationError(
            "expected exactly one narrow camera (VID:PID %04X:%04X serial %s), found %d.\n"
            "  present:\n%s" % (config.NARROW_VID_PID[0], config.NARROW_VID_PID[1],
                                config.NARROW_SERIAL, len(narrow_devs), inventory))
    if len(wide_devs) != 1:
        raise CameraIdentificationError(
            "expected exactly one wide camera (VID:PID %04X:%04X), found %d.\n"
            "  present:\n%s" % (config.WIDE_VID_PID[0], config.WIDE_VID_PID[1],
                                len(wide_devs), inventory))
    narrow_dev, wide_dev = narrow_devs[0], wide_devs[0]

    # A cache hit skips probe_max_widths entirely. MEASURED on this machine:
    # identify_cameras() is 64.0 s of a 67.5 s cold start, essentially all of it
    # probe_max_widths opening every index 0..8 on MSMF one at a time (MSMF's
    # source reader cannot be raced, so they cannot overlap). torch, CUDA and
    # YOLO together are 3.4 s of that 67.5.
    #
    # The cache is keyed on the attached-device signature, so replugging a
    # camera -- the one thing that actually moves an index -- misses and
    # re-probes. That preserves config.py's rule that indices are never
    # hardcoded: they are still MEASURED, just not re-measured every launch.
    signature = _usb_signature(pnp)
    if use_cache:
        hit = _load_index_cache(signature)
        if hit is not None:
            n_idx, w_idx, cached_widths = hit
            return CameraIdentification(
                narrow_index=n_idx, wide_index=w_idx,
                narrow_device=narrow_devs[0], wide_device=wide_devs[0],
                method="cache",
                max_widths={int(k): v for k, v in cached_widths.items()},
                notes=["indices from %s -- delete it to force a re-probe"
                       % _os.path.basename(_INDEX_CACHE)])

    ordered, method, notes = _ordered_devices(pnp)
    widths = probe_max_widths(max_index)
    if not widths:
        raise CameraIdentificationError(
            "PnP lists %d camera(s) but OpenCV could not open any index. Another "
            "process is holding them, or a previous run leaked a handle." % len(pnp))

    if ordered:
        if len(ordered) != len(widths):
            raise CameraIdentificationError(
                "enumeration lists %d cameras but OpenCV opens %d indices, so the "
                "index mapping cannot be trusted. Close whatever else is using a "
                "camera and retry.\n  enumerated: %s\n  openable indices: %s"
                % (len(ordered), len(widths),
                   ", ".join(d.name for d in ordered), sorted(widths)))
        index_of = {d.instance_id.lower(): i for i, d in enumerate(ordered)}
        narrow_index = index_of.get(narrow_dev.instance_id.lower())
        wide_index = index_of.get(wide_dev.instance_id.lower())
        if narrow_index is None or wide_index is None:
            raise CameraIdentificationError(
                "identified cameras are missing from the ordered enumeration "
                "(narrow=%s wide=%s)" % (narrow_index, wide_index))

        # The fingerprints are the checksum on the ordering assumption. They
        # cannot tell the C270 from the HP webcam, but they do catch the case
        # where the enumeration order is not OpenCV's order at all.
        if widths.get(wide_index) != config.WIDE_FINGERPRINT_WIDTH:
            raise CameraIdentificationError(
                "wide camera maps to index %d but that index caps at %s, not %d -- "
                "the enumeration order does not match OpenCV's. Refusing to guess.\n"
                "  order: %s\n  widths: %s"
                % (wide_index, widths.get(wide_index), config.WIDE_FINGERPRINT_WIDTH,
                   [d.name for d in ordered], widths))
        if widths.get(narrow_index) != config.NARROW_SIZE[0]:
            raise CameraIdentificationError(
                "narrow camera maps to index %d but that index caps at %s, not %d -- "
                "the enumeration order does not match OpenCV's. Refusing to guess.\n"
                "  order: %s\n  widths: %s"
                % (narrow_index, widths.get(narrow_index), config.NARROW_SIZE[0],
                   [d.name for d in ordered], widths))
        _save_index_cache(signature, narrow_index, wide_index, widths)
        return CameraIdentification(narrow_index, wide_index, narrow_dev, wide_dev,
                                    method, widths, notes)

    # --- No ordering source. The mapping is only recoverable when it is
    #     forced by elimination; anything else is a coin flip on which camera
    #     the tracker points at, so it is an error.
    notes.append("no device ordering available -- resolving by elimination")
    wide_indices = [i for i, w in widths.items() if w == config.WIDE_FINGERPRINT_WIDTH]
    narrow_indices = [i for i, w in widths.items() if w == config.NARROW_SIZE[0]]
    if len(wide_indices) != 1:
        raise CameraIdentificationError(
            "cannot order cameras (%s) and %d indices report the %d-wide "
            "fingerprint, so the wide camera is ambiguous."
            % ("; ".join(notes), len(wide_indices), config.WIDE_FINGERPRINT_WIDTH))
    if len(narrow_indices) != 1:
        others = [d.describe() for d in pnp if d is not narrow_dev]
        raise CameraIdentificationError(
            "cannot order cameras (%s) and %d indices cap at %d, so the narrow "
            "camera cannot be told apart from the other %d-wide device(s) by "
            "fingerprint alone. Install pygrabber, or unplug/disable:\n    %s"
            % ("; ".join(notes), len(narrow_indices), config.NARROW_SIZE[0],
               config.NARROW_SIZE[0], "\n    ".join(others) or "<none>"))
    _save_index_cache(signature, narrow_indices[0], wide_indices[0], widths)
    return CameraIdentification(narrow_indices[0], wide_indices[0], narrow_dev, wide_dev,
                                "elimination", widths, notes)


# ==========================================================================
#   OPEN / CONFIGURE / MEASURE
# ==========================================================================

def open_camera(index: int,
                size: Tuple[int, int],
                fps: int,
                backend: int = cv2.CAP_MSMF,
                exposure: Optional[float] = None) -> cv2.VideoCapture:
    """Open one camera on `backend` in MJPEG at `size`, or raise.

    Opens are serialised process-wide: MSMF's source reader cannot be raced.
    """
    width, height = size
    with _OPEN_LOCK:
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                raise CameraError("camera index %d would not open on backend %d" % (index, backend))

            # MJPEG is mandatory: the same resolutions exist as uncompressed
            # YUYV and run 3-6x slower over USB 2.0. MSMF *refuses* this set
            # and returns False -- that is normal, it negotiates the
            # compressed mode itself. Do not "fix" it by switching to DSHOW.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            # EXPOSURE, ON THIS HANDLE, AFTER THE MODE IS NEGOTIATED.
            #
            # The module header says MSMF "silently ignores exposure writes".
            # MEASURED 2026-09-20 on this rig: it does NOT. Writing exposure to
            # the live MSMF handle changes the image and the distinct-frame rate
            # exactly as asked (-5 -> 25.4 fps at brightness 79; -6 -> 27.1 at
            # 36). What is broken is only the READBACK, which returns a
            # constant -- which is why the separate DSHOW-then-reopen lock
            # reported success while changing nothing: it set a value on a
            # handle it then closed, and "verified" via a number that never
            # moves.
            #
            # Setting it here means the exposure belongs to the handle that
            # actually streams, and cannot be lost by a reopen.
            if exposure is not None:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _EXPOSURE_MANUAL)
                cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

            got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if got != (width, height):
                raise CameraError(
                    "camera index %d negotiated %dx%d, asked for %dx%d -- the mode "
                    "is unavailable, and silently running at the wrong resolution "
                    "would put every pixel-space constant out by a scale factor"
                    % (index, got[0], got[1], width, height))
            return cap
        except BaseException:
            # Includes KeyboardInterrupt on purpose: a handle leaked here is
            # only recovered by replugging the camera.
            cap.release()
            raise


def measure_delivered_fps(cap: cv2.VideoCapture,
                          frames: int = config.FPS_PROBE_FRAMES,
                          warmup: int = _FPS_WARMUP_FRAMES) -> float:
    """Frames per second actually delivered, timed over `frames` grabs.

    CAP_PROP_FPS is what the driver claims, and DSHOW will happily claim 30
    while delivering 10. The only honest number is the one on the clock.
    """
    for _ in range(warmup):
        if not cap.grab():
            raise CameraError("camera stopped delivering during fps warm-up")
    t0 = time.perf_counter()
    for _ in range(frames):
        if not cap.grab():
            raise CameraError("camera stopped delivering during fps measurement")
        cap.retrieve()          # retrieve too: decoding MJPEG is part of the cost
    dt = time.perf_counter() - t0
    if dt <= 0.0:
        raise CameraError("fps measurement took no measurable time")
    return frames / dt


def verify_camera(index: int,
                  size: Tuple[int, int],
                  fps: int,
                  min_fps: float = config.MIN_ACCEPTABLE_FPS) -> float:
    """Open, measure real delivered fps, release. Raises below `min_fps`."""
    cap = open_camera(index, size, fps)
    try:
        measured = measure_delivered_fps(cap)
    finally:
        cap.release()
    if measured < min_fps:
        raise CameraError(
            "camera index %d delivers %.1f fps at %dx%d, below the %.0f fps floor. "
            "Usual causes: auto-exposure stretching the integration time in a dim "
            "room (lock it), or both cameras sharing one xHCI root port."
            % (index, measured, size[0], size[1], min_fps))
    return measured


# ==========================================================================
#   EXPOSURE LOCK
# ==========================================================================

@dataclass
class ExposureLockResult:
    """What the exposure lock actually achieved, as opposed to what it asked for."""
    requested: float
    before: float
    after_dshow: float
    after_msmf: float
    auto_exposure: float
    gain_requested: Optional[float]
    gain_after: float
    took: bool                # the value on the device changed to what we asked
    persisted: bool           # and it survived the reopen on the streaming backend
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.took and self.persisted


def lock_exposure(index: int,
                  exposure: float,
                  gain: Optional[float] = None,
                  verify_backend: int = cv2.CAP_MSMF) -> ExposureLockResult:
    """Pin exposure (and optionally gain) on the device, and verify it took.

    `exposure` is in the UVC/DirectShow log2-seconds convention: -6 is 1/64 s,
    -7 is 1/128 s. Lower means shorter integration, which is what restores
    frame rate in a dim room.

    The detour through DSHOW is the whole point. MSMF's set() returns True and
    changes nothing, so writing exposure on the streaming handle looks like it
    worked and silently does not. DSHOW writes reach the device, and UVC
    controls are device state rather than handle state, so they survive the
    release and the reopen on MSMF.

    Returns a result instead of raising when the value does not take: the lock
    is a frame-rate optimisation, not a safety interlock, and the caller
    decides. It never reports success it did not verify.
    """
    with _OPEN_LOCK:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise CameraError("camera index %d would not open on DSHOW for exposure lock" % index)
            before = cap.get(cv2.CAP_PROP_EXPOSURE)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, _EXPOSURE_MANUAL)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if gain is not None:
                cap.set(cv2.CAP_PROP_GAIN, gain)
            # Read back from the device, not from our own request.
            auto_after = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            after_dshow = cap.get(cv2.CAP_PROP_EXPOSURE)
            gain_after = cap.get(cv2.CAP_PROP_GAIN)
        finally:
            cap.release()

    # Re-open on the backend that will actually stream and confirm the device
    # kept the value. This is the claim being tested -- verify it, don't assume.
    with _OPEN_LOCK:
        cap = cv2.VideoCapture(index, verify_backend)
        try:
            if not cap.isOpened():
                raise CameraError(
                    "camera index %d would not reopen on backend %d after exposure lock"
                    % (index, verify_backend))
            after_msmf = cap.get(cv2.CAP_PROP_EXPOSURE)
        finally:
            cap.release()

    took = abs(after_dshow - exposure) < 0.5
    persisted = abs(after_msmf - exposure) < 0.5
    if took and persisted:
        # NOT EVIDENCE, AND SAY SO. MEASURED 2026-09-20: MSMF's exposure
        # read-back on the C270 returns the SAME value whatever is written --
        # it reported -6 while the image responded correctly to -4, -5, -6 and
        # -7. So `persisted` is satisfied by a constant, and this branch fired
        # with "exposure locked at -6 (was -6)" on a camera that was running at
        # roughly -7 and delivering 17 distinct fps. It reported success and
        # changed nothing, every run, all day.
        #
        # The exposure that actually reaches the sensor is now set on the
        # STREAMING handle in open_camera(); this function survives only as a
        # pre-open nudge. The message no longer claims to have verified
        # anything it cannot.
        message = ("exposure set to %g (read-back says %g, but MSMF's read-back "
                   "is a CONSTANT on this device and cannot verify anything -- "
                   "the authoritative set is on the streaming handle, see "
                   "open_camera)" % (exposure, after_msmf))
    elif took:
        message = ("exposure %g took on DSHOW but reads %g after reopening on backend %d "
                   "-- the device did not keep it; expect auto-exposure to fight the "
                   "frame rate" % (exposure, after_msmf, verify_backend))
    else:
        message = ("exposure %g was REJECTED: device reports %g (auto_exposure=%g). "
                   "Camera is still auto-exposing." % (exposure, after_dshow, auto_after))
    return ExposureLockResult(
        requested=exposure, before=before, after_dshow=after_dshow, after_msmf=after_msmf,
        auto_exposure=auto_after, gain_requested=gain, gain_after=gain_after,
        took=took, persisted=persisted, message=message)


# ==========================================================================
#   CAPTURE THREAD
# ==========================================================================

class CameraThread(threading.Thread):
    """Tight grab/retrieve loop publishing Frames into a Slot.

    The loop body is `grab(); t = perf_counter(); retrieve()` and that order is
    mandatory. grab() returns when the frame lands, so the timestamp taken
    immediately after it is the best estimate of exposure time available;
    timing after retrieve() folds in the MJPEG decode. Running the loop as fast
    as the camera allows is the only way to keep the driver's hidden 2-5 frame
    buffer empty -- CAP_PROP_BUFFERSIZE is not settable on Windows.

    Frames are published raw. The narrow camera is mounted rotated 90 degrees
    (config.NARROW_ROTATION_DEG) but the rotation is carried in the geometry,
    never applied here: a full-frame rotate per frame buys nothing and costs a
    copy.

    No hardware is touched until start().
    """

    def __init__(self,
                 name: str,
                 index: int,
                 size: Tuple[int, int],
                 fps: int,
                 slot: Optional[Slot] = None,
                 backend: int = cv2.CAP_MSMF,
                 min_fps: float = config.MIN_ACCEPTABLE_FPS,
                 probe_frames: int = config.FPS_PROBE_FRAMES,
                 exposure: Optional[float] = None):
        super().__init__(name="capture-%s" % name, daemon=True)
        self.camera_name = name          # goes into Frame.camera: "narrow" | "wide"
        self.index = index
        self.size = size
        self.requested_fps = fps
        self.backend = backend
        #: Applied on the streaming handle in start(). None leaves whatever the
        #: PREVIOUS process left behind -- the camera retains it across
        #: restarts, so "not set" is not "default". See config.NARROW_EXPOSURE_EV.
        self.exposure = exposure
        self.min_fps = min_fps
        self.probe_frames = probe_frames

        self.slot: Slot = slot if slot is not None else Slot()
        self.cap: Optional[cv2.VideoCapture] = None

        self.startup_fps = 0.0           # measured once, before the loop runs
        self.frames = 0
        self.failed_grabs = 0
        #: Re-deliveries of an image already published. See config.DEDUP_FRAMES.
        self.duplicates_dropped = 0
        self._last_image = None
        self.max_gap_ms = 0.0
        self.error: Optional[str] = None

        # Rolling window of delivery times. Read without a lock: CPython makes
        # the float store atomic and a torn read of a telemetry number is not
        # worth a lock on the hot path.
        self.fps = 0.0
        self._times: deque = deque(maxlen=31)
        # NOT self._stop: threading.Thread._stop() is an internal method that
        # join() calls, and shadowing it with an Event breaks join().
        self._stop_evt = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "CameraThread":
        """Open the camera, prove it delivers, then run. Raises if it does not."""
        cap = open_camera(self.index, self.size, self.requested_fps,
                          self.backend, exposure=self.exposure)
        try:
            self.startup_fps = measure_delivered_fps(cap, self.probe_frames)
            if self.startup_fps < self.min_fps:
                raise CameraError(
                    "%s camera (index %d) delivers %.1f fps at %dx%d, below the "
                    "%.0f fps floor -- refusing to start. Lock the exposure, and "
                    "check the two cameras are not sharing one xHCI root port."
                    % (self.camera_name, self.index, self.startup_fps,
                       self.size[0], self.size[1], self.min_fps))
        except BaseException:
            cap.release()
            raise
        self.cap = cap
        super().start()
        return self

    def run(self) -> None:
        cap = self.cap
        consecutive_failures = 0
        last_t = None
        try:
            while not self._stop_evt.is_set():
                if not cap.grab():
                    self.failed_grabs += 1
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_GRAB_FAILURES:
                        self.error = (
                            "%s camera (index %d) failed %d consecutive grabs -- the "
                            "handle is dead. isOpened() still reports %s."
                            % (self.camera_name, self.index, consecutive_failures,
                               cap.isOpened()))
                        return
                    continue
                t = time.perf_counter()
                ok, image = cap.retrieve()
                if not ok:
                    self.failed_grabs += 1
                    consecutive_failures += 1
                    continue
                consecutive_failures = 0

                # DUPLICATE REJECTION -- see config.DEDUP_FRAMES.
                #
                # This is the ONLY place it can be done. Downstream every
                # consumer guards on Slot's sequence number, and put() bumps
                # that for a re-delivery exactly as for a real frame, so a
                # duplicate is invisible from there. It is invisible from the
                # timestamp too: t is stamped after grab(), so the copy gets a
                # fresh one.
                #
                # Exact equality, on purpose: a parked turret looking at a
                # still scene produces frames that are ALMOST identical, and
                # those are real observations. Only a bit-for-bit repeat is a
                # re-delivery.
                if config.DEDUP_FRAMES and self._last_image is not None:
                    st = config.DEDUP_STRIDE
                    # Decimated view first -- microseconds, and it rejects a
                    # genuinely new frame immediately. The full comparison runs
                    # only on what survives, which is the duplicate itself.
                    if (image.shape == self._last_image.shape
                            and np.array_equal(image[::st, ::st],
                                               self._last_image[::st, ::st])
                            and np.array_equal(image, self._last_image)):
                        self.duplicates_dropped += 1
                        continue

                self.frames += 1
                self.slot.put(Frame(image=image, t=t, index=self.frames,
                                    camera=self.camera_name))
                # A COPY, never a reference: retrieve() is free to hand back
                # the same buffer next call, and holding a reference to it
                # would make every frame compare equal to itself and drop the
                # entire stream.
                if config.DEDUP_FRAMES:
                    self._last_image = image.copy()

                if last_t is not None:
                    gap_ms = (t - last_t) * 1000.0
                    # Recorded, never acted on: worst case here is ~50 ms
                    # against a 33 ms median and one late frame is not a lost
                    # track. The tracker's miss counters own that decision.
                    if gap_ms > self.max_gap_ms:
                        self.max_gap_ms = gap_ms
                last_t = t
                self._times.append(t)
                if len(self._times) > 1:
                    span = self._times[-1] - self._times[0]
                    if span > 0.0:
                        self.fps = (len(self._times) - 1) / span
        finally:
            # Unconditional, including on exception and on Ctrl-C: a leaked
            # handle makes the next run open a camera that reports isOpened()
            # True while every read() fails.
            cap.release()
            self.cap = None
            self.fps = 0.0

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the loop and release the camera. Safe to call more than once."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout)
        cap, self.cap = self.cap, None
        if cap is not None:
            # The thread normally releases; this covers stop() before start()
            # finished, and a join() that timed out.
            cap.release()

    # -- convenience -------------------------------------------------------

    def latest(self) -> Tuple[Optional[Frame], int]:
        item, seq = self.slot.get()
        return item, seq

    def __enter__(self) -> "CameraThread":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def narrow_thread(index: int, slot: Optional[Slot] = None) -> CameraThread:
    """Capture thread for the C270, at its measured ceiling."""
    return CameraThread("narrow", index, config.NARROW_SIZE, config.NARROW_FPS,
                        slot, exposure=config.NARROW_EXPOSURE_EV)


def wide_thread(index: int, slot: Optional[Slot] = None, fast: bool = False) -> CameraThread:
    """Capture thread for the wide module.

    `fast` selects the 1280x720 @ 60 fps MJPEG mode, which is a CENTER CROP
    (horizontal field ~108 -> ~85 deg), not a downscale. That narrows the
    search field, so it is off by default.
    """
    size = config.WIDE_FAST_SIZE if fast else config.WIDE_SIZE
    fps = config.WIDE_FAST_FPS if fast else config.WIDE_FPS
    return CameraThread("wide", index, size, fps, slot,
                        exposure=config.WIDE_EXPOSURE_EV)


# ==========================================================================
#   SELF TEST
# ==========================================================================

def _main() -> int:
    print("OpenCV", cv2.__version__)
    ident = identify_cameras()
    print(ident.report())

    # Sequentially, never concurrently: MSMF cannot be raced at open.
    for label, index, size, fps in (
            ("narrow", ident.narrow_index, config.NARROW_SIZE, config.NARROW_FPS),
            ("wide", ident.wide_index, config.WIDE_SIZE, config.WIDE_FPS)):
        measured = verify_camera(index, size, fps)
        print("  %-6s index %d  %dx%d  %.1f fps delivered (floor %.0f)"
              % (label, index, size[0], size[1], measured, config.MIN_ACCEPTABLE_FPS))

    print("\nrunning both capture threads for 3 s")
    threads = []
    try:
        threads.append(narrow_thread(ident.narrow_index).start())
        threads.append(wide_thread(ident.wide_index).start())
        time.sleep(3.0)
        for th in threads:
            frame, seq = th.latest()
            shape = frame.image.shape if frame is not None else None
            print("  %-6s %.1f fps  %d frames  worst gap %.1f ms  failed grabs %d  last %s seq %d"
                  % (th.camera_name, th.fps, th.frames, th.max_gap_ms,
                     th.failed_grabs, shape, seq))
            if th.error:
                print("  %-6s ERROR: %s" % (th.camera_name, th.error))
    finally:
        for th in threads:
            th.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
