"""Operator panel and spectator display.

Runs in the main thread and owns the Tk event loop. Everything it shows arrives
from somewhere else: `set_status()` takes the published `SystemStatus`,
`set_frames()` takes display frames plus the geometry to draw on them. This
module imports no camera, detector, tracker, control or link code -- the app
wires callbacks into it. That keeps the GUI runnable with nothing attached,
which is also how the `__main__` block below checks the layout.

Two rules that look like details and are not:

* **Never draw on an inference buffer.** Every frame handed in here is still
  owned by the capture/inference side. `cv2.resize()` to `config.DISPLAY_SIZE`
  allocates a new array, and all drawing happens on that copy. Nothing in this
  file writes through a reference it was given.

* **The narrow view is rotated FOR DISPLAY ONLY.** The C270 is mounted 90 deg
  over, and the geometry in control/calibration already carries that rotation.
  Rotating frames for processing would double-count it. So the rotation is
  applied to the 640x360 display copy -- cheap -- and the overlay points are
  transformed to match, which keeps the labels upright.

Tk is not thread safe. Every public setter here is callable from any thread and
does nothing but hand data to a lock (or a `Slot`); widgets are only ever
touched from `_tick()`, on the main thread.
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
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# The path repair that makes `python turret_host/gui.py` work is at the top of
# this file: it has to run before `import threading`, not here.
from turret_host import config
from turret_host.types import (
    ControlOutput,
    Detection,
    DetectionResult,
    Frame,
    LaserState,
    LinkStatus,
    Slot,
    SystemStatus,
    TrackEstimate,
    TrackState,
)

# Optional: the 3D wrist view needs turret_host/assets/wrist_mesh.npz, which is
# built from the CAD by tools/export_wrist_mesh.py. A checkout without it (or
# without the mesh exporter's dependencies) must still bring the panel up --
# this is a diagnostic, not flight equipment.
try:
    from turret_host import wristview as _wristview
except Exception:                                            # pragma: no cover
    _wristview = None

# ==========================================================================
#   PALETTE
# ==========================================================================
BG = "#0e1116"
PANEL = "#171c24"
PANEL_HI = "#212836"
EDGE = "#2c3545"
FG = "#dfe6ef"
MUTED = "#8b98a9"

GREEN = "#2ecc71"
GREEN_HI = "#7cffb0"
AMBER = "#f0a830"
RED = "#ff3b30"
RED_HI = "#ff8a80"
CYAN = "#35d6ff"
MAGENTA = "#ff5cf0"
YELLOW = "#ffe066"
VIOLET = "#c6a8ff"
WHITE = "#ffffff"

FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"

# Faces get their own colour cycle so two overlapping faces are still two
# faces on screen. Deliberately all warm/red: a face is the inhibit condition
# and must never read as "just another box".
FACE_COLOURS = ("#ff3b30", "#ff8f1f", "#ff1f8f", "#ffd21f")

_UNSET = object()

# cv2 rotation codes, keyed by CLOCKWISE degrees. Display only -- nothing on
# the processing path reads these.
_ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# ==========================================================================
#   small helpers
# ==========================================================================
def _overlay_font(size: int) -> ImageFont.ImageFont:
    """A readable overlay face.

    PIL's built-in bitmap font is ~11 px and unreadable on a 640x360 pane at
    arm's length. Stock Windows faces are probed by path with exists() rather
    than by catching truetype()'s failure -- an explicit fallback, not a
    swallowed error.
    """
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("consolab.ttf", "consola.ttf", "arialbd.ttf", "arial.ttf"):
        candidate = fonts_dir / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _rotate_point(x: float, y: float, w: int, h: int, deg: int) -> Tuple[float, float]:
    """Map a point in a (w, h) image to its position after cv2.rotate by `deg`."""
    deg %= 360
    if deg == 0:
        return x, y
    if deg == 90:                      # clockwise
        return (h - 1) - y, x
    if deg == 180:
        return (w - 1) - x, (h - 1) - y
    if deg == 270:
        return y, (w - 1) - x
    raise ValueError(f"display rotation must be a multiple of 90, got {deg}")


def _rotated_size(w: int, h: int, deg: int) -> Tuple[int, int]:
    return (h, w) if deg % 180 == 90 else (w, h)


def _image_of(obj) -> Optional[np.ndarray]:
    """Accept a types.Frame or a bare ndarray. Anything else is a wiring bug."""
    if obj is None:
        return None
    if isinstance(obj, Frame):
        return obj.image
    if isinstance(obj, np.ndarray):
        return obj
    raise TypeError(f"expected types.Frame or ndarray for a display frame, got {type(obj)!r}")


def _fmt(value: Optional[float], spec: str, dash: str = "--") -> str:
    if value is None:
        return dash
    return format(value, spec)


# ==========================================================================
#   TurretGUI
# ==========================================================================
class TurretGUI:
    """Spectator display + operator panel.

    Wiring (all optional, all called on the Tk main thread):

        on_start()              begin tracking
        on_stop()               stop tracking
        on_arm()                master-arm the laser (permission only; the
                                interlock still decides every frame)
        on_disarm()             drop the master arm
        on_estop()              vel 0 0, disarm, disable -- immediately
        on_clear_estop()        release the latched E-STOP
        on_home()               run homing.py's sequence
        on_calibrate(name)      name in {"jacobian", "goal_pixel"}
        on_shutdown()           window closing: stop threads, vel 0 0,
                                release cameras

    Feeding:

        set_status(SystemStatus)
        set_frames(narrow=..., wide=..., narrow_det=..., wide_det=...,
                   estimate=..., control=..., dot_px=...)
        set_homing_progress(text, fraction=..., done=..., failed=...)
        set_health(ok, text)
        log(text, level)
    """

    CALIBRATIONS = (("Calibrate Jacobian", "jacobian"), ("Calibrate Goal Pixel", "goal_pixel"))
    LOG_MAX_LINES = 500

    def __init__(
        self,
        *,
        title: str = "Laser Turret -- operator",
        status_slot: Optional[Slot] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_arm: Optional[Callable[[], None]] = None,
        on_disarm: Optional[Callable[[], None]] = None,
        on_estop: Optional[Callable[[], None]] = None,
        on_clear_estop: Optional[Callable[[], None]] = None,
        on_home: Optional[Callable[[], None]] = None,
        on_calibrate: Optional[Callable[[str], None]] = None,
        on_shutdown: Optional[Callable[[], None]] = None,
        confirm_arm: bool = True,
    ):
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_arm = on_arm
        self.on_disarm = on_disarm
        self.on_estop = on_estop
        self.on_clear_estop = on_clear_estop
        self.on_home = on_home
        self.on_calibrate = on_calibrate
        self.on_shutdown = on_shutdown
        self.confirm_arm = confirm_arm

        # ---- cross-thread inboxes -------------------------------------
        # The status slot may be the app's own publication slot: newest wins,
        # and a GUI that misses an intermediate status has lost nothing.
        self._status_slot: Slot = status_slot if status_slot is not None else Slot()
        self._status_seq_seen = -1
        self._status: SystemStatus = SystemStatus()

        self._frame_lock = threading.Lock()
        self._frame_state = {
            "narrow": None, "wide": None,
            "narrow_det": None, "wide_det": None,
            "estimate": None, "control": None, "dot_px": None,
        }
        self._frame_seq = 0
        self._frame_seq_drawn = -1

        self._log_lock = threading.Lock()
        self._log_pending: deque = deque(maxlen=2000)

        self._progress_lock = threading.Lock()
        self._progress = {
            "title": "HOMING", "text": "idle -- homing has not run", "fraction": 0.0,
            "active": False, "failed": False, "t0": 0.0, "dirty": True,
        }

        self._health_lock = threading.Lock()
        self._health = (None, "not checked")     # (ok|None, text)

        self._close_requested = threading.Event()

        # ---- operator state (main thread only) ------------------------
        self._running = False
        # Disarmed at launch, disarmed on every stop, disarmed on E-STOP.
        # config.LASER_ENABLED is deliberately NOT read here: the panel comes
        # up disarmed even if someone flips that default to True.
        self._armed = False
        self._estopped = False
        self._busy = False               # homing/calibration owns the platform
        self._tick_count = 0
        self._q_peak = 0.0
        self._nearest_face_px: Optional[float] = None
        self._range_source = "assumed"

        # Registered-view calibration. Optional on purpose: a missing file
        # disables the fused pane and changes nothing else, so a machine that
        # has never run the Gray-code capture still brings the panel up.
        self._fused = False
        self._w2n_h: Optional[np.ndarray] = None
        self._laser_wide: Optional[Tuple[float, float]] = None
        self._laser_narrow: Optional[Tuple[float, float]] = None
        self._w2n_rms = 0.0
        self._w2n_range = 0.0
        try:
            _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "calibration", "wide_narrow_homography.json")
            with open(_p, "r", encoding="utf-8") as _fh:
                _b = json.load(_fh)
            self._w2n_h = np.array(_b["H_wide_to_narrow"], float)
            self._w2n_rms = float(_b.get("rms_px", 0.0))
            self._w2n_range = float(_b.get("range_m", 0.0))
            _lw, _ln = _b.get("laser_wide_px"), _b.get("laser_narrow_px")
            self._laser_wide = tuple(_lw) if _lw else None
            self._laser_narrow = tuple(_ln) if _ln else None
        except (OSError, ValueError, KeyError, TypeError):
            self._w2n_h = None
        self._closing = False
        self._after_id: Optional[str] = None

        self._tick_ms = max(1, int(round(1000.0 / config.DISPLAY_FPS)))

        # ---- display geometry -----------------------------------------
        self._dw, self._dh = config.DISPLAY_SIZE
        # _rot is CLOCKWISE degrees. config resolved the direction on the live
        # preview (NARROW_ROTATE_CLOCKWISE), so honour the flag instead of
        # assuming: getting it backwards puts the pane upside down, which reads
        # as a camera fault during a demo.
        deg = int(config.NARROW_ROTATION_DEG) % 360
        self._rot = deg if config.NARROW_ROTATE_CLOCKWISE else (-deg) % 360
        if self._rot not in _ROTATE_CODES:
            raise ValueError(
                f"NARROW_ROTATION_DEG must be 0/90/180/270, got {config.NARROW_ROTATION_DEG}"
            )
        self._nw, self._nh = _rotated_size(self._dw, self._dh, self._rot)
        self._fit = 1.0                  # set once the screen size is known

        self._font_s = _overlay_font(12)
        self._font_m = _overlay_font(14)
        self._font_l = _overlay_font(18)

        self._photo_narrow: Optional[ImageTk.PhotoImage] = None
        self._photo_wide: Optional[ImageTk.PhotoImage] = None
        self._item_narrow: Optional[int] = None
        self._item_wide: Optional[int] = None

        # ---- build ----------------------------------------------------
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.report_callback_exception = self._report_callback_exception

        # DISPLAY_SIZE stays the drawing size: overlays, line widths and text
        # are all laid out against it, and the frame copy is always exactly
        # DISPLAY_SIZE. But the rotated narrow pane is 640 px tall on its own,
        # so on a short screen the finished image is scaled once at blit time.
        # Nothing upstream of the blit changes.
        chrome_px = 540                  # banner + controls + progress + log + padding
        available = self.root.winfo_screenheight() - 90 - chrome_px
        if available < self._nh:
            self._fit = max(0.5, available / float(self._nh))
            self.log(f"panel scaled to {self._fit:.2f} to fit a "
                     f"{self.root.winfo_screenheight()} px screen", "warn")
        self._pane_n = (max(1, int(self._nw * self._fit)), max(1, int(self._nh * self._fit)))
        self._pane_w = (max(1, int(self._dw * self._fit)), max(1, int(self._dh * self._fit)))

        self._build_style()
        self._build_banner()
        self._build_body()
        self._build_controls()
        self._build_progress()
        self._build_log()
        self._build_wrist_view()

        self.root.minsize(self._pane_n[0] + self._pane_w[0] + 60,
                          min(self._pane_n[1] + 400, self.root.winfo_screenheight() - 60))
        # Pin to the top of the screen: the panel is tall, and a window manager
        # that centres it pushes the log and the E-STOP off the bottom edge.
        self.root.geometry("+30+0")
        self._refresh_controls()
        self.log("GUI up. Laser DISARMED. Homing has not run.", "warn")

    # ------------------------------------------------------------------
    #   construction
    # ------------------------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        # 'clam' is the only stock ttk theme on Windows that honours colour
        # settings; the native theme ignores them and the panel would come up
        # half light-grey.
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=(FONT_UI, 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=FG, font=(FONT_UI, 10))
        style.configure("Key.TLabel", background=PANEL, foreground=MUTED, font=(FONT_MONO, 10))
        style.configure("Val.TLabel", background=PANEL, foreground=FG, font=(FONT_MONO, 11, "bold"))
        style.configure("Head.TLabel", background=BG, foreground=MUTED,
                        font=(FONT_UI, 9, "bold"))
        style.configure("Turret.Horizontal.TProgressbar", troughcolor=PANEL_HI,
                        background=CYAN, bordercolor=EDGE, lightcolor=CYAN, darkcolor=CYAN)

    def _button(self, parent, text, command, *, fg=FG, bg=PANEL_HI, font=(FONT_UI, 10, "bold"),
                width=16, height=1) -> tk.Button:
        # tk.Button, not ttk: ttk on Windows will not take a background colour,
        # and E-STOP being red is not decoration.
        return tk.Button(
            parent, text=text, command=command, font=font, width=width, height=height,
            fg=fg, bg=bg, activebackground=bg, activeforeground=fg,
            disabledforeground="#55606f", relief="flat", bd=0,
            highlightthickness=1, highlightbackground=EDGE, cursor="hand2",
        )

    def _build_banner(self) -> None:
        bar = tk.Frame(self.root, bg=PANEL)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        bar.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self._banner = tk.Frame(bar, bg="#243447", height=104)
        self._banner.grid(row=0, column=0, sticky="ew")
        self._banner.grid_propagate(False)
        self._banner.columnconfigure(0, weight=1)

        self._banner_text = tk.Label(self._banner, text="SEARCH", bg="#243447", fg=FG,
                                     font=(FONT_UI, 36, "bold"))
        self._banner_text.grid(row=0, column=0, sticky="ew", pady=(6, 0))
        self._banner_sub = tk.Label(self._banner, text="", bg="#243447", fg=MUTED,
                                    font=(FONT_UI, 11))
        self._banner_sub.grid(row=1, column=0, sticky="ew")

        side = tk.Frame(bar, bg=PANEL)
        side.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        tk.Label(side, text="LASER", bg=PANEL, fg=MUTED, font=(FONT_UI, 9, "bold")).pack()
        self._arm_lamp = tk.Label(side, text="DISARMED", bg=PANEL, fg=MUTED,
                                  font=(FONT_UI, 18, "bold"), width=11)
        self._arm_lamp.pack(pady=(2, 4))
        self._link_lamp = tk.Label(side, text="LINK: --", bg=PANEL, fg=MUTED,
                                   font=(FONT_MONO, 9))
        self._link_lamp.pack()

    def _build_body(self) -> None:
        body = ttk.Frame(self.root)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.root.rowconfigure(1, weight=1)

        # ---- narrow (rotated for display) -----------------------------
        left = ttk.Frame(body, style="Panel.TFrame")
        left.grid(row=0, column=0, sticky="n", padx=(0, 8))
        ttk.Label(left,
                  text=f"NARROW  {config.NARROW_SIZE[0]}x{config.NARROW_SIZE[1]}"
                       f"  (display rotated {self._rot} deg CW)",
                  style="Head.TLabel", background=PANEL).pack(anchor="w", padx=6, pady=(4, 2))
        self._canvas_narrow = tk.Canvas(left, width=self._pane_n[0], height=self._pane_n[1],
                                        bg="#05070a",
                                        highlightthickness=1, highlightbackground=EDGE)
        self._canvas_narrow.pack(padx=6, pady=(0, 6))

        # ---- wide + Q bar + telemetry ---------------------------------
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="n")

        wide_box = ttk.Frame(right, style="Panel.TFrame")
        wide_box.grid(row=0, column=0, sticky="ew")
        ttk.Label(wide_box,
                  text=f"WIDE  {config.WIDE_SIZE[0]}x{config.WIDE_SIZE[1]}  (all detections + faces)",
                  style="Head.TLabel", background=PANEL).pack(anchor="w", padx=6, pady=(4, 2))
        self._canvas_wide = tk.Canvas(wide_box, width=self._pane_w[0], height=self._pane_w[1],
                                      bg="#05070a",
                                      highlightthickness=1, highlightbackground=EDGE)
        self._canvas_wide.pack(padx=6, pady=(0, 6))

        # Adaptive-Q bar. A jink shows up as q slamming to the ceiling for a
        # few frames; the peak-hold marker keeps that visible long enough for
        # a human to see it at 30 fps.
        qbox = ttk.Frame(right, style="Panel.TFrame")
        qbox.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        head = tk.Frame(qbox, bg=PANEL)
        head.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(head, text="ADAPTIVE Q  (manoeuvre / jink)", bg=PANEL, fg=MUTED,
                 font=(FONT_UI, 9, "bold")).pack(side="left")
        self._q_value = tk.Label(head, text="0.00", bg=PANEL, fg=FG, font=(FONT_MONO, 9, "bold"))
        self._q_value.pack(side="right")
        self._qw = self._pane_w[0]
        self._canvas_q = tk.Canvas(qbox, width=self._qw, height=26, bg=PANEL_HI,
                                   highlightthickness=1, highlightbackground=EDGE)
        self._canvas_q.pack(padx=6, pady=(2, 6))
        self._q_fill = self._canvas_q.create_rectangle(0, 0, 0, 26, fill=GREEN, width=0)
        self._q_peak_line = self._canvas_q.create_line(0, 0, 0, 26, fill=WHITE, width=2)
        for frac in (0.25, 0.5, 0.75):
            x = frac * self._qw
            self._canvas_q.create_line(x, 0, x, 26, fill=EDGE, width=1)

        self._build_telemetry(right)

    def _build_telemetry(self, parent) -> None:
        box = ttk.Frame(parent, style="Panel.TFrame")
        box.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        ttk.Label(box, text="TELEMETRY", style="Head.TLabel",
                  background=PANEL).grid(row=0, column=0, columnspan=4, sticky="w",
                                         padx=6, pady=(4, 2))

        self._tel: dict[str, tk.Label] = {}
        left_rows = ("narrow fps", "wide fps", "loop Hz", "inference ms", "serial RTT ms")
        right_rows = ("pitch / yaw", "pixel error", "range", "faces", "mech health")

        for col, rows in ((0, left_rows), (2, right_rows)):
            for i, key in enumerate(rows):
                ttk.Label(box, text=key, style="Key.TLabel").grid(
                    row=1 + i, column=col, sticky="w", padx=(10, 6), pady=1)
                lbl = tk.Label(box, text="--", bg=PANEL, fg=FG, font=(FONT_MONO, 11, "bold"),
                               anchor="w", width=18)
                lbl.grid(row=1 + i, column=col + 1, sticky="w", padx=(0, 10), pady=1)
                self._tel[key] = lbl
        box.columnconfigure(1, weight=1)
        box.columnconfigure(3, weight=1)

        self._status_line = tk.Label(box, text="", bg=PANEL, fg=CYAN, font=(FONT_MONO, 10),
                                     anchor="w")
        self._status_line.grid(row=7, column=0, columnspan=4, sticky="ew", padx=10, pady=(4, 6))

    def _build_controls(self) -> None:
        bar = tk.Frame(self.root, bg=PANEL)
        bar.grid(row=2, column=0, sticky="ew", padx=8, pady=4)

        track = tk.Frame(bar, bg=PANEL)
        track.pack(side="left", padx=8, pady=8)
        tk.Label(track, text="TRACKING", bg=PANEL, fg=MUTED,
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")
        row = tk.Frame(track, bg=PANEL)
        row.pack()
        self._btn_start = self._button(row, "START", self._do_start, fg="#06210f", bg=GREEN, width=10)
        self._btn_start.pack(side="left", padx=(0, 4))
        self._btn_stop = self._button(row, "STOP", self._do_stop, width=10)
        self._btn_stop.pack(side="left")

        laser = tk.Frame(bar, bg=PANEL)
        laser.pack(side="left", padx=8, pady=8)
        tk.Label(laser, text="LASER MASTER ARM", bg=PANEL, fg=MUTED,
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")
        row = tk.Frame(laser, bg=PANEL)
        row.pack()
        self._btn_arm = self._button(row, "ARM", self._do_arm, fg="#2a0000", bg=AMBER, width=10)
        self._btn_arm.pack(side="left", padx=(0, 4))
        # DISARM is never confirmed and never disabled: undoing must always be
        # one unconditional click.
        self._btn_disarm = self._button(row, "DISARM", self._do_disarm, width=10)
        self._btn_disarm.pack(side="left")

        plat = tk.Frame(bar, bg=PANEL)
        plat.pack(side="left", padx=8, pady=8)
        tk.Label(plat, text="PLATFORM (idle only)", bg=PANEL, fg=MUTED,
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")
        row = tk.Frame(plat, bg=PANEL)
        row.pack()
        self._btn_home = self._button(row, "Run Homing", self._do_home, width=13)
        self._btn_home.pack(side="left", padx=(0, 4))
        self._btn_cal: list[tk.Button] = []
        for label, name in self.CALIBRATIONS:
            btn = self._button(row, label, lambda n=name: self._do_calibrate(n), width=17)
            btn.pack(side="left", padx=(0, 4))
            self._btn_cal.append(btn)

        # Deliberately not in the PLATFORM group: that one is disabled while
        # the loop runs, and watching the wrist move is most useful precisely
        # then.
        view = tk.Frame(bar, bg=PANEL)
        view.pack(side="left", padx=8, pady=8)
        tk.Label(view, text="VIEW", bg=PANEL, fg=MUTED,
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")
        row = tk.Frame(view, bg=PANEL)
        row.pack()
        self._btn_wrist = self._button(row, "3D Wrist", self._toggle_wrist_view, width=10)
        self._btn_wrist.pack(side="left")
        self._btn_fused = self._button(row, "FUSED", self._toggle_fused, width=10)
        self._btn_fused.pack(side="left", padx=(4, 0))
        if self._w2n_h is None:
            self._btn_fused.configure(state="disabled")

        stop = tk.Frame(bar, bg=PANEL)
        stop.pack(side="right", padx=8, pady=8)
        self._btn_estop = self._button(stop, "E-STOP", self._do_estop, fg=WHITE, bg="#b3000f",
                                       font=(FONT_UI, 20, "bold"), width=12, height=2)
        self._btn_estop.pack()
        self._btn_clear_estop = self._button(stop, "clear E-STOP", self._do_clear_estop,
                                             font=(FONT_UI, 9), width=12)
        self._btn_clear_estop.pack(pady=(4, 0))

    def _build_progress(self) -> None:
        box = tk.Frame(self.root, bg=PANEL)
        box.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        self._progress_title = tk.Label(box, text="HOMING", bg=PANEL, fg=MUTED,
                                        font=(FONT_UI, 9, "bold"), width=12, anchor="w")
        self._progress_title.pack(side="left", padx=(10, 6), pady=8)
        self._progress_bar = ttk.Progressbar(box, style="Turret.Horizontal.TProgressbar",
                                             orient="horizontal", length=380,
                                             mode="determinate", maximum=1.0, value=0.0)
        self._progress_bar.pack(side="left", pady=8)
        self._progress_text = tk.Label(box, text="idle", bg=PANEL, fg=FG, font=(FONT_MONO, 10),
                                       anchor="w")
        self._progress_text.pack(side="left", padx=10, pady=8, fill="x", expand=True)
        # The elapsed counter is the whole point of this strip: homing takes
        # tens of seconds and a frozen number is what "hung" looks like.
        self._progress_clock = tk.Label(box, text="", bg=PANEL, fg=MUTED,
                                        font=(FONT_MONO, 11, "bold"), width=10)
        self._progress_clock.pack(side="right", padx=10)

    def _build_log(self) -> None:
        box = tk.Frame(self.root, bg=PANEL)
        box.grid(row=4, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.root.rowconfigure(4, weight=0)
        self._log_text = tk.Text(box, height=7, bg="#0a0d12", fg=FG, font=(FONT_MONO, 9),
                                 relief="flat", bd=0, highlightthickness=1,
                                 highlightbackground=EDGE, wrap="none", state="disabled")
        self._log_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scroll = ttk.Scrollbar(box, orient="vertical", command=self._log_text.yview)
        scroll.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self._log_text.configure(yscrollcommand=scroll.set)
        self._log_text.tag_configure("info", foreground=FG)
        self._log_text.tag_configure("good", foreground=GREEN_HI)
        self._log_text.tag_configure("warn", foreground=AMBER)
        self._log_text.tag_configure("error", foreground=RED_HI)
        self._log_text.tag_configure("time", foreground=MUTED)

    # ------------------------------------------------------------------
    #   3D wrist view
    # ------------------------------------------------------------------
    def _build_wrist_view(self) -> None:
        """Open the live CAD view in its own small window, beside the panel.

        A separate Toplevel rather than a pane inside the body: the main window
        is already sized to the screen height (see `chrome_px` above, and the
        `_fit` scaling that exists because it does not always succeed), so
        another 280 px of content would push the E-STOP off the bottom on the
        laptop this runs on. It also means the view can be closed, moved to a
        second monitor, or never opened at all, without touching the layout
        that matters.
        """
        self._wrist_win = None
        self._wrist_view = None
        # Track visibility ourselves. `winfo_viewable()` is false whenever an
        # ancestor is unmapped, so a minimised main window would silently stop
        # the feed and the view would come back showing a stale pose.
        self._wrist_shown = False
        if _wristview is None or not _wristview.MESH_PATH.exists():
            if hasattr(self, "_btn_wrist"):
                self._btn_wrist.configure(state="disabled")
            self.log("3D wrist view unavailable -- run tools/export_wrist_mesh.py", "warn")
            return
        try:
            win = tk.Toplevel(self.root)
            win.title("WRIST -- live CAD")
            win.configure(bg=BG)
            win.resizable(False, False)
            # Closing the view must not close the app, and must leave the
            # button able to bring it back.
            win.protocol("WM_DELETE_WINDOW", self._hide_wrist_view)
            view = _wristview.WristView(win, size=(400, 280))
            view.pack(padx=6, pady=6)
            tk.Label(win, text="drag to orbit   ·   wheel to zoom", bg=BG, fg=MUTED,
                     font=(FONT_MONO, 8)).pack(pady=(0, 6))

            # Park it off the panel's right edge, but never off the screen.
            right = 30 + self._pane_n[0] + self._pane_w[0] + 70
            right = min(right, self.root.winfo_screenwidth() - 430)
            win.geometry(f"+{max(0, right)}+30")
            self._wrist_win, self._wrist_view = win, view
            self._wrist_shown = True
        except Exception as exc:                             # pragma: no cover
            self.log(f"3D wrist view failed to start: {exc}", "warn")
            self._wrist_win = self._wrist_view = None

    def _toggle_fused(self) -> None:
        if self._w2n_h is None:
            self.log("FUSED unavailable: calibration/wide_narrow_homography.json "
                     "is missing. Run tools/structured_light_map.py.", "warn")
            return
        self._fused = not self._fused
        self._btn_fused.configure(bg=CYAN if self._fused else PANEL_HI,
                                  fg="#00222c" if self._fused else FG)
        # Force a repaint: _draw_panes skips when the frame sequence has not
        # advanced, so without this the pane keeps the old rendering until the
        # next frame arrives -- which looks like the button did nothing.
        self._frame_seq_drawn = -1
        self.log("fused view %s (registered at %.2f m, RMS %.2f px)"
                 % ("ON" if self._fused else "OFF",
                    self._w2n_range, self._w2n_rms), "info")

    def _toggle_wrist_view(self) -> None:
        if self._wrist_win is None:
            return
        if self._wrist_shown:
            self._hide_wrist_view()
        else:
            self._wrist_win.deiconify()
            self._wrist_win.lift()
            self._wrist_shown = True

    def _hide_wrist_view(self) -> None:
        if self._wrist_win is not None:
            self._wrist_win.withdraw()
            self._wrist_shown = False

    def _feed_wrist_view(self) -> None:
        """Hand the view the pose the link last reported. Called every tick."""
        if self._wrist_view is None or not self._wrist_shown:
            return
        link = self._status.link
        # A dim beam always shows where the boresight points; a bright one
        # means the interlock actually let it fire.
        beam = 1.0 if self._status.laser is LaserState.FIRING else 0.22
        self._wrist_view.set_pose(link.pitch_deg, link.yaw_deg, beam)

    # ------------------------------------------------------------------
    #   public API -- safe from any thread
    # ------------------------------------------------------------------
    def set_status(self, status: SystemStatus) -> None:
        """Publish the newest SystemStatus. Newest wins; skipped ones are fine."""
        self._status_slot.put(status)

    def set_frames(
        self,
        narrow=_UNSET,
        wide=_UNSET,
        narrow_det=_UNSET,
        wide_det=_UNSET,
        estimate=_UNSET,
        control=_UNSET,
        dot_px=_UNSET,
    ) -> None:
        """Hand in display frames and the geometry to draw over them.

        Omitted arguments keep their previous value; passing None clears one.
        That distinction matters because the wide detector is throttled to
        `WIDE_SEARCH_FPS_TRACKING` while tracking -- a narrow-only update must
        not blank the wide pane.

        `narrow`/`wide` take a types.Frame (or a bare ndarray). The array is
        never written to: the draw path resizes into a fresh buffer first.
        `dot_px` is (u, v) in narrow source pixels, or None when the dot was
        not seen this frame -- it is opportunistic and gates nothing.
        """
        updates = {
            "narrow": narrow, "wide": wide,
            "narrow_det": narrow_det, "wide_det": wide_det,
            "estimate": estimate, "control": control, "dot_px": dot_px,
        }
        with self._frame_lock:
            for key, value in updates.items():
                if value is not _UNSET:
                    self._frame_state[key] = value
            self._frame_seq += 1

    def set_homing_progress(self, text: str, fraction: Optional[float] = None,
                            done: bool = False, failed: bool = False,
                            title: str = "HOMING") -> None:
        """Report a long platform task. `fraction` None => indeterminate."""
        now = time.perf_counter()
        with self._progress_lock:
            if not self._progress["active"] and not done:
                self._progress["t0"] = now
            self._progress.update(title=title, text=text, fraction=fraction,
                                  active=not done, failed=failed, dirty=True)
        self.log(f"{title}: {text}", "error" if failed else ("good" if done else "info"))

    def progress_callback(self, title: str = "HOMING") -> Callable[..., None]:
        """Adapter for the long-task progress hooks the other modules take.

        homing calls its hook as progress(text, frac); the calibration
        routines call theirs as progress(text). Both shapes land here, from
        whatever thread the task is running on. The app still has to report
        the end with `set_homing_progress(..., done=True)` -- that is what
        releases the platform buttons.
        """
        def progress(text: str, fraction: Optional[float] = None) -> None:
            self.set_homing_progress(str(text), fraction, title=title)
        return progress

    def set_health(self, ok: Optional[bool], text: str) -> None:
        """Mechanism health, e.g. homing's ripple-phase lost-step detector."""
        with self._health_lock:
            self._health = (ok, text)

    def log(self, text: str, level: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            self._log_pending.append((stamp, text, level))

    def request_close(self) -> None:
        """Ask the GUI to shut down. Safe from a worker thread."""
        self._close_requested.set()

    def run(self) -> None:
        """Enter the Tk loop. Returns once the window is gone."""
        self._tick()
        self.root.mainloop()

    # ------------------------------------------------------------------
    #   control actions
    # ------------------------------------------------------------------
    def _invoke(self, name: str, cb: Optional[Callable], *args) -> None:
        if cb is None:
            self.log(f"{name}: no handler wired", "warn")
            return
        # Deliberately unguarded. A handler that throws must surface through
        # report_callback_exception, not be absorbed into a no-op that leaves
        # the panel claiming something happened.
        cb(*args)

    def _do_start(self) -> None:
        if self._estopped:
            self.log("START refused: E-STOP latched. Clear it first.", "warn")
            return
        if self._busy:
            self.log("START refused: a platform task owns the motors.", "warn")
            return
        self._running = True
        self._refresh_controls()
        self.log("tracking START", "good")
        self._invoke("on_start", self.on_start)

    def _do_stop(self) -> None:
        self._running = False
        # Disarm on EVERY stop. Coming back up armed because the last session
        # was armed is exactly the surprise the interlock exists to prevent.
        if self._armed:
            self._armed = False
            self.log("laser DISARMED (tracking stopped)", "warn")
            self._invoke("on_disarm", self.on_disarm)
        self._refresh_controls()
        self.log("tracking STOP", "info")
        self._invoke("on_stop", self.on_stop)

    def _do_arm(self) -> None:
        if self._estopped:
            self.log("ARM refused: E-STOP latched.", "warn")
            return
        if self.confirm_arm:
            # Two deliberate acts to arm; one click to undo. messagebox spins a
            # nested Tk loop, so after() keeps running and the panes stay live
            # while the dialog is up.
            ok = messagebox.askokcancel(
                "Arm laser",
                "Master-arm the laser?\n\n"
                "The interlock still decides every frame: TRACK confirmed, no face "
                f"within {config.FACE_INHIBIT_MARGIN_PX} px, platform settled, "
                f"|e| < {config.MAX_ERROR_TO_FIRE_PX} px, recent vel.\n\n"
                "Eyes and reflective surfaces clear?",
                icon="warning", default="cancel", parent=self.root,
            )
            if not ok:
                self.log("arm cancelled", "info")
                return
        self._armed = True
        self._refresh_controls()
        self.log("laser ARMED -- interlock now has permission to fire", "warn")
        self._invoke("on_arm", self.on_arm)

    def _do_disarm(self) -> None:
        was = self._armed
        self._armed = False
        self._refresh_controls()
        if was:
            self.log("laser DISARMED", "good")
        self._invoke("on_disarm", self.on_disarm)

    def _do_estop(self) -> None:
        self._estopped = True
        self._running = False
        self._armed = False
        self._busy = False
        self._refresh_controls()
        self.log("E-STOP -- zero rates, disarmed, disabled", "error")
        # One call: the app is responsible for vel 0 0 + disarm + disable, in
        # that order, before anything else it might be doing.
        self._invoke("on_estop", self.on_estop)

    def _do_clear_estop(self) -> None:
        if not self._estopped:
            return
        self._estopped = False
        self._refresh_controls()
        self.log("E-STOP cleared. Still disarmed.", "warn")
        self._invoke("on_clear_estop", self.on_clear_estop)

    def _do_home(self) -> None:
        if not self._platform_free("homing"):
            return
        self._busy = True
        self._refresh_controls()
        self.set_homing_progress("starting", fraction=0.0, title="HOMING")
        self._invoke("on_home", self.on_home)

    def _do_calibrate(self, name: str) -> None:
        if not self._platform_free(f"calibration ({name})"):
            return
        self._busy = True
        self._refresh_controls()
        self.set_homing_progress("starting", fraction=0.0, title=f"CAL {name.upper()}")
        self._invoke("on_calibrate", self.on_calibrate, name)

    def _platform_free(self, what: str) -> bool:
        if self._estopped:
            self.log(f"{what} refused: E-STOP latched.", "warn")
            return False
        if self._running:
            self.log(f"{what} refused: stop tracking first -- it drives the motors.", "warn")
            return False
        if self._busy:
            self.log(f"{what} refused: another platform task is running.", "warn")
            return False
        return True

    def _refresh_controls(self) -> None:
        def state(enabled: bool) -> str:
            return "normal" if enabled else "disabled"

        idle = not self._running and not self._busy and not self._estopped
        self._btn_start.configure(state=state(idle))
        self._btn_stop.configure(state=state(self._running))
        self._btn_arm.configure(state=state(not self._armed and not self._estopped))
        self._btn_disarm.configure(state="normal")          # never blocked
        self._btn_home.configure(state=state(idle))
        for btn in self._btn_cal:
            btn.configure(state=state(idle))
        self._btn_clear_estop.configure(state=state(self._estopped))

        if self._armed:
            self._arm_lamp.configure(text="ARMED", fg=WHITE, bg="#b3000f")
        else:
            self._arm_lamp.configure(text="DISARMED", fg=MUTED, bg=PANEL)

    # ------------------------------------------------------------------
    #   tick
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        if self._close_requested.is_set():
            self._on_close()
            return
        try:
            self._tick_count += 1
            self._drain_log()
            self._pull_status()
            self._draw_panes()
            self._update_banner()
            self._update_telemetry()
            self._update_q_bar()
            self._update_progress()
            self._feed_wrist_view()
        finally:
            # Rescheduled in a finally so one bad frame cannot kill the clock
            # and freeze the panel while the turret is still tracking. The
            # exception is not swallowed -- it propagates to Tk's handler,
            # which prints the traceback and puts it in the log.
            self._after_id = self.root.after(self._tick_ms, self._tick)

    def _drain_log(self) -> None:
        with self._log_lock:
            if not self._log_pending:
                return
            pending = list(self._log_pending)
            self._log_pending.clear()
        self._log_text.configure(state="normal")
        for stamp, text, level in pending:
            self._log_text.insert("end", f"{stamp}  ", ("time",))
            self._log_text.insert("end", f"{text}\n", (level,))
        # Trim, or a long session grows the widget without bound.
        lines = int(self._log_text.index("end-1c").split(".")[0])
        if lines > self.LOG_MAX_LINES:
            self._log_text.delete("1.0", f"{lines - self.LOG_MAX_LINES}.0")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _pull_status(self) -> None:
        item, seq = self._status_slot.get()
        if seq == self._status_seq_seen or item is None:
            return
        self._status_seq_seen = seq
        if not isinstance(item, SystemStatus):
            raise TypeError(f"status slot carried {type(item)!r}, expected SystemStatus")
        self._status = item

    # ------------------------------------------------------------------
    #   panes
    # ------------------------------------------------------------------
    def _draw_panes(self) -> None:
        with self._frame_lock:
            seq = self._frame_seq
            if seq == self._frame_seq_drawn:
                return                      # nothing new; leave the last blit alone
            state = dict(self._frame_state)
        self._frame_seq_drawn = seq

        # SystemStatus carries range_m but not where it came from; the source
        # lives on TrackEstimate. "assumed" vs "size"/"stereo" is the
        # difference between a 3 m guess and a measurement, so it is shown.
        est_for_range: Optional[TrackEstimate] = state["estimate"]
        if est_for_range is not None:
            self._range_source = est_for_range.range_source

        narrow = _image_of(state["narrow"])
        wide = _image_of(state["wide"])

        pil_n = self._render_narrow(narrow, state)
        # The fused view replaces the WIDE pane, not the narrow one: it is
        # rendered in wide coordinates at the same 16:9 aspect, so it drops
        # into the same canvas without touching the layout, and the narrow
        # pane stays available at full resolution for the aiming work.
        if self._fused:
            pil_w = self._render_fused(narrow, wide, state)
        else:
            pil_w = self._render_wide(wide, state)

        self._photo_narrow, self._item_narrow = self._blit(
            self._canvas_narrow, self._item_narrow, self._photo_narrow, pil_n)
        self._photo_wide, self._item_wide = self._blit(
            self._canvas_wide, self._item_wide, self._photo_wide, pil_w)

    def _blit(self, canvas, item, photo, pil):
        if self._fit != 1.0:
            # Only on a screen too short for a full-size panel; overlays were
            # already drawn at DISPLAY_SIZE so they scale with the image.
            pil = pil.resize((max(1, int(pil.width * self._fit)),
                              max(1, int(pil.height * self._fit))), Image.BILINEAR)
        if photo is None or (photo.width(), photo.height()) != pil.size:
            photo = ImageTk.PhotoImage(pil)
            if item is not None:
                canvas.delete(item)
            item = canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            # paste() reuses the existing Tk image buffer; recreating a
            # PhotoImage 30x/s churns a megabyte a frame for no reason.
            photo.paste(pil)
        return photo, item

    def _blank(self, w: int, h: int, text: str) -> Image.Image:
        img = Image.new("RGB", (w, h), "#05070a")
        d = ImageDraw.Draw(img)
        d.text((w // 2, h // 2), text, fill=MUTED, font=self._font_l, anchor="mm")
        return img

    def _prepare(self, image: np.ndarray, rotate_deg: int) -> Image.Image:
        """Source frame -> DISPLAY_SIZE PIL copy, optionally rotated.

        cv2.resize allocates a new array, so from here down nothing can write
        through to the inference buffer we were handed.
        """
        small = cv2.resize(image, (self._dw, self._dh), interpolation=cv2.INTER_AREA)
        code = _ROTATE_CODES[rotate_deg % 360]
        if code is not None:
            # Rotating the 640x360 copy, never the 1280x720 source: the
            # geometry already carries the mount rotation, and a full-frame
            # rotate per frame buys nothing.
            small = cv2.rotate(small, code)
        if small.ndim == 2:
            rgb = cv2.cvtColor(small, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # -- coordinate mapping --------------------------------------------
    def _narrow_map(self, src_w: int, src_h: int):
        sx, sy = self._dw / float(src_w), self._dh / float(src_h)
        rot, dw, dh = self._rot, self._dw, self._dh

        def to_disp(x: float, y: float) -> Tuple[float, float]:
            return _rotate_point(x * sx, y * sy, dw, dh, rot)
        return to_disp, sx

    def _wide_map(self, src_w: int, src_h: int):
        sx, sy = self._dw / float(src_w), self._dh / float(src_h)

        def to_disp(x: float, y: float) -> Tuple[float, float]:
            return x * sx, y * sy
        return to_disp, sx

    @staticmethod
    def _box_disp(det: Detection, to_disp) -> Tuple[float, float, float, float]:
        # Rotation can swap which corner is which, so normalise after mapping.
        x1, y1 = to_disp(det.x1, det.y1)
        x2, y2 = to_disp(det.x2, det.y2)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    # -- overlay primitives --------------------------------------------
    def _label(self, d: ImageDraw.ImageDraw, x: float, y: float, text: str,
               fg: str, bg: str = "#000000", font=None, anchor: str = "ls") -> None:
        font = font or self._font_s
        box = d.textbbox((x, y), text, font=font, anchor=anchor)
        d.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=bg)
        d.text((x, y), text, fill=fg, font=font, anchor=anchor)

    @staticmethod
    def _cross(d: ImageDraw.ImageDraw, x: float, y: float, colour: str,
               size: int = 9, width: int = 2) -> None:
        d.line((x - size, y, x + size, y), fill=colour, width=width)
        d.line((x, y - size, x, y + size), fill=colour, width=width)

    # -- narrow ---------------------------------------------------------
    def _render_narrow(self, image: Optional[np.ndarray], state: dict) -> Image.Image:
        self._nearest_face_px = None
        if image is None:
            return self._blank(self._nw, self._nh, "NARROW: no frame")

        src_h, src_w = image.shape[:2]
        img = self._prepare(image, self._rot)
        d = ImageDraw.Draw(img)
        to_disp, scale = self._narrow_map(src_w, src_h)

        det: Optional[DetectionResult] = state["narrow_det"]
        est: Optional[TrackEstimate] = state["estimate"]
        ctl: Optional[ControlOutput] = state["control"]
        dot = state["dot_px"]

        inhibited = self._status.laser is LaserState.INHIBITED_FACE

        # Faces first, underneath everything: they are why the beam is off.
        if det is not None:
            for i, face in enumerate(det.faces):
                colour = RED if inhibited else "#c25b5b"
                x1, y1, x2, y2 = self._box_disp(face, to_disp)
                d.rectangle((x1, y1, x2, y2), outline=colour, width=3 if inhibited else 2)
                self._label(d, x1, y1 - 3, f"FACE {i}", WHITE, colour)

        # The beam lands at the goal pixel, so the inhibit test is measured
        # from there, not from the box centre.
        goal = (ctl.goal_u, ctl.goal_v) if ctl is not None else None
        if goal is not None and det is not None and det.faces:
            self._nearest_face_px = min(
                float(np.hypot(f.cx - goal[0], f.cy - goal[1])) for f in det.faces)

        if goal is not None:
            gx, gy = to_disp(*goal)
            ring = config.FACE_INHIBIT_MARGIN_PX * scale
            d.ellipse((gx - ring, gy - ring, gx + ring, gy + ring),
                      outline=RED if inhibited else "#3a4a5e", width=3 if inhibited else 1)
            fire_r = config.MAX_ERROR_TO_FIRE_PX * scale
            d.ellipse((gx - fire_r, gy - fire_r, gx + fire_r, gy + fire_r),
                      outline=AMBER, width=1)
            self._cross(d, gx, gy, AMBER, size=12, width=2)
            self._label(d, gx + 14, gy - 6, "GOAL", AMBER)

        box: Optional[Detection] = None
        if est is not None and est.box is not None:
            box = est.box
        elif det is not None and det.targets:
            box = max(det.targets, key=lambda t: t.conf)
        if box is not None:
            x1, y1, x2, y2 = self._box_disp(box, to_disp)
            colour = VIOLET if (est is not None and est.occluded) else GREEN
            d.rectangle((x1, y1, x2, y2), outline=colour, width=2)
            tag = f"{box.label} {box.conf:.2f}"
            if est is not None and est.occluded:
                tag += "  OCCLUDED"
            self._label(d, x1, y1 - 3, tag, "#06210f" if colour == GREEN else "#12081f", colour)

        if est is not None:
            px, py = to_disp(est.u, est.v)
            self._cross(d, px, py, MAGENTA, size=9, width=2)
            d.ellipse((px - 5, py - 5, px + 5, py + 5), outline=MAGENTA, width=2)
            self._label(d, px + 12, py + 14, f"pred {est.du:+.0f},{est.dv:+.0f} px/s", MAGENTA)
            if goal is not None:
                gx, gy = to_disp(*goal)
                d.line((gx, gy, px, py), fill=CYAN, width=1)

        if dot is not None:
            dx, dy = to_disp(float(dot[0]), float(dot[1]))
            d.ellipse((dx - 4, dy - 4, dx + 4, dy + 4), fill=YELLOW, outline="#000000")
            d.ellipse((dx - 9, dy - 9, dx + 9, dy + 9), outline=YELLOW, width=1)
            self._label(d, dx + 12, dy - 8, "DOT", "#2a2200", YELLOW)

        if inhibited:
            # Red frame around the whole pane. Redundant with the banner on
            # purpose -- this is the one state an operator must never miss.
            d.rectangle((1, 1, img.width - 2, img.height - 2), outline=RED, width=4)

        self._label(d, 6, img.height - 6,
                    f"NARROW  {self._status.narrow_fps:4.1f} fps"
                    f"  rot {self._rot} CW (display only)", FG, "#000000")
        return img

    # -- fused / registered ---------------------------------------------
    def _render_fused(self, narrow: Optional[np.ndarray],
                      wide: Optional[np.ndarray], state: dict) -> Image.Image:
        """Both cameras in ONE frame, registered, so a physical point lands in
        one place.

        Composited in WIDE pixel coordinates, not narrow, because the wide
        camera's field strictly contains the narrow one: the narrow view maps
        to a 540x951 box inside 1920x1080, so this direction crops nothing.
        Rendering in narrow coordinates would throw away most of the wide frame.

        The registration is the measured homography (wide -> narrow) from the
        projected Gray-code correspondence, inverted. A homography is the exact
        model for two views of ONE PLANE, which is what it was measured on --
        so this is exact on the calibration plane at 2.083 m, and degrades off
        it by the disparity, ~38 narrow px at that range falling as 1/R. The
        laser dots coincide here because the IMAGES are registered, not because
        anything is drawn on top of them.
        """
        if wide is None:
            return self._blank(self._dw, self._dh, "FUSED: no wide frame")
        if self._w2n_h is None:
            return self._blank(self._dw, self._dh,
                               "FUSED: no wide_narrow_homography.json")

        base = self._prepare(wide, 0)
        out = np.array(base)                       # RGB, DISPLAY_SIZE

        # display <- wide is a pure scale, so the display-to-narrow map is
        # H * S^-1. Folding the scale in here means warpPerspective samples the
        # FULL-RESOLUTION narrow frame straight onto the display raster, with
        # one interpolation instead of two.
        sx = self._dw / float(config.WIDE_SIZE[0])
        sy = self._dh / float(config.WIDE_SIZE[1])
        s_inv = np.array([[1.0 / sx, 0, 0], [0, 1.0 / sy, 0], [0, 0, 1.0]])
        m = self._w2n_h @ s_inv

        if narrow is not None:
            nrgb = cv2.cvtColor(narrow, cv2.COLOR_BGR2RGB)
            # WARP_INVERSE_MAP: dst(u,v) = src(m*(u,v)). m already goes
            # display -> narrow, so no inverse is computed here at all.
            warped = cv2.warpPerspective(
                nrgb, m, (self._dw, self._dh),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            cover = cv2.warpPerspective(
                np.full(narrow.shape[:2], 255, np.uint8), m,
                (self._dw, self._dh),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            a = (cover > 0)[:, :, None]
            # 50/50 inside the overlap. A blend rather than narrow-on-top on
            # purpose: misregistration shows up as a visible double image, so
            # the operator can SEE the calibration rather than trust it.
            out = np.where(a, (0.5 * out + 0.5 * warped).astype(np.uint8), out)

        img = Image.fromarray(out)
        d = ImageDraw.Draw(img)

        def to_disp(x: float, y: float) -> Tuple[float, float]:
            return x * sx, y * sy

        # Outline of the narrow field, mapped through the same transform.
        h_inv = np.linalg.inv(self._w2n_h)
        corners = np.array([[0, 0, 1], [config.NARROW_SIZE[0] - 1, 0, 1],
                            [config.NARROW_SIZE[0] - 1, config.NARROW_SIZE[1] - 1, 1],
                            [0, config.NARROW_SIZE[1] - 1, 1]], float)
        q = corners @ h_inv.T
        poly = [to_disp(p[0] / p[2], p[1] / p[2]) for p in q]
        d.line(poly + [poly[0]], fill=CYAN, width=2)
        self._label(d, poly[0][0] + 4, poly[0][1] + 14, "NARROW FIELD", CYAN)

        # The beam, as MEASURED in each camera independently. Two markers, not
        # one: if the calibration is right they sit on top of each other, and
        # the gap between them is the registration error made visible. Drawing
        # a single marker would hide exactly the thing worth watching.
        if self._laser_wide is not None:
            lx, ly = to_disp(*self._laser_wide)
            self._cross(d, lx, ly, GREEN, size=11, width=2)
            self._label(d, lx + 13, ly - 6, "BEAM wide", GREEN)
        if self._laser_narrow is not None:
            p = np.array([self._laser_narrow[0], self._laser_narrow[1], 1.0]) @ h_inv.T
            lx, ly = to_disp(p[0] / p[2], p[1] / p[2])
            d.ellipse((lx - 7, ly - 7, lx + 7, ly + 7), outline=YELLOW, width=2)
            self._label(d, lx + 13, ly + 12, "BEAM narrow", YELLOW)

        det: Optional[DetectionResult] = state["wide_det"]
        if det is not None:
            for t in det.targets:
                x1, y1 = to_disp(t.x1, t.y1)
                x2, y2 = to_disp(t.x2, t.y2)
                d.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                            outline=CYAN, width=2)

        self._label(d, 6, img.height - 6,
                    "FUSED  registered at %.2f m  RMS %.2f px"
                    % (self._w2n_range, self._w2n_rms), FG, "#000000")
        return img

    # -- wide -----------------------------------------------------------
    def _render_wide(self, image: Optional[np.ndarray], state: dict) -> Image.Image:
        if image is None:
            return self._blank(self._dw, self._dh, "WIDE: no frame")

        src_h, src_w = image.shape[:2]
        img = self._prepare(image, 0)
        d = ImageDraw.Draw(img)
        to_disp, _ = self._wide_map(src_w, src_h)

        det: Optional[DetectionResult] = state["wide_det"]
        n_faces = 0
        if det is not None:
            for t in det.targets:
                x1, y1, x2, y2 = self._box_disp(t, to_disp)
                d.rectangle((x1, y1, x2, y2), outline=CYAN, width=2)
                self._label(d, x1, y1 - 3, f"{t.label} {t.conf:.2f}", "#00222c", CYAN)
            for i, f in enumerate(det.faces):
                colour = FACE_COLOURS[i % len(FACE_COLOURS)]
                x1, y1, x2, y2 = self._box_disp(f, to_disp)
                d.rectangle((x1, y1, x2, y2), outline=colour, width=3)
                # Corner ticks make overlapping faces separable at 640x360.
                d.line((x1, y1, x1 + 10, y1), fill=WHITE, width=2)
                d.line((x1, y1, x1, y1 + 10), fill=WHITE, width=2)
                self._label(d, x1, y2 + 13, f"FACE {i}  {f.conf:.2f}", WHITE, colour)
            n_faces = len(det.faces)

        self._label(d, 6, img.height - 6,
                    f"WIDE  {self._status.wide_fps:4.1f} fps  faces {n_faces}", FG, "#000000")
        return img

    # ------------------------------------------------------------------
    #   banner / telemetry / q / progress
    # ------------------------------------------------------------------
    def _banner_style(self) -> Tuple[str, str, str, str, bool]:
        """(text, sub, bg, fg, flash) for the current status."""
        st = self._status
        laser, track = st.laser, st.track

        if laser is LaserState.INHIBITED_FACE:
            near = ("" if self._nearest_face_px is None
                    else f"  nearest face {self._nearest_face_px:.0f} px")
            sub = (f"{st.face_count} face(s) inside the {config.FACE_INHIBIT_MARGIN_PX} px "
                   f"margin -- beam held off{near}")
            return laser.value.upper(), sub, "#c1121f", WHITE, True
        if laser is LaserState.FIRING:
            return ("FIRING",
                    f"|e| {st.error_px:.1f} px  <  {config.MAX_ERROR_TO_FIRE_PX} px   "
                    f"range {st.range_m:.2f} m ({self._range_source})",
                    "#00a344", "#00140a", False)
        if laser in (LaserState.INHIBITED_ERROR, LaserState.INHIBITED_UNSETTLED):
            return (laser.value.upper(),
                    f"interlock holding fire  --  |e| {st.error_px:.1f} px",
                    "#7a5200", "#ffe6b0", False)
        if laser is LaserState.INHIBITED_NO_LOCK:
            # Deliberately not alarming: this is the resting state of a
            # fail-closed interlock, not a fault. No confirmed drone under the
            # beam means no permission, which is the normal case whenever the
            # detector is not looking straight at the target.
            return ("NO LOCK",
                    "no confirmed drone box under the beam -- no permission to "
                    "fire",
                    "#14313f", "#9fd4e8", False)

        if track is TrackState.TRACK:
            sub = f"|e| {st.error_px:.1f} px   range {st.range_m:.2f} m ({self._range_source})"
            return "TRACK", sub, "#0b5f2a", GREEN_HI, False
        if track is TrackState.ACQUIRE:
            return ("ACQUIRE", f"need {config.ACQUIRE_FRAMES} consecutive hits",
                    "#6b4c00", "#ffd66b", False)
        if track is TrackState.COAST:
            return ("COAST", f"target lost -- holding {config.COAST_HOLD_MS} ms then decaying",
                    "#3a2a5c", VIOLET, False)
        return "SEARCH", "no target", "#243447", "#9fb6d0", False

    def _update_banner(self) -> None:
        text, sub, bg, fg, flash = self._banner_style()
        if flash:
            # ~4 Hz. Fast enough to grab peripheral vision, slow enough to read.
            on = (self._tick_count // max(1, int(config.DISPLAY_FPS // 8))) % 2 == 0
            bg, fg = (bg, fg) if on else (WHITE, "#c1121f")
        if self._estopped:
            text, sub, bg, fg = "E-STOP", "latched -- clear it to re-enable", "#000000", RED
        self._banner.configure(bg=bg)
        self._banner_text.configure(text=text, bg=bg, fg=fg)
        self._banner_sub.configure(text=sub, bg=bg, fg=fg)

        link: LinkStatus = self._status.link
        if link.connected:
            self._link_lamp.configure(text=f"LINK {link.port or '?'}  {link.sent} sent",
                                      fg=GREEN_HI)
        else:
            self._link_lamp.configure(text="LINK: down", fg=RED_HI)

    def _update_telemetry(self) -> None:
        st = self._status
        link = st.link

        def put(key: str, text: str, colour: str = FG) -> None:
            self._tel[key].configure(text=text, fg=colour)

        def fps_colour(f: float) -> str:
            # MIN_ACCEPTABLE_FPS is the refuse-to-start floor, so anything
            # under it on a running system is already a fault, not a wobble.
            if f >= config.MIN_ACCEPTABLE_FPS:
                return GREEN_HI
            return AMBER if f > 0 else MUTED

        put("narrow fps", f"{st.narrow_fps:5.1f}", fps_colour(st.narrow_fps))
        put("wide fps", f"{st.wide_fps:5.1f}", fps_colour(st.wide_fps))

        # The watchdog is the reason this number matters: a command period
        # above VEL_COMMAND_PERIOD_MAX_MS means the motors park between frames.
        period_ms = (1000.0 / st.loop_hz) if st.loop_hz > 0 else float("inf")
        put("loop Hz", f"{st.loop_hz:5.1f}  ({period_ms:5.0f} ms)" if st.loop_hz > 0 else "--",
            GREEN_HI if period_ms <= config.VEL_COMMAND_PERIOD_MAX_MS else RED_HI)
        put("inference ms", f"{st.infer_ms:5.1f}")
        put("serial RTT ms", f"{link.rtt_ms:5.1f}",
            RED_HI if link.errors else (GREEN_HI if link.connected else MUTED))
        put("pitch / yaw", f"{link.pitch_deg:+7.2f} /{link.yaw_deg:+8.2f}")
        put("pixel error", f"{st.error_px:6.1f} px",
            GREEN_HI if st.error_px < config.MAX_ERROR_TO_FIRE_PX else AMBER)
        put("range", f"{st.range_m:5.2f} m  {self._range_source}")
        put("faces", f"{st.face_count:3d}", RED_HI if st.face_count else MUTED)

        with self._health_lock:
            ok, health_text = self._health
        put("mech health", health_text,
            MUTED if ok is None else (GREEN_HI if ok else RED_HI))

        self._status_line.configure(text=st.message or "")

    def _update_q_bar(self) -> None:
        q = float(min(1.0, max(0.0, self._status.q_level)))
        # Peak-hold decays over ~1.5 s so a single-frame jink is still on
        # screen by the time an eye gets there.
        self._q_peak = max(q, self._q_peak - 1.0 / (1.5 * config.DISPLAY_FPS))
        colour = GREEN if q < 0.34 else (AMBER if q < 0.67 else RED)
        self._canvas_q.coords(self._q_fill, 0, 0, q * self._qw, 26)
        self._canvas_q.itemconfigure(self._q_fill, fill=colour)
        px = self._q_peak * self._qw
        self._canvas_q.coords(self._q_peak_line, px, 0, px, 26)
        self._q_value.configure(text=f"{q:0.2f} of {config.Q_MAX_MULT:.0f}x q_base",
                                fg=colour)

    def _update_progress(self) -> None:
        with self._progress_lock:
            p = dict(self._progress)
            dirty = self._progress["dirty"]
            self._progress["dirty"] = False

        if dirty:
            self._progress_title.configure(text=p["title"])
            self._progress_text.configure(
                text=p["text"], fg=RED_HI if p["failed"] else (FG if p["active"] else GREEN_HI))
            if p["fraction"] is None and p["active"]:
                if self._progress_bar["mode"] != "indeterminate":
                    self._progress_bar.configure(mode="indeterminate")
                    self._progress_bar.start(40)
            else:
                if self._progress_bar["mode"] != "determinate":
                    self._progress_bar.stop()
                    self._progress_bar.configure(mode="determinate")
                frac = p["fraction"]
                if frac is None:
                    # Only reachable when the task is finished: a finished task
                    # reads full, and `or` would turn a legitimate 0.0 into it.
                    frac = 1.0
                self._progress_bar.configure(value=float(frac))
            # Busy follows the progress report in BOTH directions. The button
            # handlers set it when the operator starts a task, but app.py's
            # automatic startup homing is not operator-driven and still owns
            # the motors -- without this, START would be live during the 40 s
            # homing run, which is exactly when it must not be.
            if p["active"] and not self._busy:
                self._busy = True
                self._refresh_controls()
            elif not p["active"] and self._busy:
                self._busy = False
                self._refresh_controls()

        # Ticks every frame whether or not a step reported, which is what
        # distinguishes "slow" from "hung" during a 40 s homing run.
        if p["active"]:
            self._progress_clock.configure(text=f"{time.perf_counter() - p['t0']:6.1f} s",
                                           fg=AMBER)
        elif p["t0"]:
            self._progress_clock.configure(text=f"{self._progress_elapsed(p):6.1f} s", fg=MUTED)

    @staticmethod
    def _progress_elapsed(p: dict) -> float:
        return max(0.0, time.perf_counter() - p["t0"])

    # ------------------------------------------------------------------
    #   shutdown
    # ------------------------------------------------------------------
    def _report_callback_exception(self, exc, val, tb) -> None:
        # Tk's default handler prints and continues. Keep the printing (this
        # is the loud part) and mirror it into the operator log so a failure
        # during a demo is visible on the screen people are looking at.
        text = "".join(traceback.format_exception(exc, val, tb))
        sys.stderr.write(text)
        sys.stderr.flush()
        self.log(f"CALLBACK FAILED: {val!r}", "error")

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        try:
            # Safety first, in order: drop the arm, stop the loop, then let the
            # app stop threads / send vel 0 0 / release cameras. try/finally,
            # not try/except: a failing shutdown still propagates and prints,
            # it just cannot leave the window alive with the motors moving.
            if self._armed:
                self._armed = False
                self._invoke("on_disarm", self.on_disarm)
            if self._running:
                self._running = False
                self._invoke("on_stop", self.on_stop)
            self._invoke("on_shutdown", self.on_shutdown)
        finally:
            # Stop the view's render thread before the interpreter tears Tk
            # down, or it wakes up against a destroyed widget.
            if getattr(self, "_wrist_view", None) is not None:
                try:
                    self._wrist_view._stop()
                except Exception:
                    pass
            self.root.destroy()


# ==========================================================================
#   standalone layout check -- no camera, no board
# ==========================================================================
def _demo() -> None:
    """Drive the panel from synthetic frames on a worker thread.

    The worker exists to exercise the real path: every setter below is called
    off the main thread, exactly as the capture/control threads will call it.
    """
    dw_n, dh_n = config.NARROW_SIZE
    dw_w, dh_w = config.WIDE_SIZE

    def backdrop(w: int, h: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        grad = np.linspace(20, 70, h, dtype=np.float32)[:, None]
        base = np.repeat(grad, w, axis=1)
        base += rng.normal(0, 4, size=(h, w)).astype(np.float32)
        img = np.clip(base, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(0, w, 160):
            cv2.line(bgr, (i, 0), (i, h), (40, 46, 54), 1)
        return bgr

    bg_n = backdrop(dw_n, dh_n, 1)
    bg_w = backdrop(dw_w, dh_w, 2)

    stop = threading.Event()
    gui = TurretGUI(
        title="Laser Turret -- operator (SYNTHETIC, no hardware)",
        confirm_arm=False,
        on_start=lambda: gui.log("app: start tracking", "good"),
        on_stop=lambda: gui.log("app: stop tracking"),
        on_arm=lambda: gui.log("app: master arm set", "warn"),
        on_disarm=lambda: gui.log("app: master arm cleared"),
        on_estop=lambda: gui.log("app: vel 0 0 sent, laser disabled", "error"),
        on_clear_estop=lambda: gui.log("app: E-STOP released"),
        on_home=lambda: _fake_task(gui, "HOMING", [
            "link confirmed, platform idle", "imu cal -- hold still",
            "level -- driving to gravity datum", "yaw sweep for the motor field",
            "fitting -4.96 LSB/deg trend", "lash preload, one direction",
            "sethome"]),
        on_calibrate=lambda name: _fake_task(gui, f"CAL {name.upper()}",
                                             ["stepping motor A 200",
                                              "measuring pixel shift",
                                              "stepping motor B 200", "solving J"]),
        on_shutdown=lambda: (stop.set(), gui.log("app: threads stopped, cameras released")),
    )
    gui.set_health(True, "ripple phase OK")

    def feeder() -> None:
        t0 = time.perf_counter()
        i = 0
        q = 0.0
        while not stop.wait(1.0 / 30.0):
            t = time.perf_counter() - t0
            i += 1

            # The goal pixel is fixed (it is where the beam lands); a tracked
            # target hunts around it by a few tens of pixels, which is what
            # makes |e| cross MAX_ERROR_TO_FIRE_PX both ways during the run.
            goal_u, goal_v = dw_n * 0.5 + 24, dh_n * 0.5 - 18
            cx = goal_u + 55 * np.sin(t * 0.7) + 22 * np.sin(t * 1.9)
            cy = goal_v + 40 * np.sin(t * 1.1 + 1.0)
            half = 66
            n = bg_n.copy()
            cv2.rectangle(n, (int(cx - half), int(cy - half * 0.55)),
                          (int(cx + half), int(cy + half * 0.55)), (60, 190, 110), -1)
            cv2.circle(n, (int(cx), int(cy)), 8, (240, 240, 240), -1)

            target = Detection(cx - half, cy - half * 0.55, cx + half, cy + half * 0.55,
                               0.62, "drone")

            # A face sweeps straight through the beam path every 9 s -- the
            # inhibit demo. It crosses the goal at phase 1.5 s.
            face_phase = (t % 9.0)
            faces = []
            if face_phase < 3.0:
                fx = goal_u + (1.5 - face_phase) * 300
                fy = goal_v + 30
                cv2.rectangle(n, (int(fx - 55), int(fy - 70)), (int(fx + 55), int(fy + 70)),
                              (120, 140, 210), -1)
                faces.append(Detection(fx - 55, fy - 70, fx + 55, fy + 70, 0.88, "face"))

            err = float(np.hypot(cx - goal_u, cy - goal_v))

            # Jink every ~5 s: q spikes then decays at Q_DECAY, as the filter does.
            if abs((t % 5.0) - 0.0) < 1.0 / 30.0:
                q = 1.0
            q = max(0.06, 0.06 + config.Q_DECAY * (q - 0.06))

            if t < 2.0:
                track = TrackState.SEARCH
            elif t < 3.5:
                track = TrackState.ACQUIRE
            else:
                track = TrackState.TRACK

            nearest = min((float(np.hypot(f.cx - goal_u, f.cy - goal_v)) for f in faces),
                          default=1e9)
            if track is not TrackState.TRACK:
                laser = LaserState.DISARMED
            elif nearest < config.FACE_INHIBIT_MARGIN_PX:
                laser = LaserState.INHIBITED_FACE
            elif err < config.MAX_ERROR_TO_FIRE_PX:
                laser = LaserState.FIRING
            else:
                laser = LaserState.INHIBITED_ERROR

            w = bg_w.copy()
            wide_faces = []
            for k in range(2):
                fx = dw_w * (0.25 + 0.4 * k) + 60 * np.sin(t * 0.4 + k)
                fy = dh_w * 0.42
                cv2.rectangle(w, (int(fx - 70), int(fy - 90)), (int(fx + 70), int(fy + 90)),
                              (110, 130, 200), -1)
                wide_faces.append(Detection(fx - 70, fy - 90, fx + 70, fy + 90, 0.9, "face"))
            wx = dw_w * (0.5 + 0.3 * np.sin(t * 0.7))
            wy = dh_w * (0.5 + 0.2 * np.sin(t * 1.1 + 1.0))
            cv2.rectangle(w, (int(wx - 50), int(wy - 28)), (int(wx + 50), int(wy + 28)),
                          (60, 190, 110), -1)
            wide_targets = [Detection(wx - 50, wy - 28, wx + 50, wy + 28, 0.41, "drone")]

            # The dot is only present on alternating frames: LASER_PULSE_HZ is
            # half the capture rate, so it blinks in the pane exactly as the
            # dot detector sees it.
            dot = None
            if laser is LaserState.FIRING and i % 2 == 0:
                dot = (goal_u + np.random.uniform(-3, 3), goal_v + np.random.uniform(-3, 3))

            gui.set_frames(
                narrow=Frame(n, time.perf_counter(), i, "narrow"),
                wide=Frame(w, time.perf_counter(), i, "wide"),
                narrow_det=DetectionResult(time.perf_counter(), i, "narrow",
                                           [target], faces, 8.4),
                wide_det=DetectionResult(time.perf_counter(), i, "wide",
                                         wide_targets, wide_faces, 11.2),
                estimate=TrackEstimate(cx, cy, 120.0 * np.cos(t * 0.7), 90.0 * np.cos(t * 1.1),
                                       time.perf_counter(), track, q, 3.1,
                                       occluded=(t % 13.0) < 1.2, box=target,
                                       range_m=config.ASSUMED_RANGE_M,
                                       range_source="assumed"),
                control=ControlOutput(820.0, -410.0, cx - goal_u, cy - goal_v,
                                      goal_u, goal_v, False, dot is not None),
                dot_px=dot,
            )
            gui.set_status(SystemStatus(
                track=track, laser=laser,
                narrow_fps=30.1, wide_fps=30.3, loop_hz=29.4, infer_ms=8.4,
                q_level=q, error_px=err, range_m=config.ASSUMED_RANGE_M,
                face_count=len(faces),
                link=LinkStatus(True, "COM7", 3.2, i, 0, "", 12.4 + np.sin(t), -35.0 + 4 * np.sin(t * 0.6)),
                message="SYNTHETIC DATA -- no camera, no board attached",
            ))

    threading.Thread(target=feeder, name="demo-feeder", daemon=True).start()
    gui.log("synthetic feeder running at 30 fps", "good")
    gui.run()
    stop.set()


def _fake_task(gui: "TurretGUI", title: str, steps: Sequence[str]) -> None:
    """Step a progress bar without blocking the Tk loop (after(), not sleep())."""
    def step(i: int) -> None:
        if i >= len(steps):
            gui.set_homing_progress("datum established", fraction=1.0, done=True, title=title)
            return
        gui.set_homing_progress(steps[i], fraction=(i + 1) / len(steps), title=title)
        gui.root.after(1200, step, i + 1)
    step(0)


if __name__ == "__main__":
    _demo()
