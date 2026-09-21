"""Live 3D view of the wrist, driven by the pose the link reports.

**The pose overlay.** `render()` draws the machine at two poses at once: the
SOLID model where the payload measurably is, from the payload accelerometer,
and a cyan GHOST silhouette where the controller believes it is, from counting
motor steps. The gap between them is the mechanical error the control loop
cannot see -- backlash, lost steps, compliance, slip -- because the controller
is open loop on position and trusts its own count. `PoseInset` is this in a
corner of the wide camera pane; `gui.TurretGUI._pose_args` decides what to feed
it and `app.TurretApp._measured_pitch` produces the measurement.

Only PITCH is measured. An accelerometer sees gravity, and at pitch zero the
payload's yaw axis IS the gravity vector, so yaw cannot move the reading at
all; observability grows only as sin(pitch) and is negligible anywhere this
turret works. The ghost supplies yaw, and the HUD says so rather than drawing
a measured-looking yaw that can never disagree.


A small dark-mode viewport in the style of a MuJoCo scene: gradient sky,
fading checkerboard ground plane, soft projected shadow, flat-shaded solids.
It renders the real CAD -- `tools/export_wrist_mesh.py` reduces
`Full assm.step` to seven rigid bodies and about 13k triangles -- and moves
them with the differential's actual kinematics, so what you see is the machine,
not a cartoon of it.

**Why a software rasteriser.** The operator panel is Tk, which has no GL
surface, and pulling in an OpenGL toolkit for a 400 px inset would add a
windowing dependency to a program that must start reliably on a laptop in a
demo. Instead this draws into a numpy framebuffer and blits one PhotoImage per
frame, which is the same path `gui.py` already uses for camera panes.

The rasteriser is fully vectorised -- no Python loop over triangles. Each
triangle is intersected with the scanlines it touches to get exact horizontal
spans, those spans are expanded into pixels all at once, and the resulting
fragments are resolved by sorting them far-to-near and scattering them into the
framebuffer so the nearest one writes last and wins. That is a true z-buffer
and it costs a single argsort.

**The kinematics.** The bevel differential has no parent/child chain a scene
graph can express, so the composition is explicit (`_pose`):

    carrier  = Rz(pitch)                        the differential case
    head     = Rz(pitch) . Ry(yaw)              output gear + payload
    input_a  = Rz(pitch + yaw)                  60T pulley, motor A's bevel
    input_b  = Rz(pitch - yaw)                  60T pulley, motor B's bevel
    pulley_a = Rz(N (pitch + yaw)) about the motor shaft, N = 2.3077

Pitch is the outer joint: the yaw axis is carried by the carrier, which is why
`head` is a product and not two independent rotations. Driving pitch alone
turns both input pulleys the same way; driving yaw alone turns them opposite
ways and the case stays put -- which is the differential, visible.

Thread safety: `set_pose()` may be called from any thread and only stores
floats under a lock. `WristRenderer` is pure numpy and owns no Tk state, so
`WristView` runs it on a worker thread and the Tk callback only blits the
newest finished image.
"""
from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk

MESH_PATH = Path(__file__).resolve().parent / "assets" / "wrist_mesh.npz"

# Belt reduction, 26T -> 60T. Only used to spin the motor pulleys at the right
# rate relative to the bevels; the joint angles themselves arrive in degrees.
BELT_N = 2.3077

# ---- palette -------------------------------------------------------------
# Linear light: everything here is squared-down, because `render()` finishes
# with a sqrt as a cheap gamma. Pick a value by taking the 0..255 colour you
# want on screen, dividing by 255 and squaring.
SKY_TOP = np.array([0.009, 0.011, 0.015])
SKY_HORIZON = np.array([0.026, 0.030, 0.040])
FLOOR_A = np.array([0.030, 0.033, 0.040])
FLOOR_B = np.array([0.047, 0.051, 0.061])
CHECK_MM = 45.0                       # checker pitch in model mm
# Fog distance is set from the model's own size in `__init__` -- a constant in
# mm would be wrong the moment the CAD is rescaled, and at 900 mm it swallowed
# the entire ground plane one model-length from the camera.
FOG_SPANS = 14.0

LIGHT_DIR = np.array([-0.42, -0.86, -0.30])       # key, points along the rays
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
FILL_DIR = np.array([0.55, -0.25, 0.79])
FILL_DIR = FILL_DIR / np.linalg.norm(FILL_DIR)

AMBIENT = 0.30
KEY = 0.78
FILL = 0.22
RIM = 0.30

HUD_FG = (214, 226, 242)
HUD_DIM = (110, 126, 148)
HUD_ACCENT = (108, 196, 232)
HUD_WARN = (232, 176, 84)
HUD_BAD = (236, 108, 108)

# ---- ghost overlay -------------------------------------------------------
# The dead-reckoned pose, drawn over the measured one. Linear light, like the
# palette above. Cyan because the CAD is orange/white/blue-grey and a fourth
# hue has to be unmistakably NOT part of the machine.
GHOST_TINT = np.array([0.055, 0.34, 0.52])
GHOST_EDGE = np.array([0.32, 0.86, 1.00])
#: How strongly the ghost's exposed body tints the background. Low: it is a
#: hint of volume, not a second machine competing with the real one.
GHOST_FILL_A = 0.30
#: The silhouette, drawn X-RAY -- over the solid, not occluded by it. When the
#: two poses nearly agree the ghost is almost entirely hidden inside the solid,
#: and an occluded outline would vanish exactly when the reader most needs to
#: see that it is still there. The outline is the measurement; the fill is
#: decoration.
GHOST_EDGE_A = 0.92
# ---- rigid groups --------------------------------------------------------
# Several bodies share one matrix. They are separate only so they can be
# COLOURED separately -- `tools/export_wrist_mesh.py` splits the payload into
# gear, rear plate, IMU board, silkscreen, optical faceplate and shells, and
# the static side into base plate, frames, motors and end caps.
STATIC_BODIES = ("base_plate", "frame", "motor", "endcap")
CARRIER_BODIES = ("carrier",)
HEAD_BODIES = ("head_gear", "head_back", "head_imu", "head_silk",
               "head_face", "head_rest")

#: Which bodies get a ghost: the ones whose POSE depends on (pitch, yaw) in a
#: way a reader can see. The static side cannot diverge from itself, and the
#: bevels and pulleys only spin about fixed axes, so their outlines are
#: identical at both poses. See `WristRenderer._coverage`.
GHOST_BODIES = CARRIER_BODIES + HEAD_BODIES


def _font(size: int, bold: bool = False):
    for name in (("consolab.ttf", "consola.ttf") if bold else ("consola.ttf",)) + (
            "DejaVuSansMono.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _rgb8(linear: np.ndarray) -> Tuple[int, int, int]:
    """A linear-light palette entry as an 8-bit HUD colour, same gamma as the
    framebuffer resolve, so the label matches the pixels it describes."""
    v = np.sqrt(np.clip(linear, 0.0, 1.0)) * 255.0
    return (int(v[0]), int(v[1]), int(v[2]))


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix, right-handed, degrees."""
    a = axis / np.linalg.norm(axis)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


# ===========================================================================
#   renderer
# ===========================================================================
class WristRenderer:
    """Renders the wrist at a pose into an RGB uint8 array. No Tk, no threads."""

    def __init__(self, size: Tuple[int, int] = (430, 300), mesh_path: Path = MESH_PATH,
                 supersample: int = 2):
        self.w, self.h = size
        self.ss = max(1, int(supersample))
        self.rw, self.rh = self.w * self.ss, self.h * self.ss

        d = np.load(mesh_path, allow_pickle=False)
        self.bodies = [str(b) for b in d["bodies"]]
        self.verts = {b: d[f"{b}_v"].astype(np.float32) for b in self.bodies}
        self.faces = {b: d[f"{b}_f"].astype(np.int32) for b in self.bodies}
        self.color = {b: d[f"{b}_c"].astype(np.float32) for b in self.bodies}
        self.shadow_v = {b: d[f"{b}_sv"].astype(np.float32)
                         for b in self.bodies if f"{b}_sv" in d}
        self.shadow_f = {b: d[f"{b}_sf"].astype(np.int32)
                         for b in self.bodies if f"{b}_sf" in d}
        self.pitch_axis = d["pitch_axis"].astype(np.float64)
        self.yaw_axis = d["yaw_axis"].astype(np.float64)
        self.boresight = d["boresight"].astype(np.float64) if "boresight" in d else np.array([-1., 0., 0.])
        self.muzzle = d["muzzle"].astype(np.float64) if "muzzle" in d else np.zeros(3)
        self.floor_y = float(d["floor_y"])
        self.bounds = d["bounds"].astype(np.float64)

        # The motor pulleys turn about their own shafts, not the pitch axis.
        # Take each shaft from the pulley's own centroid: it is a disc, so its
        # centre lies on its axis, and the axis is parallel to pitch (+Z).
        self.shaft = {}
        for b in ("pulley_a", "pulley_b"):
            if b in self.verts:
                self.shaft[b] = self.verts[b].mean(axis=0).astype(np.float64)

        # One flat colour per triangle, and the model-space normals. Both are
        # constant; only the rotation changes per frame, so bake them now.
        self.tri_normal = {}
        for b in self.bodies:
            v, f = self.verts[b], self.faces[b]
            n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            self.tri_normal[b] = (n / np.maximum(ln, 1e-9)).astype(np.float32)

        self.span = float(np.linalg.norm(self.bounds[1] - self.bounds[0]))
        self.fog = self.span * FOG_SPANS
        self.fov = 34.0
        # Frame the whole machine: look at the centre of its bounds and stand
        # far enough back that the bounding sphere fits the vertical field,
        # with a margin. The model is 220 mm tall and 90 mm wide, so height is
        # always the binding constraint -- `fov` is the vertical one.
        self.target = self.bounds.mean(axis=0) * np.array([0.0, 1.0, 0.0])
        radius = 0.5 * float(np.linalg.norm(self.bounds[1] - self.bounds[0]))
        self.dist = radius / math.sin(math.radians(self.fov) * 0.5) * 0.78
        self.az = 38.0
        self.el = 16.0
        # BACK-FACE CULLING IS OFF, and the claim that used to justify it here
        # -- that the exporter leaves every body consistently wound and
        # positively oriented -- is measurably false. An edge audit of the
        # shipped mesh finds 1319 boundary and 1269 non-manifold edges: the
        # NEMA 17s and the GY-85 board arrive open from the tessellator, and
        # vertex clustering on the toothed pulleys adds its own. Culling those
        # shows straight THROUGH the part, and it only looked acceptable while
        # the whole static side was one grey mesh, where a gap reads as a
        # shadow. Coloured white it reads as a hole.
        #
        # Drawing both sides costs roughly 1.8x the fragments (111 -> 205 ms at
        # 430x300) and fills the gap with the part's own inside wall, which is
        # what the shading already expects -- `render` flips each normal toward
        # the eye. Correctness over frame time: nothing here is in the control
        # path, and the inset renders at 1x while moving.
        self.cull = False

        self._font_s = _font(11)
        self._font_m = _font(12, bold=True)
        # The sky and ground depend only on the camera, so they survive every
        # frame in which the operator is not dragging -- which is all of them.
        # Without this the per-pixel ray maths dominates the frame time.
        # Keyed by (resolution, camera), so switching supersample between
        # frames does not thrash it. Two or three entries is all it ever holds.
        self._bg_cache: dict = {}

    # -- camera ----------------------------------------------------------
    def clamp_camera(self, az: float, el: float, dist: float) -> Tuple[float, float, float]:
        """Legal camera, given a requested one. Pure -- safe off-thread.

        Elevation stops just above the ground plane (nothing below the floor is
        worth seeing) and short of straight down; distance is bounded by the
        model's own size so the operator cannot lose it.
        """
        return (az % 360.0,
                float(np.clip(el, -12.0, 78.0)),
                float(np.clip(dist, self.span * 0.55, self.span * 4.0)))

    def set_supersample(self, n: int) -> None:
        """Switch the internal render scale. Cheap: only the caches are keyed."""
        n = max(1, int(n))
        if n != self.ss:
            self.ss = n
            self.rw, self.rh = self.w * n, self.h * n

    def _camera(self):
        a, e = math.radians(self.az), math.radians(self.el)
        eye = self.target + self.dist * np.array(
            [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, np.stack([right, up, -fwd])          # rows: world -> view

    # -- pose ------------------------------------------------------------
    def _pose(self, pitch: float, yaw: float) -> dict:
        """Rotation matrix per body. See the module docstring for the algebra."""
        Rp = _rot(self.pitch_axis, pitch)
        Rh = Rp @ _rot(self.yaw_axis, yaw)
        eye = np.eye(3)
        out = {
            "input_a": _rot(self.pitch_axis, pitch + yaw),
            "input_b": _rot(self.pitch_axis, pitch - yaw),
            "pulley_a": _rot(self.pitch_axis, BELT_N * (pitch + yaw)),
            "pulley_b": _rot(self.pitch_axis, BELT_N * (pitch - yaw)),
        }
        for b in STATIC_BODIES:
            out[b] = eye
        for b in CARRIER_BODIES:
            out[b] = Rp
        for b in HEAD_BODIES:
            out[b] = Rh
        return out

    def _place(self, b: str, v: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Apply body `b`'s rotation to vertices `v`."""
        if b in self.shaft:
            # Spin about the motor shaft, which is offset from the origin.
            c = self.shaft[b]
            return (v - c) @ m.T + c
        return v @ m.T

    def _world(self, pitch: float, yaw: float):
        """Stack every body into one triangle soup in world space."""
        R = self._pose(pitch, yaw)
        V, N, C = [], [], []
        for b in self.bodies:
            m = R.get(b, np.eye(3))
            v = self._place(b, self.verts[b], m)
            V.append(v[self.faces[b]])                       # (T, 3, 3)
            N.append(self.tri_normal[b] @ m.T)
            C.append(np.repeat(self.color[b][None, :], len(self.faces[b]), axis=0))
        return (np.concatenate(V).astype(np.float32),
                np.concatenate(N).astype(np.float32),
                np.concatenate(C).astype(np.float32))

    def _world_shadow(self, pitch: float, yaw: float) -> np.ndarray:
        """The low-poly proxy, posed. Only the silhouette matters here."""
        R = self._pose(pitch, yaw)
        V = []
        for b in self.bodies:
            if b not in self.shadow_v:
                continue
            v = self._place(b, self.shadow_v[b], R.get(b, np.eye(3)))
            V.append(v[self.shadow_f[b]])
        return np.concatenate(V).astype(np.float32) if V else np.zeros((0, 3, 3), np.float32)

    # -- the rasteriser --------------------------------------------------
    @staticmethod
    def _fragments(p: np.ndarray, w: int, h: int):
        """Fragments for screen-space triangles `p` of shape (T,3,2).

        Returns (pixel_index, tri_index, barycentric) for every pixel covered
        by a triangle. Vectorised: no Python loop over triangles.

        Coverage is found by SCANLINE SPAN, not by bounding box. A decimated
        CAD mesh is full of long thin slivers whose bounding boxes are ~92 %
        empty -- measured at 1.43 M candidate pixels for 119 k real fragments --
        and testing that many candidates cost more than everything else in the
        frame combined. Intersecting each triangle with the scanlines it
        touches generates the covered pixels almost exactly.
        """
        ax = np.ascontiguousarray(p[:, 0, 0]); ay = np.ascontiguousarray(p[:, 0, 1])
        bx = np.ascontiguousarray(p[:, 1, 0]); by = np.ascontiguousarray(p[:, 1, 1])
        cx = np.ascontiguousarray(p[:, 2, 0]); cy = np.ascontiguousarray(p[:, 2, 1])

        y0 = np.clip(np.ceil(np.minimum(np.minimum(ay, by), cy) - 0.5), 0, h - 1).astype(np.int32)
        y1 = np.clip(np.floor(np.maximum(np.maximum(ay, by), cy) - 0.5), 0, h - 1).astype(np.int32)
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        keep = ((y1 - y0) >= 0) & (np.abs(det) > 1e-9) & \
               (np.maximum(np.maximum(ax, bx), cx) >= 0) & \
               (np.minimum(np.minimum(ax, bx), cx) <= w - 1)
        if not keep.any():
            return None
        idx = np.nonzero(keep)[0]

        # Barycentrics are affine in screen space, so fold them into
        # l = A*x + B*y + C per edge and gather the coefficients once. This is
        # the whole optimisation: the naive form re-gathers six vertex
        # components per candidate pixel and subtracts them every time.
        inv = (1.0 / det[idx]).astype(np.float32)
        cxi, cyi = cx[idx], cy[idx]
        A0 = (by[idx] - cyi) * inv
        B0 = (cxi - bx[idx]) * inv
        C0 = -(A0 * cxi + B0 * cyi)
        A1 = (cyi - ay[idx]) * inv
        B1 = (ax[idx] - cxi) * inv
        C1 = -(A1 * cxi + B1 * cyi)

        # ---- one entry per (triangle, scanline it touches) ----------------
        nrow = (y1 - y0 + 1)[idx].astype(np.int64)
        tri = np.repeat(np.arange(len(idx), dtype=np.int32), nrow)
        start = np.zeros(len(nrow), np.int64)
        np.cumsum(nrow[:-1], out=start[1:])
        py = (np.arange(int(nrow.sum()), dtype=np.int64) - np.repeat(start, nrow)
              + np.repeat(y0[idx].astype(np.int64), nrow))
        fy = py.astype(np.float32) + 0.5

        # Gather the vertices once and reuse them for all three edges.
        gax, gay = ax[idx][tri], ay[idx][tri]
        gbx, gby = bx[idx][tri], by[idx][tri]
        gcx, gcy = cx[idx][tri], cy[idx][tri]

        lo = np.full(len(fy), np.inf, np.float32)
        hi = np.full(len(fy), -np.inf, np.float32)
        for (X1, Y1, X2, Y2) in ((gax, gay, gbx, gby), (gbx, gby, gcx, gcy),
                                 (gcx, gcy, gax, gay)):
            dy = Y2 - Y1
            # Half-open in y so a vertex shared by two edges is counted once.
            on = (fy >= np.minimum(Y1, Y2)) & (fy < np.maximum(Y1, Y2)) & (dy != 0)
            x = X1 + (fy - Y1) * (X2 - X1) / np.where(dy == 0, 1.0, dy)
            lo = np.minimum(lo, np.where(on, x, np.inf))
            hi = np.maximum(hi, np.where(on, x, -np.inf))

        px0 = np.clip(np.ceil(lo - 0.5), 0, w - 1).astype(np.int32)
        px1 = np.clip(np.floor(hi - 0.5), 0, w - 1).astype(np.int32)
        span = (px1 - px0 + 1).astype(np.int64)
        good = np.isfinite(lo) & np.isfinite(hi) & (span > 0)
        if not good.any():
            return None
        tri, py, px0, span = tri[good], py[good], px0[good], span[good]

        # ---- expand the spans into pixels --------------------------------
        total = int(span.sum())
        tri = np.repeat(tri, span)
        start = np.zeros(len(span), np.int64)
        np.cumsum(span[:-1], out=start[1:])
        px = (np.arange(total, dtype=np.int64) - np.repeat(start, span)
              + np.repeat(px0.astype(np.int64), span))
        py = np.repeat(py, span)

        fx = px.astype(np.float32) + 0.5
        fy = py.astype(np.float32) + 0.5
        l0 = A0[tri] * fx + B0[tri] * fy + C0[tri]
        l1 = A1[tri] * fx + B1[tri] * fy + C1[tri]
        # Spans are exact up to rounding; clamp rather than discard, so a pixel
        # that lands a hair outside still shades instead of punching a hole.
        l0 = np.clip(l0, 0.0, 1.0)
        l1 = np.clip(l1, 0.0, 1.0 - l0)
        return (py * w + px, idx[tri],
                np.stack([l0, l1, 1.0 - l0 - l1], axis=1))

    def _ground(self, eye, M):
        """Sky gradient + fading checkerboard, as a full-screen background.

        Cached on the camera. Also returns the ground hit distance per pixel so
        the shadow pass knows which pixels are ground at all.
        """
        key = (self.rw, self.rh, round(self.az, 3), round(self.el, 3), round(self.dist, 2))
        hit = self._bg_cache.get(key)
        if hit is not None:
            return hit
        w, h = self.rw, self.rh

        # Ray directions for every pixel, in world space.
        f = 0.5 * h / math.tan(math.radians(self.fov) * 0.5)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.stack([(xx + 0.5 - w * 0.5) / f,
                      -(yy + 0.5 - h * 0.5) / f,
                      -np.ones_like(xx)], axis=-1)
        d = d @ M.astype(np.float32)                  # view -> world; M is orthonormal
        d /= np.linalg.norm(d, axis=-1, keepdims=True)

        # Sky: vertical gradient on the ray's elevation.
        t = np.clip(d[..., 1] * 1.9 + 0.20, 0.0, 1.0)[..., None]
        img = (SKY_HORIZON * (1 - t) + SKY_TOP * t).astype(np.float32)

        hit = np.where(d[..., 1] < -1e-6, (self.floor_y - eye[1]) / d[..., 1], -1.0)
        on = hit > 0
        if on.any():
            P = eye + np.where(on, hit, 0.0)[..., None] * d
            chk = (np.floor(P[..., 0] / CHECK_MM) + np.floor(P[..., 2] / CHECK_MM)) % 2
            floor = np.where(chk[..., None] > 0.5, FLOOR_B, FLOOR_A)
            # Fade the checker toward its own mean before fogging: past a few
            # model-lengths the squares are sub-pixel and would alias into
            # noise. Then fade the plane into the horizon so there is no seam.
            soft = np.clip(hit / (self.fog * 0.30), 0, 1)[..., None]
            floor = floor * (1 - soft) + (FLOOR_A + FLOOR_B) * 0.5 * soft
            fog = np.clip(hit / self.fog, 0, 1)[..., None]
            floor = floor * (1 - fog) + SKY_HORIZON * fog
            img = np.where(on[..., None], floor, img).astype(np.float32)

        if len(self._bg_cache) > 4:
            self._bg_cache.clear()
        self._bg_cache[key] = (img, hit, on)
        return self._bg_cache[key]

    def _shadow(self, tri_v, eye, M, img, on):
        """Project the solids onto the ground along the key light and darken it.

        Rasterised at half resolution into a coverage mask and blurred, which
        is cheap and gives the soft edge a real scene has.
        """
        L = LIGHT_DIR
        if abs(L[1]) < 1e-3:
            return img
        v = tri_v.reshape(-1, 3)
        t = (self.floor_y - v[:, 1]) / L[1]
        g = v + t[:, None] * L
        g[:, 1] = self.floor_y
        # Quarter resolution. The mask is blurred to a soft edge anyway, so
        # rasterising it at full size buys nothing; and blurring BEFORE the
        # upscale makes the blur itself sixteen times cheaper.
        sw, sh = self.rw // 4, self.rh // 4
        scr, z = self._project(g.reshape(-1, 3, 3), eye, M, sw, sh)
        vis = (z > 1e-3).all(axis=1)
        if not vis.any():
            return img
        frag = self._fragments(scr[vis], sw, sh)
        if frag is None:
            return img
        mask = np.zeros(sh * sw, np.uint8)
        mask[frag[0]] = 255
        mask = mask.reshape(sh, sw)

        # Blur, upscale and composite only inside the shadow's own bounding
        # box. The shadow covers maybe a fifth of the frame, and doing the
        # full-screen resize plus multiply was costing more than rasterising
        # it. Padding leaves room for the blur to fall off.
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return img
        pad = 5
        x0, x1 = max(0, xs.min() - pad), min(sw, xs.max() + 1 + pad)
        y0, y1 = max(0, ys.min() - pad), min(sh, ys.max() + 1 + pad)
        m = Image.fromarray(mask[y0:y1, x0:x1]).filter(ImageFilter.GaussianBlur(1.6))

        dx0, dx1 = round(x0 * self.rw / sw), round(x1 * self.rw / sw)
        dy0, dy1 = round(y0 * self.rh / sh), round(y1 * self.rh / sh)
        m = m.resize((dx1 - dx0, dy1 - dy0), Image.BILINEAR)
        a = (np.asarray(m, np.float32) / 255.0)[..., None] * 0.62

        img = img.copy()
        sub = img[dy0:dy1, dx0:dx1]
        img[dy0:dy1, dx0:dx1] = sub - sub * np.where(on[dy0:dy1, dx0:dx1, None], a, 0.0)
        return img

    def _project(self, tri, eye, M, w, h):
        """World triangles -> screen pixels + view depth."""
        v = (tri.reshape(-1, 3) - eye) @ M.T
        z = -v[:, 2]
        f = 0.5 * h / math.tan(math.radians(self.fov) * 0.5)
        zc = np.where(z < 1e-3, 1e-3, z)
        sx = v[:, 0] / zc * f + w * 0.5
        sy = -v[:, 1] / zc * f + h * 0.5
        return np.stack([sx, sy], 1).reshape(-1, 3, 2), z.reshape(-1, 3)

    def _coverage(self, pitch: float, yaw: float, eye, M) -> np.ndarray:
        """Silhouette of the whole machine at `pitch, yaw`, as a bool mask.

        Coverage only -- no shading, no depth sort, no shadow. That is what
        makes a second pose cheap enough to draw every frame: the ghost costs
        one projection and one fragment pass, against the shading, sorting and
        scatter the solid pass also pays for.

        DELIBERATELY NOT BACK-FACE CULLED, unlike `render`. The union of the
        front-facing triangles is the silhouette only for a closed body, and
        `export_wrist_mesh.py` leaves several of these open (A frame: 1
        boundary edge, B frame: 26 -- the mirrored instance tessellates more
        coarsely). Taking every triangle makes the outline exact regardless.

        Only `GHOST_BODIES`. The base cannot diverge from itself, and the
        pulleys and input bevels only SPIN -- their silhouettes are the same
        at both poses, so outlining them draws a bright line that never moves
        and buries the one that does. Restricting to the two bodies that
        actually change pose also takes this from 246 ms to well under the
        solid pass, which is what makes a second pose affordable per frame.
        """
        R = self._pose(pitch, yaw)
        V = [self._place(b, self.verts[b], R.get(b, np.eye(3)))[self.faces[b]]
             for b in self.bodies if b in GHOST_BODIES]
        if not V:
            return np.zeros((self.rh, self.rw), bool)
        tri_v = np.concatenate(V).astype(np.float32)
        scr, z = self._project(tri_v, eye, M, self.rw, self.rh)
        idx = np.nonzero((z > 1e-3).all(axis=1))[0]
        mask = np.zeros(self.rh * self.rw, bool)
        if len(idx):
            frag = self._fragments(scr[idx], self.rw, self.rh)
            if frag is not None:
                mask[frag[0]] = True
        return mask.reshape(self.rh, self.rw)

    @staticmethod
    def _outline(mask: np.ndarray, grow: int = 1) -> np.ndarray:
        """The boundary of `mask`, `grow` pixels thick, by shifted ANDs.

        scipy would give this in one call and is not a dependency of this
        program; four shifts are the whole algorithm anyway. `grow` is in
        RENDER pixels, so it is passed `self.ss` and the line survives the
        supersample resolve at roughly one screen pixel.
        """
        er = mask.copy()
        er[1:, :] &= mask[:-1, :]
        er[:-1, :] &= mask[1:, :]
        er[:, 1:] &= mask[:, :-1]
        er[:, :-1] &= mask[:, 1:]
        edge = mask & ~er
        for _ in range(max(0, grow - 1)):
            g = edge.copy()
            g[1:, :] |= edge[:-1, :]
            g[:-1, :] |= edge[1:, :]
            g[:, 1:] |= edge[:, :-1]
            g[:, :-1] |= edge[:, 1:]
            edge = g
        return edge

    def render(self, pitch: float, yaw: float, *, beam: float = 0.0,
               label: str = "", sub: str = "",
               ghost: Optional[Tuple[float, float]] = None,
               hud: Optional[dict] = None) -> Image.Image:
        """Draw the machine at `pitch, yaw`, optionally with a `ghost` pose.

        `pitch, yaw` is the SOLID model -- where the payload actually is.
        `ghost` is `(pitch, yaw)` of the dead-reckoned pose the controller
        believes it is holding, drawn as a cyan silhouette over the top. Pass
        `ghost=None` to draw one pose, which is also the honest thing to do
        when the IMU has stopped answering: see `WristView.set_pose`.
        """
        eye, M = self._camera()
        bg, _hit, on = self._ground(eye, M)

        tri_v, tri_n, tri_c = self._world(pitch, yaw)
        img = self._shadow(self._world_shadow(pitch, yaw), eye, M, bg, on)
        if img is bg:                     # no shadow drawn; do not scribble on the cache
            img = bg.copy()

        scr, z = self._project(tri_v, eye, M, self.rw, self.rh)
        # Drop anything with a vertex behind the eye rather than clipping it:
        # at this framing nothing straddles the near plane, and a real clipper
        # is a lot of code for a case that cannot happen.
        vis = (z > 1e-3).all(axis=1)
        if self.cull:
            e = scr[:, 1] - scr[:, 0]
            g = scr[:, 2] - scr[:, 0]
            vis &= (e[:, 0] * g[:, 1] - e[:, 1] * g[:, 0]) > 0
        idx = np.nonzero(vis)[0]
        solid_mask = np.zeros((self.rh, self.rw), bool)
        if len(idx):
            # Shade PER TRIANGLE, not per fragment. The mesh is flat-shaded, so
            # every fragment of a triangle resolves to the same colour anyway,
            # and there are ten times fewer triangles than fragments.
            n = tri_n[idx]
            cen = tri_v[idx].mean(axis=1)
            view = eye - cen
            view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
            # Two-sided: a shell seen from inside must light like a surface,
            # not like a black hole. Flip the normal towards the eye.
            n = np.where(((n * view).sum(1) < 0)[:, None], -n, n)
            nd = np.clip(n @ (-LIGHT_DIR), 0.0, None)
            nf = np.clip(n @ (-FILL_DIR), 0.0, None)
            rim = np.clip(1.0 - np.abs((n * view).sum(1)), 0, 1) ** 2.4
            # Blinn-Phong highlight, narrow and weak: enough to read the
            # curvature of a printed part without looking like plastic wrap.
            hv = -LIGHT_DIR + view
            hv /= np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-9)
            spec = np.clip((n * hv).sum(1), 0, 1) ** 42 * 0.34
            col = (tri_c[idx] * (AMBIENT + KEY * nd + FILL * nf)[:, None]
                   + rim[:, None] * RIM * np.array([0.40, 0.56, 0.78], np.float32)
                   + spec[:, None]).astype(np.float32)

            frag = self._fragments(scr[idx], self.rw, self.rh)
            if frag is not None:
                pix, tri, bary = frag
                depth = (bary * z[idx][tri]).sum(axis=1)
                # Resolve occlusion by scattering far-to-near: the nearest
                # fragment for a pixel is written last and wins. One argsort,
                # no z-buffer read-modify-write.
                order = np.argsort(-depth, kind="stable")
                flat = img.reshape(-1, 3)
                flat[pix[order]] = col[tri[order]]
                img = flat.reshape(self.rh, self.rw, 3)
                solid_mask = np.zeros(self.rh * self.rw, bool)
                solid_mask[pix] = True
                solid_mask = solid_mask.reshape(self.rh, self.rw)

        # ---- the ghost -------------------------------------------------
        # Composited in LINEAR light, before the gamma below, so the blend
        # matches the shading it sits on rather than darkening against it.
        if ghost is not None:
            gmask = self._coverage(ghost[0], ghost[1], eye, M)
            edge = self._outline(gmask, grow=self.ss)
            # Fill only where the ghost is NOT behind the solid: a translucent
            # wash over the machine itself would read as fog on the lens. Where
            # the two poses agree this leaves nothing but the outline, which is
            # exactly right -- no divergence, nothing to see.
            fill = gmask & ~solid_mask & ~edge
            img[fill] = img[fill] * (1.0 - GHOST_FILL_A) + GHOST_TINT * GHOST_FILL_A
            img[edge] = img[edge] * (1.0 - GHOST_EDGE_A) + GHOST_EDGE * GHOST_EDGE_A

        # Resolve supersampling by averaging in LINEAR light, before gamma --
        # which is what makes the edges read as smooth rather than as a row of
        # half-lit pixels. Also four times less work for the sqrt below.
        if self.ss > 1:
            s = self.ss
            img = img.reshape(self.h, s, self.w, s, 3).mean(axis=(1, 3))
        pil = Image.fromarray((np.sqrt(np.clip(img, 0, 1)) * 255.0).astype(np.uint8))
        if beam > 0.0:
            pil = self._beam(pil, eye, M, pitch, yaw, beam)
        self._hud(pil, pitch, yaw, label, sub, ghost, hud or {})
        return pil

    def _beam(self, pil: Image.Image, eye, M, pitch, yaw, strength) -> Image.Image:
        """Add the boresight as a glowing green line -- the laser, in air.

        Drawn at final resolution, after the supersample is resolved: a glow is
        already soft, so rendering it at 2x and shrinking it only costs time.
        """
        R = self._pose(pitch, yaw)[HEAD_BODIES[0]]   # any head body: one matrix
        o = R @ self.muzzle
        pts = np.stack([o, o + (R @ self.boresight) * self.span * 5.0])
        v = (pts - eye) @ M.T
        if (-v[:, 2] <= 1e-3).any():
            return pil
        f = 0.5 * self.h / math.tan(math.radians(self.fov) * 0.5)
        sx = v[:, 0] / (-v[:, 2]) * f + self.w * 0.5
        sy = -v[:, 1] / (-v[:, 2]) * f + self.h * 0.5

        # 8-bit, not float: PIL's Gaussian blur rejects mode "F".
        layer = Image.new("L", (self.w, self.h), 0)
        ImageDraw.Draw(layer).line([(sx[0], sy[0]), (sx[1], sy[1])], fill=255, width=2)
        glow = layer.filter(ImageFilter.GaussianBlur(2.6))
        a = (np.asarray(layer, np.float32) + np.asarray(glow, np.float32) * 2.1) / 255.0
        base = np.asarray(pil, np.float32)
        lit = base + (a[..., None] * np.array([40.0, 225.0, 80.0], np.float32)
                      * (0.62 * float(strength)))
        return Image.fromarray(np.clip(lit, 0, 255).astype(np.uint8))

    #: |pitch divergence| above which the gap is called out in amber. Measured
    #: backlash is 0.65 deg near level and 2.0-2.3 deg at 20-40 deg of payload
    #: pitch (bench, 2026-09-20), so a threshold below ~1 deg would sit in
    #: amber permanently and mean nothing. This is set to flag a gap that is
    #: large even for a reversal.
    DIVERGE_WARN_DEG = 2.5

    def _hud(self, pil: Image.Image, pitch, yaw, label, sub,
             ghost: Optional[Tuple[float, float]] = None,
             info: Optional[dict] = None):
        info = info or {}
        d = ImageDraw.Draw(pil, "RGBA")
        w, h = pil.size
        d.rectangle([0, 0, w - 1, h - 1], outline=(46, 56, 70, 255))
        d.rectangle([0, 0, w - 1, 19], fill=(12, 16, 22, 210))
        d.text((7, 4), label or "WRIST", font=self._font_m, fill=HUD_ACCENT)

        # State chip, right of the title bar. This is load-bearing: a solid
        # model that has stopped being updated looks exactly like a mechanism
        # that has stopped moving, so the reason it is not moving has to be on
        # screen next to it.
        x = w - 7
        state = info.get("state")
        if state:
            fill = {"warn": HUD_WARN, "bad": HUD_BAD}.get(info.get("level"), HUD_ACCENT)
            x -= d.textlength(state, font=self._font_m)
            d.text((x, 4), state, font=self._font_m, fill=fill)
            x -= 10
        if sub:
            d.text((x - d.textlength(sub, font=self._font_s), 5), sub,
                   font=self._font_s, fill=HUD_DIM)

        if ghost is None:
            th = BELT_N * (pitch + yaw)
            tb = BELT_N * (pitch - yaw)
            rows = ((("pitch", f"{pitch:+7.2f}°"), HUD_FG),
                    (("yaw", f"{yaw:+7.2f}°"), HUD_FG),
                    (("θA", f"{th:+7.1f}°"), HUD_DIM),
                    (("θB", f"{tb:+7.1f}°"), HUD_DIM))
            note = info.get("note") or ""
        else:
            # The three numbers this display exists for: what gravity says,
            # what the step counter says, and the gap. The gap is the
            # mechanical error the control loop cannot see.
            gap = pitch - ghost[0]
            rows = [(("meas", f"{pitch:+7.2f}°"), HUD_FG),
                    (("cmd", f"{ghost[0]:+7.2f}°"), _rgb8(GHOST_EDGE)),
                    ((" Δ", f"{gap:+7.2f}°"),
                     HUD_WARN if abs(gap) >= self.DIVERGE_WARN_DEG else HUD_FG),
                    (("yaw", f"{ghost[1]:+7.2f}°"), HUD_DIM)]
            # Measured tilt per commanded degree, once enough large-angle
            # samples exist to mean anything. 1.000 is the kinematics being
            # right; this is the one number on the rig that checks
            # DIFFERENTIAL_N against something external.
            k = info.get("scale")
            if k is not None:
                rows.append(((" k", f"{k:7.3f}"),
                             HUD_WARN if abs(k - 1.0) > 0.06 else HUD_DIM))
            note = info.get("note") or "pitch measured · yaw dead-reckoned"

        y = h - 4 - len(rows) * 14 - (13 if note else 0)
        box = max(116.0, d.textlength(note, font=self._font_s) + 16.0) if note else 116.0
        d.rectangle([0, y - 6, min(box, w - 1), h - 1], fill=(12, 16, 22, 190))
        for (k, v), fg in rows:
            d.text((8, y), k, font=self._font_s, fill=HUD_DIM)
            d.text((40, y), v, font=self._font_s, fill=fg)
            y += 14
        if note:
            d.text((8, y), note, font=self._font_s, fill=HUD_DIM)


# ===========================================================================
#   Tk widget
# ===========================================================================
class WristView(tk.Frame):
    """The renderer in a Tk frame: drag to orbit, wheel to zoom.

    Feed it with `set_pose()` from any thread.

    **Rendering happens on a worker thread**, not in the Tk callback. A frame
    costs 40-100 ms depending on quality, and `gui.py` drives the camera panes
    from the same event loop at 30 fps -- rendering inline would drop every
    other camera frame to draw a diagram. The worker publishes finished PIL
    images; the Tk tick does nothing but wrap the newest one in a PhotoImage,
    which must happen on the main thread.

    **Quality adapts.** While the pose is moving it renders at 1x, because the
    motion hides the aliasing and 1x is nearly three times cheaper. Once the
    turret has been still for `CRISP_AFTER_S` it re-renders the same pose at
    2x, so a parked machine is shown clean. Nothing re-renders at all when the
    pose has not changed and the image is already crisp.
    """

    CRISP_AFTER_S = 0.35

    def __init__(self, parent, *, size=(400, 280), fps: int = 24,
                 mesh_path: Path = MESH_PATH, label: str = "WRIST — live CAD",
                 bg: str = "#0a0d12", supersample: int = 2, **kw):
        super().__init__(parent, bg=bg, **kw)
        self.r = WristRenderer(size=size, mesh_path=mesh_path, supersample=1)
        self._label = label
        self._best_ss = max(1, int(supersample))

        self._lock = threading.Lock()
        # pitch, yaw, beam, ghost(pitch,yaw)|None, state, level, note
        self._pose = (0.0, 0.0, 0.0, None, "", "", "", None)
        self._cam = (self.r.az, self.r.el, self.r.dist)
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._out: Optional[Image.Image] = None
        self._out_seq = 0
        self._drawn_seq = -1
        self._t_render = 0.0

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._after: Optional[str] = None
        self._period = max(20, int(1000 / max(1, fps)))
        self._drag: Optional[Tuple[int, int]] = None

        self.canvas = tk.Canvas(self, width=size[0], height=size[1], bg=bg,
                                highlightthickness=0, bd=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self._item = None

        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom(0.9))
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1.1))
        self.bind("<Destroy>", self._stop)

        self._worker = threading.Thread(target=self._run, name="wristview",
                                        daemon=True)
        self._worker.start()
        self._tick()

    # -- feeding ---------------------------------------------------------
    def set_pose(self, pitch_deg: float, yaw_deg: float, beam: float = 0.0,
                 ghost: Optional[Tuple[float, float]] = None,
                 state: str = "", level: str = "", note: str = "",
                 scale: Optional[float] = None) -> None:
        """Thread safe. `beam` is 0..1; anything > 0 draws the laser.

        `ghost` is the dead-reckoned `(pitch, yaw)` to draw as a silhouette
        over the measured pose in `pitch_deg, yaw_deg`.

        **Pass `ghost=None` and the dead-reckoned pose as the solid one when
        the IMU is not answering.** Holding the last measured pose instead
        would draw a stationary solid model against a moving ghost, which is
        the signature of a stalled mechanism -- the single most alarming thing
        this display can show, fabricated out of a dropped serial reply.
        """
        with self._lock:
            new = (float(pitch_deg), float(yaw_deg), float(beam),
                   None if ghost is None else (float(ghost[0]), float(ghost[1])),
                   state, level, note, scale)
            if new == self._pose:
                return
            self._pose = new
        self._wake.set()

    # -- interaction -----------------------------------------------------
    def _press(self, e):
        self._drag = (e.x, e.y)

    def _motion(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        self._nudge(-dx * 0.45, dy * 0.4, 1.0)

    def _wheel(self, e):
        self._zoom(0.9 if e.delta > 0 else 1.1)

    def _zoom(self, f):
        self._nudge(0.0, 0.0, f)

    def _nudge(self, daz, dele, zoom):
        """Record the camera the operator wants; the worker applies it.

        The renderer object belongs to the worker thread. Locking it around a
        90 ms render would freeze the drag instead of smoothing it.
        """
        with self._lock:
            az, el, dist = self._cam
            self._cam = self.r.clamp_camera(az + daz, el + dele, dist * zoom)
        self._wake.set()

    def _stop(self, _e=None):
        self._stopping.set()
        self._wake.set()
        if self._after is not None:
            try:
                self.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    # -- worker ----------------------------------------------------------
    def _run(self):
        shown = None
        crisp = False
        while not self._stopping.is_set():
            with self._lock:
                pose, cam = self._pose, self._cam
            moved = (pose, cam) != shown
            if not moved and crisp:
                # Nothing to do until something changes.
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            if not moved:
                # Held still long enough: redraw the same pose properly.
                if self._wake.wait(self.CRISP_AFTER_S):
                    self._wake.clear()
                    continue
                self.r.set_supersample(self._best_ss)
                crisp = True
            else:
                self.r.set_supersample(1)
                crisp = self._best_ss == 1
            try:
                self.r.az, self.r.el, self.r.dist = cam
                t0 = time.perf_counter()
                sub = f"{self._t_render * 1000:.0f} ms" if self._t_render else ""
                pil = self.r.render(pose[0], pose[1], beam=pose[2],
                                    label=self._label, sub=sub, ghost=pose[3],
                                    hud={"state": pose[4], "level": pose[5],
                                         "note": pose[6], "scale": pose[7]})
                self._t_render = time.perf_counter() - t0
                with self._lock:
                    self._out = pil
                    self._out_seq += 1
                shown = (pose, cam)
            except Exception:
                import traceback
                traceback.print_exc()
                time.sleep(0.5)

    # -- Tk loop ---------------------------------------------------------
    def _tick(self):
        self._after = None
        try:
            with self._lock:
                pil, seq = self._out, self._out_seq
            if pil is not None and seq != self._drawn_seq:
                # PhotoImage must be built on the Tk thread; keep a reference
                # or the image is garbage collected and the canvas goes blank.
                self._photo = ImageTk.PhotoImage(pil)
                if self._item is None:
                    self._item = self.canvas.create_image(0, 0, anchor="nw",
                                                          image=self._photo)
                else:
                    self.canvas.itemconfigure(self._item, image=self._photo)
                self._drawn_seq = seq
        except Exception:
            import traceback
            traceback.print_exc()
        if self.winfo_exists() and not self._stopping.is_set():
            self._after = self.after(self._period, self._tick)


class PoseInset:
    """A `WristRenderer` on a worker thread, with no Tk in it at all.

    For compositing into a camera pane rather than living in a widget: the GUI
    feeds it poses and takes whatever image is finished, and a slow frame
    delays nothing. `WristView` is the same idea wrapped in a canvas; this one
    exists because the pose overlay is drawn INTO the wide pane, where there is
    no canvas of its own to blit to.

    Deliberately decoupled from the perception view -- its own renderer, its
    own scene, its own thread. It cannot touch the panes' layout and it cannot
    slow the video down; the worst it can do is show a slightly old pose.
    """

    def __init__(self, *, size=(300, 210), mesh_path: Path = MESH_PATH,
                 fps: float = 12.0, supersample: int = 2,
                 label: str = "POSE"):
        self.r = WristRenderer(size=size, mesh_path=mesh_path, supersample=1)
        # Frame the MOVING parts, not the whole machine. The default camera
        # fits base to faceplate, which at 260 px leaves the head small and the
        # ghost's offset hard to read -- and the base cannot diverge, so the
        # space it takes is spent on the one body guaranteed to agree. Aim
        # just above the wrist centre -- the CAD puts the pitch and yaw axes
        # crossing at (0, 28, 0), and the payload that swings about it sits
        # above that -- with enough margin that a head pitched toward the
        # camera still clears the top edge.
        self.r.target[1] = 32.0
        self.r.dist *= 0.92
        self._label = label
        self._best_ss = max(1, int(supersample))
        self._period = 1.0 / max(1.0, float(fps))
        self._lock = threading.Lock()
        self._pose = (0.0, 0.0, 0.0, None, "", "", "", None)
        self._out: Optional[Image.Image] = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._t_render = 0.0
        self._worker = threading.Thread(target=self._run, name="pose-inset",
                                        daemon=True)
        self._worker.start()

    # Same contract as WristView.set_pose, including the ghost=None rule.
    set_pose = WristView.set_pose

    def latest(self) -> Optional[Image.Image]:
        """The newest finished frame, or None. Never blocks on the renderer."""
        with self._lock:
            return self._out

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()

    def _run(self) -> None:
        shown = None
        crisp = False
        while not self._stopping.is_set():
            with self._lock:
                pose = self._pose
            moved = pose != shown
            if not moved and crisp:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            if not moved:
                # Held still: redraw the same pose properly. Same policy as
                # WristView -- 1x while it moves, 2x once it settles.
                if self._wake.wait(WristView.CRISP_AFTER_S):
                    self._wake.clear()
                    continue
                self.r.set_supersample(self._best_ss)
                crisp = True
            else:
                self.r.set_supersample(1)
                crisp = self._best_ss == 1
            try:
                t0 = time.perf_counter()
                pil = self.r.render(pose[0], pose[1], beam=pose[2],
                                    label=self._label, sub="", ghost=pose[3],
                                    hud={"state": pose[4], "level": pose[5],
                                         "note": pose[6], "scale": pose[7]})
                self._t_render = time.perf_counter() - t0
                with self._lock:
                    self._out = pil
                shown = pose
            except Exception:
                import traceback
                traceback.print_exc()
                self._stopping.wait(1.0)
            # Rate limit: the video pane runs at 30 fps and a human reading a
            # diagram does not need it. Nothing here is in the control path.
            self._stopping.wait(self._period)


def open_window(parent=None, *, size=(430, 300), title="WRIST — live CAD",
                topmost: bool = False) -> Tuple[tk.Toplevel, WristView]:
    """Pop the view in its own small window. Returns (toplevel, view)."""
    top = tk.Toplevel(parent) if parent is not None else tk.Tk()
    top.title(title)
    top.configure(bg="#0a0d12")
    top.resizable(False, False)
    if topmost:
        top.attributes("-topmost", True)
    view = WristView(top, size=size)
    view.pack(padx=6, pady=6)
    return top, view


if __name__ == "__main__":
    # Standalone demo: sweep the joints so the differential is visible with no
    # hardware attached. `python -m turret_host.wristview`
    root = tk.Tk()
    root.title("wrist view — demo sweep")
    root.configure(bg="#0a0d12")
    view = WristView(root, size=(520, 380))
    view.pack(padx=8, pady=8)
    t0 = time.time()

    def drive():
        t = time.time() - t0
        view.set_pose(18.0 * math.sin(t * 0.55), 42.0 * math.sin(t * 0.31),
                      beam=0.6 + 0.4 * math.sin(t * 3.0))
        root.after(33, drive)

    drive()
    root.mainloop()
