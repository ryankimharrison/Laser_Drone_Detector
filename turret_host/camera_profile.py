"""Camera configuration without probing hardware at import time.

TURRET_CAMERA_CONFIG points to a local JSON object with narrow/wide overrides.
Device identifiers describe the USB module, not the sensor chip. Never guess
them from a sensor name. See CAMERA_UPGRADE.md for commissioning.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
import os


@dataclass(frozen=True)
class CameraProfile:
    name: str
    vid_pid: tuple | None
    serial: str | None
    size: tuple
    fps: int
    fourcc: str
    exposure: float | None
    rotation_deg: int
    clockwise: bool
    monochrome: bool
    shutter: str
    rolling_shutter_s: float
    focal_px: float
    lens_id: str
    mount_id: str
    max_width: int | None
    pipeline_delay_s: float = 0.0
    command_delay_s: float = 0.0

    def __post_init__(self):
        if (len(self.size) != 2 or any(type(v) is not int or v <= 0 for v in self.size)
                or type(self.fps) is not int or self.fps <= 0):
            raise ValueError("camera size and fps must be positive integers")
        if self.vid_pid is not None and (len(self.vid_pid) != 2 or any(
                type(v) is not int or not 0 <= v <= 65535 for v in self.vid_pid)):
            raise ValueError("vid_pid must contain two USB identifiers (integers)")
        if len(self.fourcc) != 4 or self.rotation_deg not in (0, 90, 180, 270):
            raise ValueError("fourcc must have four characters; rotation must be a quarter turn")
        if self.shutter not in ("global", "rolling"):
            raise ValueError("shutter must be global or rolling")
        for value in (self.focal_px, self.rolling_shutter_s, self.pipeline_delay_s,
                      self.command_delay_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("camera geometry and delays must be finite and nonnegative")
        if self.focal_px == 0 or (self.shutter == "global" and self.rolling_shutter_s != 0):
            raise ValueError("focal length must be positive; global shutter correction must be zero")
        if self.exposure is not None and not math.isfinite(self.exposure):
            raise ValueError("exposure must be finite or null")
        if any(not isinstance(v, str) or not v for v in (self.name, self.lens_id, self.mount_id)):
            raise ValueError("name, lens_id and mount_id must be nonempty strings")

    def geometry(self):
        """Only geometry/identity, so exposure or FPS tuning does not invalidate a fit."""
        keys = ("name", "vid_pid", "serial", "size", "rotation_deg", "clockwise",
                "focal_px", "lens_id", "mount_id")
        return json.loads(json.dumps({k: asdict(self)[k] for k in keys}))


C270 = CameraProfile("c270", (0x046D, 0x0825), "C8258920", (1280, 720),
                     30, "MJPG", -5, 90, True, False, "rolling", .010,
                     1400., "c270-ir-cut-removed", "legacy-narrow", 1280)
OV9281 = CameraProfile("ov9281-uncommissioned", None, None, (1280, 800),
                       30, "MJPG", None, 90, True, True, "global", 0.,
                       1400., "unmeasured", "unverified", None)
WIDE = CameraProfile("ov2710", (0x32E4, 0x9230), None, (1920, 1080),
                     30, "MJPG", -6, 0, True, False, "rolling", 0.,
                     700., "2.1mm-m12", "legacy-wide", 1920)


def load_profiles(path=None, narrow_name=None):
    name = narrow_name or os.environ.get("TURRET_NARROW_PROFILE", "ov9281")
    if name not in ("ov9281", "c270"):
        raise ValueError("TURRET_NARROW_PROFILE must be ov9281 or c270")
    profiles = {"narrow": OV9281 if name == "ov9281" else C270, "wide": WIDE}
    path = path or os.environ.get("TURRET_CAMERA_CONFIG")
    if path:
        with open(path, encoding="utf-8") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict) or set(overrides) - set(profiles):
            raise ValueError("camera configuration must contain only narrow/wide objects")
        for role, values in overrides.items():
            values = dict(values)
            for key in ("size", "vid_pid"):
                if values.get(key) is not None:
                    values[key] = tuple(values[key])
            profiles[role] = replace(profiles[role], **values)
    return profiles
