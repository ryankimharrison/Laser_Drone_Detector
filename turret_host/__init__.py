"""turret_host -- the tracking stack.

This file exists for one concrete reason beyond tidiness: `types.py` in this
directory SHADOWS THE STDLIB `types` MODULE. Running any file in here by path
puts this directory at `sys.path[0]`, and the next lazy stdlib import chain
(`re -> enum -> from types import ...`, `threading -> functools -> from types
import GenericAlias`) resolves to our `types.py` and dies with a circular-import
error that names our file and looks like our bug.

`python -m turret_host.<mod>` never has that problem. Every module that can be
run by path carries a sys.path repair that RUNS BEFORE ANY OTHER IMPORT and
*replaces* the script directory with the project root -- inserting the root
without removing the script dir is not enough, because sys.path[0] still wins.

Importing this package is free: nothing here touches hardware, loads a model or
opens a port.
"""

import os as _os
import sys as _sys
import warnings as _warnings

# ==========================================================================
#   MSMF HARDWARE TRANSFORMS -- THIS MUST HAPPEN BEFORE cv2 IS IMPORTED
# ==========================================================================
# OpenCV's Media Foundation backend tries to negotiate a hardware transform
# (the DXVA/MFT colour-conversion path) when it opens a camera. With the C270
# on this machine that negotiation stalls for over a minute before falling
# back to software. MEASURED, same process, same camera, only this variable
# changed:
#
#       open_camera()        default    HW_TRANSFORMS=0
#       narrow (C270)        73.20 s          0.29 s
#       wide   (OV2710)       4.85 s          0.15 s
#
# 250x on the narrow camera, and delivered frame rate is unchanged (30.8 vs
# 30.2 fps) -- the transform was buying nothing here. lock_exposure() pays it
# too, because it re-opens on MSMF to verify the value took, which is where
# its own 15.7 s went.
#
# It has to be set before the first `import cv2`: videoio reads the
# environment once, at module init, and setting it afterwards does nothing.
# This package is imported before any turret_host module that imports cv2, so
# here is the one place that is reliably early enough.
#
# setdefault, not assignment: an operator who exports it explicitly -- to test
# whether a future OpenCV has fixed this -- should win over us.
_os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

if "cv2" in _sys.modules:
    # Too late to help, and silently slow is how this cost a night once.
    _warnings.warn(
        "turret_host was imported AFTER cv2, so "
        "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS could not take effect. "
        "Expect camera opens to take ~60 s instead of ~0.3 s. Import "
        "turret_host (or set the variable yourself) before importing cv2.",
        RuntimeWarning, stacklevel=2)

__all__ = [
    "config", "types", "cameras", "detector", "tracker",
    "control", "link", "homing", "calibrate", "gui", "app",
]
