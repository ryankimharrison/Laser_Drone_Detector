# Replacement narrow camera and indoor flight

The C270 has been replaced. The default narrow profile targets an **uncommissioned
OV9281-class monochrome global-shutter USB module** (the sensor name OV9821 must
still be confirmed). No USB identity, exposure, measured lens, or mount identity
is invented. 1280x800 at 30 fps, MJPG, 90-degree clockwise mounting and 1400 px
focal length are provisional settings, not hardware measurements. Hardware camera
identification refuses to guess until its USB VID/PID is configured.

Set `TURRET_CAMERA_CONFIG` to an absolute local JSON path, for example
`$env:TURRET_CAMERA_CONFIG = "$PWD\camera.local.json"` in PowerShell. The root
object accepts `narrow` and `wide` objects overriding the dataclass fields in
`turret_host/camera_profile.py`. `camera.local.json` is ignored by Git.

Configure the actual USB module `vid_pid` as two decimal integers; optionally
add its `serial`. A serial-less module is accepted only when exactly one device
matches. Set `name`, `lens_id`, and `mount_id` to stable identities for this rig.
Set `size`, `fps`, `fourcc` (e.g. MJPG or YUY2), `rotation_deg`, `clockwise`,
`exposure`, and `focal_px` after measuring them. `max_width` is an optional
enumeration checksum, not the requested streaming width. Do not claim 120 fps
solely from the sensor specification. Confirm simultaneous streaming with the
wide camera and measure distinct-frame delivery and blur in the actual lighting.

`TURRET_NARROW_PROFILE=c270` explicitly selects the previous rig for comparison.
Both capture and dataset tools use the same profile. Decoded uint8 monochrome
images are expanded into three identical channels once at capture; 16-bit/raw
formats are rejected rather than silently rescaled. Null exposure leaves the
device setting unchanged and reports that exposure has not been commissioned.

The first flight milestone is **camera detection and following with the laser
disabled**. Use `--tracking-only` with the main application. This change does
not validate laser operation with a monochrome camera: the existing green-dot
calibration cannot recover chroma from identical channels.

Preserve the working 5–10 m indoor dataset and detector. Compare the existing
weights on replacement-camera handheld and flying footage before fine-tuning.
Hardware calibration and flight acceptance must be measured on the rig.
