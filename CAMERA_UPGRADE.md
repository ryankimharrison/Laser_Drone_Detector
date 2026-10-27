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

Camera-dependent calibration writers now store `camera_context`. Loaders compare
it against the active profile before using Jacobians, goals, intrinsics, focal
measurements, control mappings, or fused-view registration. Unstamped legacy
files are accepted only on the exact original C270/OV2710 geometry profiles.
Do not add new metadata to an old fit to bypass this check. Move incompatible
files aside and measure again. Structured-light correspondence captures also
carry context so the registration fitter cannot stamp old captures as new.
The GUI disables an incompatible overlay; motion calibration loaders refuse it.
Lens or mount changes require updating their IDs even when image size is unchanged.

Flight tracking uses elapsed-time confirmation, loss, reseed and wide-fallback
thresholds referenced to the former 30 fps behavior. At least two hits and
0.133 s of uninterrupted evidence are required to confirm; loss from TRACK
takes 0.333 s, independent of delivery rate. These are starting values to
validate in flight, not performance claims. The replacement profile disables
the handheld 1.6 aspect-ratio association veto; invalid boxes are still rejected
and laser containment rules are unchanged. Range estimates now allow 1.5–12 m,
covering the previous 5–10 m work, but remain approximate silhouette-size
estimates using the configured focal length. They are not ground-truth depth.

## Dataset and training workflow

Keep each capture session wholly in one partition, including both cameras.
Choose independent sessions for train, validation and test; do not split two
parts of the same flight into differently named sessions. Use a new output
version instead of modifying the previous reviewed dataset:

```powershell
python -m turret_host.ingest_session training_flight --split train --output dataset/flight_v1 --no-prelabel
python -m turret_host.ingest_session validation_flight --split val --output dataset/flight_v1 --no-prelabel
python -m turret_host.ingest_session test_flight --split test --output dataset/flight_v1 --no-prelabel
```

Replace these session names with actual directories under `dataset/` containing
`clips/<clip>/narrow_*.jpg` and `wide_*.jpg`. Existing original images/labels are
not migrated or changed by these commands. Reuse older reviewed examples by
copying their image/label pairs into their assigned session partition in the
new version, preserving `session__clip__camera_frame` filenames. A former
temporally split session must be assigned wholly to one partition.

Label and review every partition, including missed detections. Empty labels
mean negatives, not unreviewed positives; the audit cannot establish human
review. Ingestion does not automatically prelabel test footage, and reruns
prelabel only newly added train/val images. The old automatic temporal-split
fallback now refuses and directs users to session ingestion.

After review:

```powershell
python -m turret_host.train_detector --data dataset/flight_v1/data.yaml --freeze-test
python -m turret_host.train_detector --data dataset/flight_v1/data.yaml --weights models/drone_y11n_v4_1280.pt --check
python -m turret_host.train_detector --data dataset/flight_v1/data.yaml --weights models/drone_y11n_v4_1280.pt --name flight_v1_candidate --seed 0
python -m turret_host.train_detector --data dataset/flight_v1/data.yaml --weights models/drone_y11n_v4_1280.pt --evaluate --split test --name flight_v1_baseline_test
```

Training audits session leakage, exact duplicate image content across partitions,
label presence/format, and test image/label hashes before loading a model. A
frozen test set cannot be extended or re-frozen after edits. Validation is used
for iterative selection; reserve test evaluation for final comparisons. Exact
hash checks cannot detect re-encoded near-duplicates; independent sessions are
still required. The YAML supports directory partitions ending in `images` with
sibling `labels` directories and one class, `drone`.

The training default input is 1280; tune `--batch` to measured VRAM capacity.
`--weights`, `--data`, `--device`, `--seed`, `--degrees`, `--scale`, and HSV
augmentation flags are explicit experiment controls. Retain color imagery and
augmentation appropriate for the unchanged wide camera. Every training or
evaluation run writes `experiment.json` containing arguments, dataset/checkpoint
hashes and library versions, and refuses an existing run name. No data has been
collected, labels changed, weights trained or models promoted by this code update.
