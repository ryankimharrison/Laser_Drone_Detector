"""Target detection (YOLO) and face detection (YuNet).

Two independent detectors, deliberately not one model:

* `Detector`   -- Ultralytics YOLO on the full native frame. Proposes targets.
                  Its output is a *proposal*; the association gate in tracker.py
                  decides what is real.
* `FaceDetector` -- OpenCV's YuNet. Its output drives the laser inhibit. It is a
                  SAFETY INTERLOCK, not a feature.

Why YuNet and not the `person` class of the YOLO model: the drone is *held in a
hand*. A person detector fires on the person holding it, every frame, and the
laser would never be allowed on. Faces are what must never be lased; faces are
what we detect. Do not "simplify" this by reusing the person class.

The narrow camera (C270) has had its IR-CUT FILTER REMOVED. Near-IR bleeds into
all three Bayer channels, so colour is no longer photometrically meaningful and
COCO weights -- trained entirely on IR-cut imagery -- will read low-confidence
on this camera. That is expected, not a bug to tune around. Consequences that
are enforced here:

  * NOTHING in this module gates on colour. There is no HSV test, no green-
    channel test, no white-balance normalisation. Shape and motion survive the
    domain shift; colour does not. Do not add a colour heuristic here later.
  * Confidence will be degraded, so `config.YOLO_CONF` is a *proposal*
    threshold, not an accept threshold.
  * The real fix is fine-tuning on frames from this camera with the IR LEDs in
    whatever state the demo will use. `Detector(weights=...)` and the
    TURRET_YOLO_WEIGHTS environment variable are the hook for that; nothing
    else in the stack needs to change when those weights arrive.

Nothing here touches hardware at import time: no model is loaded and no CUDA
context is created until `start()`. The module imports cleanly on a machine
with no camera, no board, and no GPU.

Neither class is thread-safe. One inference thread owns CUDA (see BUILD_SPEC);
give each thread its own instance if that ever changes.

Run the self-test with:

    python -m turret_host.detector            # synthetic frame, full pipeline
    python -m turret_host.detector --fetch-face-model
    python -m turret_host.detector --export-engine
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ in (None, ""):
    # MUST run before any other import, and may only use `os`/`sys` (both are
    # already loaded at interpreter start).
    #
    # Running this file by path puts turret_host/ at sys.path[0], where
    # `types.py` SHADOWS THE STDLIB `types` MODULE -- the first lazy stdlib
    # import of it then dies with a circular-import error that names nothing
    # relevant. Dropping the script directory fixes the shadow and putting the
    # project root there makes `import turret_host` resolve. `python -m
    # turret_host.detector` from the project root never reaches this.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or os.getcwd()) != _here]
    sys.path.insert(0, os.path.dirname(_here))

import argparse
import hashlib
import math
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from turret_host import config
from turret_host.types import Detection, DetectionResult, Frame

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent

#: Override the directory searched for model files (weights, ONNX).
MODEL_DIR_ENV = "TURRET_MODEL_DIR"
#: Override `config.YOLO_WEIGHTS` without editing config -- this is the hook for
#: dropping in fine-tuned weights once they exist.
WEIGHTS_ENV = "TURRET_YOLO_WEIGHTS"

#: Default download target when a model has to be fetched.
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"

#: Assets shipped with the package. Currently one file, and it is a SAFETY
#: asset, not a test fixture: FaceDetector.orientation_self_check() uses it to
#: prove the face interlock can actually see a face through this build's
#: rotation constant, and app.py refuses to ARM the laser until it has.
ASSET_DIR = _PKG_DIR / "assets"
ORIENTATION_FACE_ASSET = "orientation_check_face.jpg"

# YuNet lives in OpenCV's model zoo. Pinned by content hash: this file drives a
# safety interlock, so a silently different model is a safety change. If the
# hash check ever fires, look at what you downloaded before disabling it.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# YuNet cost per call, MEASURED on this machine 2026-09-16 (CPU, OpenCV 5.0,
# warm, median of 15):
#   1920x1080 -> 54 ms    1280x720 -> 23 ms    960x540 -> 14 ms   640x360 -> 7 ms
#
# So at the wide camera's native 1920x1080 a face pass costs MORE than the
# 33 ms frame period. That is a scheduling problem for the inference thread,
# not a reason to downscale by default: the wide lens puts a face at 5 m at
# only ~22 px across (0.16 m * WIDE_F_PX / 5 m), and 1280 wide takes that to
# ~15 px, which is at YuNet's floor. A face that is never detected is never
# inhibited, so native resolution is the default and the frame rate is what
# gives. OpenCV releases the GIL inside the forward pass, so this CPU work
# overlaps GPU inference if the caller runs it on its own thread.
#
# `max_side` exists for the caller that has measured its own budget and made
# that trade deliberately.
#
# AMENDED 2026-09-16 (a): the NARROW detector -- the only one that gates the
# beam -- also rotates its input (see FaceDetector). A 640 downscale was added
# by default to pay for that rotation.
#
# AMENDED 2026-09-16 (b): THAT DOWNSCALE IS REVERTED, and the reasoning that
# justified it was measured to be wrong. It claimed "a face out at 3 m is still
# ~37 px, far above YuNet's ~15 px floor". Measured on this machine with this
# pinned model, on raw-buffer-orientation 1280x720 narrow frames, at
# FACE_CONF = 0.60:
#
#     raw face px   ~range    max_side=640      native
#           320       0.70 m        0.876       0.870
#           200       1.12 m        0.815       0.885
#           160       1.40 m        0.765       0.896
#           128       1.75 m        0.000       0.877
#            96       2.33 m        0.000       0.747
#            80       2.80 m        0.000       0.765
#            64       3.50 m        0.000       0.000
#
# The real floor is ~90 px IN THE IMAGE FED TO YUNET, six times the claimed 15,
# so a 37 px face scores 0.000 even at native resolution. The downscale did not
# cost a little margin at long range; it halved the interlock's range outright:
# faces stopped being seen past ~1.40 m instead of ~2.80 m.
#
# That is a person standing 1.5 m behind the one holding the drone being
# invisible to the only detector that keeps light off a face. The timing
# argument does not pay for it either: measured 19.4 ms native vs 5.6 ms
# downscaled, on a DEDICATED THREAD, against a 33 ms frame period and a 200 ms
# staleness budget. Native fits with room to spare.
#
# So native is the default for both detectors, which is what the cost note
# above argued for in the first place. `max_side` remains for a caller that has
# measured its own budget -- but note that no caller on the beam-gating path
# should be using it without repeating the table above.
_FACE_MAX_SIDE = None

# The rotations the box mapping below implements, clockwise degrees.
_ROTATIONS = (0, 90, 180, 270)

_ROTATE_FLAG = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

FACE_LABEL = "face"


def _unrotate_box(x, y, bw, bh, rot: int, src_w: int, src_h: int):
    """Map one box from a clockwise-rotated frame back to source coordinates.

    `src_w`/`src_h` are the size of the frame BEFORE rotation; the detection
    was made on the rotated one.

    EDGE CONVENTION, and it is the whole subtlety here. A box is (x, y, w, h)
    with EXCLUSIVE far edges: it covers source pixels x .. x+w-1. So the
    transform to use is the one on continuous EDGE coordinates, not the one on
    pixel indices. For a pixel index the 90 CW map is

        (px, py) -> (src_h - 1 - py, px)

    and it is tempting to invert that directly -- but applying it to an
    exclusive edge x2 is an off-by-one, because the edge at x2 is not a pixel.
    On edges the same rotation is

        rot= 90 CW : (ex, ey) -> (src_h - ey, ex)          size (W,H) -> (H,W)
        rot=180    : (ex, ey) -> (src_w - ex, src_h - ey)
        rot=270 CW : (ex, ey) -> (ey, src_w - ex)          size (W,H) -> (H,W)

    with no -1 anywhere. Inverted below.

    This was wrong (a `- 1` on each flipped axis) and shifted every rotated box
    by exactly 1 px pre-scale, which is 2 px in raw narrow coordinates at a 2x
    downscale. It is a pure translation, so for roughly half of face/beam
    geometries it INFLATED the reported clearance -- in the one number that
    keeps light off a face. Small (1.7% of the 120 px margin) but free to fix,
    and the docstring below claimed exactness it did not have.

    A rotation maps an axis-aligned box to an axis-aligned box, so this is now
    genuinely exact -- no bounding-box inflation, which matters because the face
    box feeds a distance-to-beam margin.
    """
    x2, y2 = x + bw, y + bh
    if rot == 0:
        return x, y, bw, bh
    if rot == 90:
        # source edge x = ry, source edge y = src_h - rx2
        return y, src_h - x2, bh, bw
    if rot == 180:
        return src_w - x2, src_h - y2, bw, bh
    if rot == 270:
        # source edge x = src_w - ry2, source edge y = rx
        return src_w - y2, x, bh, bw
    raise ValueError("unsupported rotation %r" % (rot,))


# ==========================================================================
#   Model file resolution
# ==========================================================================
def model_search_dirs() -> List[Path]:
    """Directories searched for a bare model filename, in priority order."""
    dirs: List[Path] = []
    env = os.environ.get(MODEL_DIR_ENV)
    if env:
        dirs.append(Path(env).expanduser())
    dirs += [DEFAULT_MODEL_DIR, _PROJECT_ROOT, _PKG_DIR]
    return dirs


def find_model(name: str) -> Optional[Path]:
    """Resolve a model name to an existing file, or None.

    A name containing a separator (or an absolute path) is taken literally --
    if it does not exist we do not go looking elsewhere for something with the
    same basename, because loading a *different* model than the one asked for
    is exactly the kind of silent substitution this stack must not do.
    """
    p = Path(name).expanduser()
    if p.is_absolute() or len(p.parts) > 1:
        return p if p.is_file() else None
    for d in model_search_dirs():
        candidate = d / name
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_face_model(dest_dir: Optional[Path] = None,
                     verify_hash: bool = True) -> Path:
    """Download the YuNet ONNX into `dest_dir` and return its path.

    Downloads to a `.part` file and renames only after the hash checks out, so
    an interrupted download can never leave a truncated model that loads and
    then under-detects.
    """
    dest_dir = Path(dest_dir) if dest_dir else DEFAULT_MODEL_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / config.FACE_MODEL
    part = dest.with_suffix(dest.suffix + ".part")

    print(f"[detector] downloading YuNet -> {dest}")
    req = urllib.request.Request(YUNET_URL, headers={"User-Agent": "turret_host"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        out.write(resp.read())

    got = _sha256(part)
    if verify_hash and got != YUNET_SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"YuNet download from {YUNET_URL} has sha256 {got}, expected "
            f"{YUNET_SHA256}. Refusing to install a face model that is not the "
            f"one this interlock was validated against."
        )
    part.replace(dest)
    print(f"[detector] YuNet installed: {dest} ({dest.stat().st_size} bytes)")
    return dest


# ==========================================================================
#   Target detector -- Ultralytics YOLO
# ==========================================================================
class Detector:
    """YOLO target detector for one camera's frames.

    `start()` loads the model and builds the CUDA context; nothing before that
    touches the GPU.
    """

    def __init__(self,
                 weights: Optional[str] = None,
                 imgsz: int = config.YOLO_IMGSZ,
                 conf: float = config.YOLO_CONF,
                 classes: Sequence[str] = config.TARGET_CLASSES,
                 half: bool = config.YOLO_HALF,
                 device: str = "cuda:0",
                 use_engine: bool = False,
                 export_if_missing: bool = False,
                 warmup_sizes: Sequence[tuple] = (config.NARROW_SIZE, config.WIDE_SIZE)):
        # weights precedence: explicit argument > environment > config. The
        # environment hook is how fine-tuned weights get swapped in without an
        # edit anywhere in the stack.
        self.weights_name = weights or os.environ.get(WEIGHTS_ENV) or config.YOLO_WEIGHTS
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.class_names = tuple(classes)
        self.half = bool(half)
        self.device = device
        self.use_engine = bool(use_engine)
        self.export_if_missing = bool(export_if_missing)
        self.warmup_sizes = tuple(tuple(s) for s in warmup_sizes)

        self._model = None
        self.weights_path: Optional[Path] = None
        self.model_names: dict = {}      # class index -> name, filled by start()
        self.class_ids: List[int] = []
        self.missing_classes: List[str] = []
        self.last_infer_ms: float = 0.0
        self.last_speed: dict = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "Detector":
        from ultralytics import YOLO          # heavy (pulls torch); load lazily
        import torch

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Detector was asked for CUDA but torch.cuda.is_available() "
                    "is False. Fix the torch/driver install, or construct with "
                    "device='cpu' for an explicitly-slow bench run. Not falling "
                    "back silently: a CPU fallback would miss the frame deadline "
                    "and look like a tracking bug."
                )
        elif self.half:
            raise RuntimeError(
                "half precision requires CUDA; pass half=False for a CPU run."
            )

        path = self._resolve_weights()
        if self.use_engine:
            path = self._resolve_engine(path)
        self.weights_path = path

        self._model = YOLO(str(path))
        self._resolve_classes()
        self._warmup()
        return self

    def close(self) -> None:
        self._model = None
        if self.device.startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- internals ---------------------------------------------------------
    def _resolve_weights(self) -> Path:
        found = find_model(self.weights_name)
        if found is not None:
            return found
        name = Path(self.weights_name).name
        if Path(self.weights_name).parent != Path("."):
            raise FileNotFoundError(
                f"YOLO weights not found: {self.weights_name}"
            )
        # A bare stock name (yolo11n.pt, yolov8n.pt ...) is fetched by
        # ultralytics itself on construction. A fine-tuned name is not, so say
        # so plainly rather than letting ultralytics 404 with its own message.
        if not (name.startswith(("yolo", "rtdetr", "sam")) and name.endswith(".pt")):
            raise FileNotFoundError(
                f"Weights '{self.weights_name}' are not a stock Ultralytics "
                f"asset and were not found in any of: "
                f"{[str(d) for d in model_search_dirs()]}. Put the file in "
                f"{DEFAULT_MODEL_DIR}, or set {MODEL_DIR_ENV} / "
                f"{WEIGHTS_ENV} to point at it."
            )
        return Path(self.weights_name)

    def _resolve_engine(self, pt_path: Path) -> Path:
        """Return the cached TensorRT engine beside the .pt, building if asked.

        THE ENGINE IS NOT PORTABLE. It is built for this exact GPU
        (architecture *and* SM count) and this exact TensorRT/driver version.
        Copying an .engine to another machine, or upgrading the driver, gives a
        load failure or -- worse -- silently wrong output. Delete and rebuild;
        never ship one.
        """
        engine = pt_path.with_suffix(".engine")
        if engine.is_file():
            return engine
        if not self.export_if_missing:
            raise FileNotFoundError(
                f"use_engine=True but {engine} does not exist. Build it once "
                f"with `python -m turret_host.detector --export-engine` (takes "
                f"minutes), or construct with export_if_missing=True. The "
                f"engine is specific to this GPU and TensorRT version."
            )
        return export_engine(str(pt_path), imgsz=self.imgsz, half=self.half)

    def _resolve_classes(self) -> None:
        """Map `config.TARGET_CLASSES` onto this model's class indices.

        Passing indices to predict() lets NMS drop everything else before it
        reaches Python. COCO has no `drone` class -- that one only appears once
        fine-tuned weights are loaded, so a missing name is reported, not fatal.
        Losing *every* target class is fatal: it would mean detecting nothing,
        forever, quietly.
        """
        self.model_names = {int(k): str(v) for k, v in self._model.names.items()}
        names = {v.lower(): k for k, v in self.model_names.items()}
        self.class_ids, self.missing_classes = [], []
        for want in self.class_names:
            idx = names.get(str(want).lower())
            if idx is None:
                self.missing_classes.append(want)
            else:
                self.class_ids.append(idx)
        if not self.class_ids:
            raise RuntimeError(
                f"None of TARGET_CLASSES {self.class_names} exist in "
                f"{self.weights_path}. Model knows: {sorted(names)[:20]}..."
            )
        if self.missing_classes:
            print(
                f"[detector] {self.weights_path.name if self.weights_path else '?'}"
                f" has no class(es) {self.missing_classes} -- detecting"
                f" {[self.model_names[i] for i in self.class_ids]} only."
                f" Expected on stock COCO weights; a fine-tune adds 'drone'.",
                file=sys.stderr,
            )

    def _predict_kwargs(self) -> dict:
        kw = dict(imgsz=self.imgsz, conf=self.conf, device=self.device,
                  classes=self.class_ids, verbose=False)
        # ultralytics 8.4 deprecated `half=` in favour of `quantize=16`; older
        # releases only understand `half=`. Pick by what the installed version
        # actually declares rather than by version string.
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            kw["quantize"] = 16 if self.half else 32
        else:
            kw["half"] = self.half
        return kw

    def _warmup(self) -> None:
        """Run inference once per expected frame shape.

        The first call on a new input shape allocates workspace and (for
        TensorRT) may re-profile -- 1-2 s of stall. Paying it here keeps it out
        of the control loop, where a one-off stall reads as a lost track.
        """
        kw = self._predict_kwargs()
        for (w, h) in self.warmup_sizes:
            blank = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            self._model.predict(blank, **kw)

    # -- inference ---------------------------------------------------------
    def detect(self, frame: Frame) -> DetectionResult:
        """Run the target detector on one frame.

        The frame is passed through unrotated and uncropped. The narrow camera
        is mounted 90 deg over but rotating the buffer costs a full-frame copy
        every frame and buys nothing -- the rotation is carried in the geometry
        (see config.NARROW_ROTATION_DEG), not in the pixels.
        """
        if self._model is None:
            raise RuntimeError("Detector.start() has not been called.")

        t0 = time.perf_counter()
        result = self._model.predict(frame.image, **self._predict_kwargs())[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        self.last_infer_ms = infer_ms
        self.last_speed = dict(result.speed) if result.speed else {}

        targets: List[Detection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            # One device->host transfer for the whole batch; per-box .item()
            # calls sync the CUDA stream once each and cost more than inference.
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clsids = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clsids):
                targets.append(Detection(
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    conf=float(c), label=str(result.names[int(k)]),
                ))
            # Strongest proposal first. The gate still decides, but when it has
            # no prior (SEARCH) it takes the top of this list.
            targets.sort(key=lambda d: d.conf, reverse=True)

        return DetectionResult(
            frame_t=frame.t,
            frame_index=frame.index,
            camera=frame.camera,
            targets=targets,
            faces=[],
            infer_ms=infer_ms,
        )


# ==========================================================================
#   TensorRT export helper
# ==========================================================================
def export_engine(weights: Optional[str] = None,
                  imgsz: int = config.YOLO_IMGSZ,
                  half: bool = config.YOLO_HALF,
                  force: bool = False) -> Path:
    """Export (or reuse) a TensorRT engine beside the .pt weights.

    Worth roughly 2-3x over the PyTorch path. Costs minutes to build, so the
    result is cached next to the weights and reused.

    THE ENGINE IS GPU-SPECIFIC AND TENSORRT-VERSION-SPECIFIC. It is not a build
    artifact you can commit or copy to the demo machine -- rebuild it there.
    Delete the .engine after any driver, TensorRT, or GPU change.
    """
    from ultralytics import YOLO

    name = weights or os.environ.get(WEIGHTS_ENV) or config.YOLO_WEIGHTS
    pt = find_model(name) or Path(name)
    engine = pt.with_suffix(".engine")
    if engine.is_file() and not force:
        print(f"[detector] engine already built: {engine}")
        return engine

    print(f"[detector] exporting TensorRT engine from {pt} (minutes, not seconds)")
    model = YOLO(str(pt))
    out = model.export(format="engine", imgsz=int(imgsz), half=bool(half))
    out = Path(out)
    print(f"[detector] engine written: {out}  (valid on THIS GPU only)")
    return out


# ==========================================================================
#   Face detector -- YuNet. SAFETY INTERLOCK.
# ==========================================================================
class FaceDetector:
    """YuNet face detector whose output inhibits the laser.

    Treat every failure here as fatal. A face detector that quietly returns an
    empty list is indistinguishable, to control.py, from a room with no faces
    in it -- and that is the one failure mode that ends with a laser on a face.
    So: no try/except around detection, no "skip if the model is missing", no
    degraded mode.

    THIS DETECTOR ROTATES ITS INPUT. THE DRONE DETECTOR DOES NOT. WHY:
    BUILD_SPEC's rule -- "never rotate frames for processing, carry the
    rotation in the geometry" -- is right for `Detector` above and WRONG here,
    and the difference is a property of the models, not of the frames.

      * YOLO is trained with rotation and flip augmentation on objects that
        have no canonical up, and a drone at 90 degrees is still a drone to it.
        Rotating for YOLO buys a full-frame copy per frame and nothing else.

      * YuNet is a SINGLE-SHOT UPRIGHT FACE DETECTOR. Its anchors and its
        training set are upright faces. On a frame rotated 90 degrees -- which
        is exactly what the C270 delivers, see config.NARROW_ROTATION_DEG -- a
        real face in the room is a sideways face in the buffer and YuNet's
        recall against it collapses. Fed the raw buffer this detector is
        STRUCTURALLY BLIND, and a blind face detector returns an empty list,
        which control.py cannot distinguish from a room with no faces in it.
        That is the whole safety story of the machine doing nothing.

    So the frame handed to YuNet is rotated to upright and the boxes are mapped
    BACK into raw-frame coordinates before they are returned, which keeps every
    consumer (control.nearest_face_px, the GUI overlay) unchanged. The rotation
    runs at native resolution -- see the _FACE_MAX_SIDE note, where the earlier
    "run it on a downscaled copy" trade is retracted with measurements.

    AND THE ROTATION CONSTANT IS ITSELF A SAFETY PARAMETER. Rotating fixes the
    blindness only if NARROW_ROTATION_DEG / NARROW_ROTATE_CLOCKWISE have the
    right SENSE. Get the sign backwards and the face arrives upside-down,
    YuNet returns [] for every frame, and control.py reads that as an empty
    room -- the same structural blindness, silently, with the beam permitted.
    Measured: mounted as config says, 1 face at conf 0.88; mounted the other
    way, 0 faces.

    That constant rests on a builder's eyeball observation of a live preview,
    so orientation_self_check() below exists to make the machine prove it
    rather than assert it, and app.py refuses to ARM until it has passed.
    """

    def __init__(self,
                 model: Optional[str] = None,
                 conf: float = config.FACE_CONF,
                 nms: float = 0.3,
                 top_k: int = 5000,
                 max_side: Optional[int] = _FACE_MAX_SIDE,
                 rotation_deg: Optional[int] = None,
                 auto_download: bool = True):
        self.model_name = model or config.FACE_MODEL
        # Lowering conf is the SAFE direction: more spurious inhibits, never
        # fewer real ones. Raising it trades safety for demo uptime.
        self.conf = float(conf)
        self.nms = float(nms)
        self.top_k = int(top_k)
        # Longest side fed to YuNet. Defaults to _FACE_MAX_SIDE rather than
        # native resolution because this detector now also rotates, and
        # rotating a half-size copy is a quarter of the memory traffic. Pass
        # None for native resolution.
        self.max_side = int(max_side) if max_side else None
        # Clockwise degrees needed to make the RAW buffer upright. Taken from
        # config so there is exactly one place the mounting is described; the
        # GUI's preview does the same arithmetic (gui.py: _rot).
        if rotation_deg is None:
            deg = int(config.NARROW_ROTATION_DEG) % 360
            rotation_deg = deg if config.NARROW_ROTATE_CLOCKWISE else (-deg) % 360
        self.rotation_deg = int(rotation_deg) % 360
        if self.rotation_deg not in _ROTATIONS:
            raise ValueError(
                "face-detector rotation must be one of %s, got %r. This is the "
                "laser's face interlock: an unsupported rotation is refused "
                "rather than silently ignored."
                % (sorted(_ROTATIONS), self.rotation_deg))
        self.auto_download = bool(auto_download)

        self._yn = None
        self._input_size = None          # (w, h) currently configured on YuNet
        # One YuNet instance PER INPUT SIZE, instead of re-pointing a single
        # instance with setInputSize(). See _detector_for() for why.
        self._yn_by_size: dict = {}
        # SERIALISES EVERY INFERENCE. This class is not thread-safe and has
        # two callers: app._faces_loop at 30 Hz, and the pre-arm orientation
        # check that ARM runs. They overlap -- pressing ARM does not stop the
        # faces thread -- so they were driving one cv2 net concurrently, which
        # surfaced as
        #     (-215:Assertion failed) buf.shape() == m.shape()  in forwardGraph
        # and killed the ARM check. Worse, orientation_evidence() MUTATES
        # rotation_deg, so the faces thread could read a rotation meant for the
        # other test -- a silent wrong answer from the laser's interlock.
        self._lock = threading.Lock()
        self.model_path: Optional[Path] = None
        self.last_infer_ms: float = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "FaceDetector":
        self.model_path = self._resolve_model()
        # Built here so a missing or corrupt ONNX fails at start() rather than
        # on the first frame. Seeded into the per-size cache so the load is not
        # repeated if a 320x320 pass ever arrives.
        self._yn = cv2.FaceDetectorYN.create(
            str(self.model_path), "", (320, 320),
            self.conf, self.nms, self.top_k,
        )
        self._input_size = (320, 320)
        self._yn_by_size = {(320, 320): self._yn}
        return self

    def close(self) -> None:
        self._yn = None
        self._yn_by_size = {}

    def _resolve_model(self) -> Path:
        found = find_model(self.model_name)
        if found is not None:
            return found
        if self.auto_download:
            return fetch_face_model()
        raise FileNotFoundError(
            f"Face model '{self.model_name}' not found in "
            f"{[str(d) for d in model_search_dirs()]}.\n"
            f"This is the laser's face interlock -- the system must not run "
            f"without it. Fix with either:\n"
            f"  python -m turret_host.detector --fetch-face-model\n"
            f"  curl -L -o \"{DEFAULT_MODEL_DIR / config.FACE_MODEL}\" {YUNET_URL}\n"
            f"or set {MODEL_DIR_ENV} to the directory holding it."
        )

    # -- inference ---------------------------------------------------------
    def _detector_for(self, size):
        """A YuNet instance built for exactly `size`, cached.

        Was `self._yn.setInputSize(size)` on one shared instance. Under OpenCV
        5.0's new DNN graph engine that raises, mid-inference, on the FIRST
        frame after a size change:

            (-215:Assertion failed) buf.shape() == m.shape()
            in 'cv::dnn::Net::Impl::forwardGraph'

        The graph is compiled for the shape it was created with and
        setInputSize() does not recompile it, so the allocated buffers and the
        new input disagree. This stack alternates sizes constantly -- the
        narrow pass is rotated to 720x1280 and the wide pass is 1920x1080 --
        so it threw almost immediately.

        THIS IS THE LASER'S FACE INTERLOCK, and the failure mode is the worst
        one available: the exception killed the faces thread, after which the
        face pass reported nothing at all. "No faces" and "the detector is
        dead" are the same empty list. The fail-closed design caught it --
        heads_valid went stale, ARM was refused -- but the verifier still
        reported `0/58 frames detected at the configured 90 deg` and concluded
        the ROTATION was backwards, which was a crashed detector being read as
        a geometry error. Do not reintroduce setInputSize() here.

        Caching per size rather than recreating per call: construction reloads
        the ONNX, and the sizes seen are a fixed small set (one per camera per
        rotation), so this settles after the first few frames.
        """
        yn = self._yn_by_size.get(size)
        if yn is None:
            yn = cv2.FaceDetectorYN.create(
                str(self.model_path), "", size,
                self.conf, self.nms, self.top_k,
            )
            self._yn_by_size[size] = yn
        self._input_size = size
        return yn

    def detect(self, frame: Frame) -> List[Detection]:
        """Return every face in the frame, in RAW full-frame pixel coordinates.

        Internally the frame is downscaled and then rotated to upright for
        YuNet (see the class docstring for why this detector rotates when the
        drone detector does not), and the boxes are mapped back. Callers --
        control.nearest_face_px, the GUI overlay -- see raw-buffer coordinates
        exactly as they did before, which is the point.
        """
        if self._yn is None:
            raise RuntimeError("FaceDetector.start() has not been called.")
        with self._lock:
            return self._detect_locked(frame)

    def _detect_locked(self, frame: Frame) -> List[Detection]:
        img = frame.image
        h, w = img.shape[:2]

        scale = 1.0
        if self.max_side and max(w, h) > self.max_side:
            scale = self.max_side / float(max(w, h))
            # INTER_AREA: the only downscale filter that does not alias small
            # faces out of existence, which is the failure that matters here.
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)

        # Size AFTER the downscale and BEFORE the rotation -- what _unrotate_box
        # needs to invert the rotation.
        small_h, small_w = img.shape[:2]
        rot = self.rotation_deg
        if rot:
            # cv2.rotate is a transpose plus a flip on a contiguous buffer: on
            # the downscaled 640x360 narrow frame it is well under a
            # millisecond, against the ~7 ms forward pass.
            img = cv2.rotate(img, _ROTATE_FLAG[rot])

        size = (img.shape[1], img.shape[0])
        yn = self._detector_for(size)

        t0 = time.perf_counter()
        _, raw = yn.detect(img)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        faces: List[Detection] = []
        if raw is None:                  # YuNet returns None, not an empty array
            return faces

        inv = 1.0 / scale
        for row in raw:
            # row = [x, y, w, h, 5 landmark xy pairs..., score]
            rx, ry, rbw, rbh = (float(v) for v in row[:4])
            # Back out of the rotation FIRST, at the downscaled scale the
            # rotation happened at, then undo the downscale. Doing it the other
            # way round would need the raw dimensions and rounds twice.
            sx, sy, sbw, sbh = _unrotate_box(rx, ry, rbw, rbh, rot,
                                             small_w, small_h)
            x, y, bw, bh = sx * inv, sy * inv, sbw * inv, sbh * inv
            faces.append(Detection(
                x1=x, y1=y, x2=x + bw, y2=y + bh,
                conf=float(row[-1]), label=FACE_LABEL,
            ))
        return faces

    def detect_into(self, result: DetectionResult, frame: Frame) -> DetectionResult:
        """Fill `result.faces` for a frame already run through `Detector`.

        `infer_ms` accumulates, so it stays what the control loop cares about:
        the total time this frame spent in inference.
        """
        result.faces = self.detect(frame)
        result.infer_ms += self.last_infer_ms
        return result

    # ------------------------------------------------------------------
    #   Orientation self-check -- run before the beam is ever permitted
    # ------------------------------------------------------------------
    def pipeline_self_check(self, frame_size=None, face_px=320):
        """Prove the face pipeline WORKS. Does NOT prove the rotation sign.

        Returns (ok, detail). Builds a synthetic raw buffer by applying the
        inverse of this detector's own configured rotation to an upright scene
        containing a real face photo, runs detect(), and requires a detection
        at or above config.FACE_CONF whose box maps back to where the face was
        painted.

        WHAT THIS PROVES: the model file is present and loads, YuNet reaches
        FACE_CONF on a real face, the rotate/downscale/unrotate chain is
        self-consistent, and _unrotate_box puts the box back where it belongs.
        Those are real failure modes and this catches all of them.

        WHAT THIS CANNOT PROVE, AND THE DISTINCTION IS THE WHOLE POINT: it
        cannot validate the SENSE of NARROW_ROTATION_DEG. It constructs its
        test buffer using the very constant it would be checking, so it is
        self-consistent for every rotation -- verified: all four of 0/90/180/270
        pass this check. Anything claiming otherwise is testing the code
        against its own assumptions, which is how the last round's rotation
        check passed while being off by a pixel.

        The rotation SIGN can only be settled by a real face in front of the
        real camera. That is orientation_evidence() below, and it is item 1 on
        the pre-arm checklist for exactly this reason.
        """
        asset = ASSET_DIR / ORIENTATION_FACE_ASSET
        if not asset.is_file():
            return False, ("face asset missing: %s -- cannot prove the face "
                           "interlock can see a face" % asset)
        face = cv2.imread(str(asset))
        if face is None:
            return False, "face asset unreadable: %s" % asset

        w, h = frame_size or config.NARROW_SIZE
        rot = self.rotation_deg

        def build(rotation):
            """An upright scene, turned into the raw buffer a camera mounted
            `rotation` degrees over would deliver."""
            # The upright scene has the raw buffer's dimensions swapped when
            # the rotation is a quarter turn.
            uw, uh = (h, w) if rotation in (90, 270) else (w, h)
            scene = np.full((uh, uw, 3), 70, np.uint8)
            fh0, fw0 = face.shape[:2]
            s = float(face_px) / fw0
            f = cv2.resize(face, (max(1, int(fw0 * s)), max(1, int(fh0 * s))))
            fh, fw = f.shape[:2]
            if fh > uh or fw > uw:
                return None, None
            y0, x0 = (uh - fh) // 2, (uw - fw) // 2
            scene[y0:y0 + fh, x0:x0 + fw] = f
            true_box = (x0, y0, fw, fh)
            if rotation == 0:
                return scene, true_box
            # Inverse of a clockwise rotation by `rotation` is a clockwise
            # rotation by 360 - rotation.
            return cv2.rotate(scene, _ROTATE_FLAG[(360 - rotation) % 360]), true_box

        buf, true_box = build(rot)
        if buf is None:
            return False, "frame too small for the %d px test face" % face_px
        faces = self.detect(Frame(image=buf, t=0.0, index=0, camera="narrow"))
        best = max((f.conf for f in faces), default=0.0)

        # Negative control: the mount the other way round.
        opposite = (360 - rot) % 360 if rot in (90, 270) else (180 if rot == 0 else 0)
        obuf, _ = build(opposite)
        obest = 0.0
        if obuf is not None and opposite != rot:
            obest = max((f.conf
                         for f in self.detect(Frame(image=obuf, t=0.0, index=0,
                                                    camera="narrow"))),
                        default=0.0)

        if best < config.FACE_CONF:
            return False, (
                "NO FACE DETECTED through rotation_deg=%d (best conf %.3f < "
                "FACE_CONF %.2f). The face interlock is BLIND with this "
                "orientation. Check config.NARROW_ROTATION_DEG / "
                "NARROW_ROTATE_CLOCKWISE against the actual camera mount "
                "(the opposite sense scored %.3f)."
                % (rot, best, config.FACE_CONF, obest))

        # The box must land near the face we painted, or the mapping is wrong
        # even though the detection worked.
        tx, ty, tw, th = true_box
        # true_box is in UPRIGHT coordinates; map its centre into raw-buffer
        # coordinates the same way the camera would.
        ucx, ucy = tx + tw / 2.0, ty + th / 2.0
        uw, uh = (h, w) if rot in (90, 270) else (w, h)
        if rot == 0:
            ecx, ecy = ucx, ucy
        elif rot == 90:      # raw = upright rotated 90 CCW
            ecx, ecy = ucy, uw - ucx
        elif rot == 180:
            ecx, ecy = uw - ucx, uh - ucy
        else:                # rot == 270; raw = upright rotated 90 CW
            ecx, ecy = uh - ucy, ucx
        got = max(faces, key=lambda f: f.conf)
        err = math.hypot(got.cx - ecx, got.cy - ecy)
        tol = 0.5 * max(tw, th)
        if err > tol:
            return False, (
                "face detected (conf %.3f) but its box maps back %.0f px from "
                "where the face was painted (tolerance %.0f px) -- the box "
                "mapping or the rotation sense is wrong" % (got.conf, err, tol))

        return True, (
            "pipeline OK: face at conf %.3f through rotation_deg=%d, box maps "
            "back within %.0f px (self-consistent by construction -- this does "
            "NOT validate the rotation SIGN; opposite-sense control scored "
            "%.3f)" % (best, rot, err, obest))

    def orientation_evidence(self, image):
        """Score a REAL raw camera frame at this rotation and at its opposite.

        Returns (conf_configured, conf_opposite). This is the only thing that
        can settle the rotation SIGN, because it is the only input that comes
        from the actual camera on the actual mount rather than from a buffer
        this module built out of its own assumptions.

        Used by the pre-arm face-interlock verification: put a real face in
        front of the narrow camera, and the configured rotation must win. If
        the opposite wins, NARROW_ROTATE_CLOCKWISE is backwards and the
        interlock is blind in normal operation -- silently, because a blind
        YuNet returns [] and control.py reads [] as an empty room.
        """
        rot = self.rotation_deg
        opposite = (360 - rot) % 360 if rot in (90, 270) else (180 if rot == 0 else 0)
        frame = Frame(image=image, t=0.0, index=0, camera="narrow")

        # BOTH scores under ONE lock hold. This method flips rotation_deg, and
        # the faces thread is still running at 30 Hz against the same object:
        # releasing between the two would let the live interlock read the
        # OPPOSITE rotation for a few frames, which is precisely the blind
        # configuration this check exists to rule out.
        with self._lock:
            here = max((f.conf for f in self._detect_locked(frame)), default=0.0)
            saved = self.rotation_deg
            try:
                self.rotation_deg = opposite
                there = max((f.conf for f in self._detect_locked(frame)),
                            default=0.0)
            finally:
                self.rotation_deg = saved
        return here, there


# ==========================================================================
#   Self-test
# ==========================================================================
def _synthetic_frame(size=config.NARROW_SIZE, camera="narrow") -> Frame:
    """A frame with drone-ish structure. Not a detection test -- a wiring test.

    Deliberately greyscale-ish: after the IR-cut removal, colour carries no
    information, and a synthetic scene that relied on colour would be testing
    something the real camera cannot deliver.
    """
    w, h = size
    img = np.full((h, w, 3), 90, np.uint8)
    img[: h // 2] = 130                                    # horizon
    cx, cy = w // 2, h // 2
    cv2.rectangle(img, (cx - 45, cy - 18), (cx + 45, cy + 18), (35, 35, 35), -1)
    for dx in (-90, 90):                                   # rotor arms + discs
        cv2.line(img, (cx, cy), (cx + dx, cy - 40), (40, 40, 40), 5)
        cv2.circle(img, (cx + dx, cy - 40), 34, (60, 60, 60), 2)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Frame(image=img, t=time.perf_counter(), index=0, camera=camera)


def _main() -> int:
    ap = argparse.ArgumentParser(description="turret_host detector self-test")
    ap.add_argument("--image", help="run on a real image instead of a synthetic one")
    ap.add_argument("--image-is-raw-buffer", action="store_true",
                    help="treat --image as a RAW narrow-camera buffer (already "
                         "rotated 90 deg by the mounting) rather than as an "
                         "upright picture. Changes which rotation the face "
                         "detector is given; see --image's output.")
    ap.add_argument("--weights", default=None, help="override YOLO weights")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 | cpu")
    ap.add_argument("--engine", action="store_true", help="load the TensorRT engine")
    ap.add_argument("--export-engine", action="store_true",
                    help="build the TensorRT engine and exit (minutes)")
    ap.add_argument("--fetch-face-model", action="store_true",
                    help="download the YuNet ONNX and exit")
    ap.add_argument("--repeat", type=int, default=5, help="timing iterations")
    args = ap.parse_args()

    if args.fetch_face_model:
        fetch_face_model()
        return 0
    if args.export_engine:
        export_engine(args.weights)
        return 0

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"could not read image: {args.image}")
        frame = Frame(image=img, t=time.perf_counter(), index=0, camera="narrow")
    else:
        frame = _synthetic_frame()
    print(f"[selftest] frame {frame.image.shape[1]}x{frame.image.shape[0]}")

    half = config.YOLO_HALF and args.device.startswith("cuda")
    det = Detector(weights=args.weights, device=args.device,
                   half=half, use_engine=args.engine).start()
    print(f"[selftest] weights   : {det.weights_path}")
    print(f"[selftest] classes   : "
          f"{[(i, det.model_names[i]) for i in det.class_ids]}")

    # THE ROTATION MATTERS AND IS EASY TO MISREAD, so the self-test is explicit
    # about it. The live narrow path feeds RAW BUFFERS, which the mounting has
    # already rotated 90 deg, so FaceDetector's default rotation un-rotates
    # them. A picture handed to --image is normally an UPRIGHT photo, i.e.
    # already in the orientation YuNet wants -- rotating that one would be the
    # bug, not the fix. Hence the flag, and the both-ways report below.
    upright_input = bool(args.image) and not args.image_is_raw_buffer
    faces = FaceDetector(rotation_deg=0 if upright_input else None).start()
    print(f"[selftest] face model: {faces.model_path}")
    print(f"[selftest] face pass : rotation {faces.rotation_deg} deg CW, "
          f"max_side {faces.max_side} "
          f"({'--image treated as an UPRIGHT picture' if upright_input else 'raw camera buffer, mounting rotation undone'})")

    # Both orientations on the same frame, once, so a human can see which one
    # the face was found in. A face that only appears at one rotation is the
    # whole content of BUILD_SPEC's face-detector exception.
    probe = FaceDetector(rotation_deg=0).start()
    probe_rot = FaceDetector(rotation_deg=None).start()
    n_flat = len(probe.detect(frame))
    n_rot = len(probe_rot.detect(frame))
    print(f"[selftest] orientation check: unrotated={n_flat} face(s), "
          f"rotated {probe_rot.rotation_deg} deg CW={n_rot} face(s)")
    if n_flat != n_rot:
        print(f"[selftest]   -> ORIENTATION IS DECISIVE on this frame. YuNet is "
              f"an upright-face detector; the live narrow path uses "
              f"{config.NARROW_ROTATION_DEG} deg "
              f"({'CW' if config.NARROW_ROTATE_CLOCKWISE else 'CCW'}).")
    probe.close()
    probe_rot.close()

    # The pipeline self-check, against the shipped face asset rather than the
    # synthetic scene -- on a drone-and-horizon test image both orientations
    # correctly find 0 faces, which tells a reader nothing about whether the
    # interlock works. This one uses a real face and therefore fails loudly if
    # the model, FACE_CONF, or the box mapping is broken.
    ok, detail = faces.pipeline_self_check()
    print(f"[selftest] pipeline self-check: {'PASS' if ok else 'FAIL'} -- {detail}")
    if not ok:
        print("[selftest]   -> THE FACE INTERLOCK IS NOT FUNCTIONAL. Do not arm.")
    print("[selftest] NOTE: the check above cannot validate the SIGN of "
          "config.NARROW_ROTATION_DEG -- it builds its test frame with the same "
          "constant. Only a real face through the real camera can "
          "(app.verify_face_interlock, pre-arm checklist item 1).")

    for i in range(max(1, args.repeat)):
        frame.index = i
        res = det.detect(frame)
        faces.detect_into(res, frame)
        print(f"[selftest] pass {i}: targets={len(res.targets)} "
              f"faces={len(res.faces)} infer={res.infer_ms:.1f} ms "
              f"(yolo {det.last_infer_ms:.1f} + yunet {faces.last_infer_ms:.1f})")
    for d in res.targets:
        print(f"           target {d.label} {d.conf:.2f} "
              f"[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}] area={d.area:.0f}")
    for d in res.faces:
        print(f"           face  {d.conf:.2f} "
              f"[{d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}]")

    if not args.image:
        print("[selftest] a synthetic scene is expected to yield 0 targets and "
              "0 faces -- this checks wiring and timing, not detection quality. "
              "Point --image at a real photo to sanity-check the face interlock.")
    det.close()
    faces.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
