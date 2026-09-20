# Detection & training brief — Laser Turret

**For:** whoever is building the perception layer, and the assistant helping them.
**From:** the hardware side. Written 2026-09-17.
**Showcase:** 22 September 2026.

---

## 0. Read this first — your job includes disagreeing with this document

This brief was written by someone who **has not built the perception layer and
has not tested most of what follows**. It is a considered starting point, not a
specification. Several decisions in it are reasoned from first principles
rather than measured, and §9 lists exactly which ones.

**Before executing any of this, audit it.** Specifically:

- Is the training approach in §5–§6 actually the right one, or is there a
  better path given five days?
- Is the detection approach in §4 sound, or is it solving the wrong problem?
- Are the numbers (frame counts, sampling rates, model sizes) defensible, or
  are they folklore?
- Is anything here going to waste a day that cannot be recovered?

**Start with §10.** Survey the actual machine before planning around it — two
of the worst failures on this project so far were machine-level (USB topology,
and a serial read that silently cost 150 ms per command), not code-level.

If you think a section is wrong, **say so before doing it**. A confident
objection now is worth more than a completed task that turns out to be the
wrong task. The deadline is tight enough that one wasted day is most of the
remaining budget.

Assume the author was wrong about at least one important thing. Find it.

---

## 1. What this system is, in one paragraph

A pan/tilt turret carries two cameras and a 5 mW green laser. It has to find a
drone held in a person's hand, 2–5 m away, indoors, in a room with people in
it, and keep the laser on the drone while the person moves it around. The
mechanical and firmware side is built and documented in `AGENT_HANDOFF.md` —
**read that before this**; it has the control API, the measured constants, and
the things that will surprise you. This brief covers only the part that is not
built: getting a detector that works on this specific footage.

## 2. What is already done (so you don't rebuild it)

- **Motion, firmware, control loop.** Velocity mode is built and measured —
  200 Hz loop, 43 ms host round trip, watchdog, travel limits. The API is
  `AGENT_HANDOFF.md` §4.8. You should not need to touch firmware.
- **A working 3D viewer** (`sim/wrist-ik.html`) that drives the real turret
  with a closed-loop velocity scheme. It is a reference implementation of the
  control side you can read and steal from.
- **Cameras characterised.** Use the **MSMF** backend on Windows, never DSHOW
  (measured: DSHOW caps the C270 at 10 fps while reporting 30). The wide camera
  is 1920×1080 and is *slower* at 720p. Details in `AGENT_HANDOFF.md` §5.2.
- **Backlash measured at 0.57°**, which is bigger than the step resolution and
  matters for how tightly you can expect to hold aim on a reversing target.

## 3. The hardware you are pointing at

| | |
|---|---|
| narrow camera | Logitech C270, 1280×720 @ 30 fps, **mounted rotated 90°**, **IR-cut filter removed** |
| wide camera | generic 120° UVC module, 1920×1080 @ 30 fps |
| laser | 532 nm, 5 mW, on the payload between the cameras |
| target | black quadcopter, ~283 mm diagonal, **held in a hand** |
| compute | RTX 4070 on the demo machine |

**Two of those matter more than they look.**

The C270's **IR-cut filter has been removed**. Near-infrared now reaches all
three colour channels, so colour is not photometrically meaningful and every
pretrained model you might use was trained on IR-cut imagery. This is a real
domain shift, and it is the main reason this brief argues for capturing your
own data.

The C270 is **mounted rotated 90°**, so its effective horizontal field is
28.8°, not 49°. Frames come out of the camera rotated. Decide early whether you
rotate frames (costs a copy per frame) or carry the rotation in your geometry
(free, but you must be consistent). The hardware side carries it in geometry.

## 4. The detection problem, and why public models disappoint

At 3 m the drone is about **132 px wide** in the narrow camera — roughly 10% of
frame width. That is a **large, close object**.

Every public drone dataset is the opposite: small drones against sky, because
that is the counter-UAS problem people fund. Two concrete examples that were
evaluated and rejected:

- `doguilmak/Drone-Detection-YOLOv7` — no metrics, no licence, no resolution
  stated, and the dataset explicitly keeps drones small.
- `FilippTrigub/yolov11x-drone-finetuned` — well documented (MIT, real
  metrics), but 56.8M parameters, recall 0.606, mAP50-95 0.279, single class
  with no `bird`, and trained on outdoor flight footage.

Neither has seen: a large close drone, indoor clutter, a hand partly occluding
it, or IR-contaminated colour.

**Suggested approach — challenge it if you disagree:**

1. **Start with YOLO-World** for zero-shot, so you have *something* detecting
   on day one while data collection happens:
   ```python
   from ultralytics import YOLO
   m = YOLO("yolov8s-worldv2.pt")
   m.set_classes(["drone", "quadcopter"])
   m.save("drone-world.pt")      # embeds vocabulary, drops the text encoder
   ```
   Try several prompts on real footage; which one fires is empirical.
2. **Use it to pre-label your own captured frames.**
3. **Fine-tune YOLO11n** on those frames. This is the step that actually makes
   it work.

The claim underlying all of this: *300 frames of your drone in your room beats
17,000 images of drones against sky.* **That claim is not measured. Test it** —
if zero-shot turns out to work well on your footage, skip the fine-tune and
spend the days on the control integration instead.

### Considered and rejected — overturn these if you disagree

**SAHI (Slicing Aided Hyper Inference).** Wrong tool here, on two counts. It
exists to find *small* objects in *large* images — satellite and aerial work,
where a target is a few pixels in a 4K frame. This drone is ~10% of frame
width, already large, so slicing finds nothing a whole-frame pass misses. And
it runs inference once per tile, so N slices means N forward passes; its own
documentation says it is not intended for real-time use. Since end-to-end
latency sets the control loop's gain ceiling, multiplying inference cost is the
one trade that cannot be afforded. It would be the right call for drones at
50 m against sky. Not for one in someone's hand at 3 m.

**A larger model (YOLO11x, the 56.8M-parameter drone checkpoint).** Same
reasoning: accuracy bought with latency is a poor trade when latency caps the
control bandwidth. Once fine-tuned on in-domain frames, a 132-px object against
a fixed indoor background does not need a large model.

## 5. Capture protocol — this determines your accuracy, not the model

You will record video of people holding the drone in front of the cameras.
**The variation in that footage matters more than the amount of it.** 300
near-identical frames teach almost nothing.

Deliberately vary:

- **distance** — 2, 3, 4, 5 m
- **occlusion** — different grips, covering different parts of the airframe.
  This is the hardest case and the one the demo actually contains.
- **drone orientation** — all three axes, including nose-on where it presents
  its smallest silhouette
- **background** — cluttered and plain parts of the room, people moving behind
- **motion** — record while moving, so you capture the motion blur the live
  system will see
- **lighting** — the range the actual demo room will have

**Capture negatives.** Frames with people holding *nothing*, and holding other
objects — a phone, a bottle, a mug. Without these the model learns "person
therefore drone", which fails in a room full of spectators.

**Use both cameras.** Their optics differ enough (49° vs 105°, different
distortion, one with its IR filter removed) that a model trained only on the
C270 may not transfer to the wide camera.

**You do NOT need to calibrate the cameras first.** Calibration buys stereo
range; it has nothing to do with labelling boxes. Do it later, in parallel.

**Match the live pipeline.** If you record H.264 to a drive and train on frames
extracted from it, you train on H.264 artifacts while inference sees
MJPEG-decoded frames. Capture stills through the same OpenCV path the live
system uses, or record MJPEG.

## 6. Frame sampling — and the mistake that invalidates your metrics

**Sample at 1–2 Hz** — roughly every 15–30th frame at 30 fps. Consecutive
frames are near-duplicates and teach nothing new.

**The rule that actually matters: split by clip, never by frame.** All frames
from one recording go entirely to train or entirely to validation. If
near-duplicate frames land on both sides, your validation score measures
memorisation and will look excellent while the model fails live. This is the
most common way people fool themselves, and it is silent.

Better than fixed-interval sampling: keep a frame only when it differs
sufficiently from the last kept one (frame difference or perceptual hash above
a threshold). This naturally keeps more frames during motion.

**Is 1–2 Hz right? Is 300 frames enough?** Both are heuristics. If you have a
better-grounded number, use it.

## 7. Labelling — do not build a tool

The hardware side sketched a custom labelling suite. **That was probably a
mistake and you should push back on it.** CVAT, Label Studio, Roboflow and
FiftyOne all already do model-assisted pre-labelling, box editing, reject,
keyboard-driven review, and YOLO-format export. Roboflow does the whole
upload → auto-label → correct → export → train loop in a browser.

Building a labelling tool with five days left is the most expensive way to
spend them.

Whatever tool you use, make sure it supports three distinct outcomes, because
they are not the same thing:

- **no drone present** — a valuable negative, keep the frame
- **frame unusable** — blur, drone half out of frame — discard it
- **model missed it** — the labeller must be able to draw a box from scratch

One idea worth keeping: **sort frames by model confidence and review the
least-confident first.** If you run out of time, you will have labelled the
frames that mattered rather than a random half.

## 8. Laser safety — this part is not negotiable

The demo points a laser at people holding a drone. Their heads are near their
hands. Treat this as a safety system, not a feature.

**Make the logic fail-closed.** "Fire unless a face is detected" is fail-open:
a dropped frame, a missed profile, or a model hiccup and the beam fires into an
unknown scene. **"Fire only when the aim point is inside a confirmed drone
box"** is fail-closed — any detection failure turns the beam off. Same models,
opposite failure mode. Make the drone lock the *permission*, and head detection
a veto on top of it.

**Heads, not faces.** Someone turned away is invisible to a face detector and
can turn round in ~0.3 s. Suggested approach, no training required:

- **`YOLO11n-pose`** — COCO-pretrained, nano speed, 17 keypoints including
  nose/eyes/ears. It estimates head position from body configuration, so it
  works from behind.
- **COCO `person` detection** as a backstop — the most reliable class in the
  dataset, fires from any angle; the head is roughly the top 15% of the box.
- **YuNet** (`cv2.FaceDetectorYN`) as a third signal when a face *is* visible.

Union all three. Note the subtlety: inhibit on the **head region only**, not
the whole person — the drone is in their hands, so a whole-person inhibit means
the laser never fires at all.

**Size the inhibit margin, don't guess it.** It must cover detection latency
plus turret slew plus human motion. At 3 m a head is ~25 cm ≈ 4.8°, so expect
the margin to be several head-widths. Compute it from the measured latency `L`
(`AGENT_HANDOFF.md` §6.1) once that exists.

**A hardware hazard worth raising with the hardware side:** cheap 532 nm DPSS
modules often emit substantial 1064/808 nm IR because the blocking filter is
omitted. Invisible light triggers no blink reflex, so the body's own protection
does not operate. This should be measured or an IR-blocking filter fitted. No
amount of head detection substitutes for it.

## 9. Things the author is NOT confident about — challenge these first

Listed honestly, worst-founded first. If you can cheaply test one, do that
instead of trusting it.

1. **That zero-shot YOLO-World will fire on this drone at all.** Completely
   untested. Ten minutes with real footage settles it, and the answer changes
   the plan.
2. **That 300 frames is enough.** Rule of thumb. Could be 150, could be 800.
3. **That fine-tuning from COCO beats fine-tuning from a drone-pretrained
   checkpoint.** Argued from "the drone checkpoints are out-of-domain", but
   starting from 17k drone images might still transfer better. Worth one
   experiment if time allows.
4. **That the IR-filter removal hurts pretrained models as much as claimed.**
   Reasoned, not measured. Easy to check: run any COCO model on a filtered and
   an unfiltered frame of the same scene.
5. **That `YOLO11n-pose` reliably locates heads from behind.** Plausible,
   unverified.
6. **That nano at `imgsz=1280` is fast enough on a 4070** alongside capture,
   pose, tracking and display. `AGENT_HANDOFF.md` asserts a few ms. Measure it.
7. **That 1–2 Hz is the right sampling rate.**
8. **Whether to train one model on both cameras or one per camera.** Untested
   either way.
9. **Whether the demo narrative needs a `bird` class at all.** It is in the
   story ("confirms it isn't a bird") but may be pure narrative — COCO already
   knows `bird` if you need it.

## 10. Survey the machine before you plan around it

Do this first. It takes ten minutes and it changes what is worth attempting.
Several of the decisions in this brief assume performance that your specific
machine may or may not have, and two of the biggest failures on this project so
far were machine-level rather than code-level.

### What to collect

```bash
# GPU, driver, VRAM
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('VRAM GB', torch.cuda.get_device_properties(0).total_memory/1e9)"

# CPU and RAM
python -c "import os,psutil; print('cores', os.cpu_count(), 'RAM GB', round(psutil.virtual_memory().total/1e9,1))"
```

```powershell
# USB topology -- which ports share a controller (see below, this matters)
Get-PnpDevice -Class Camera,USB | Where-Object Status -eq OK |
  ForEach-Object { $_.FriendlyName; (Get-PnpDeviceProperty -InstanceId $_.InstanceId `
    -KeyName DEVPKEY_Device_LocationInfo).Data }

# Sustained write speed of the drive you plan to record to
winsat disk -drive E          # substitute your drive letter
```

### What each result should change

**GPU and VRAM.** Decides model size and whether TensorRT is worth it.
TensorRT FP16 export typically buys 2–3× over raw PyTorch and is one line
(`model.export(format='engine', half=True)`), but the first export takes
minutes and needs a working CUDA toolchain. If `torch.cuda.is_available()` is
False, stop and fix that before anything else — everything downstream assumes
it. Compute capability ≥ 7.0 means FP16 tensor cores are worth using; on a 4070
(8.9) they definitely are.

**CPU cores.** Matters more than people expect here, because **MJPEG decode
happens on the CPU and holds the GIL**. `AGENT_HANDOFF.md` §5.1 lays out the
thread structure. If you are core-starved, move JPEG decode to the GPU with
`torchvision.io.decode_jpeg(..., device='cuda')` — 1–2 ms instead of 6–12 ms,
and the frame lands where inference wants it anyway.

**USB topology — this one has already bitten this project.** Two UVC cameras
on the same hub or USB-C splitter will not both stream. Each reserves
isochronous bandwidth at start-up, so the second one is refused outright, and
Windows reports `ERROR_DEVICE_NOT_CONNECTED` (`0x8007048F`) — which is a lie,
because the device is present and works perfectly alone. **Lowering the
resolution does not rescue it:** measured on this project, the second camera
refused to start at 1080p, at 640×480 *and* at 320×240.

Compare the `LocationInfo` strings. Two cameras sharing a path prefix share an
upstream link. Put each camera on a **separate native port**, not two sockets
on the same hub. `turret_vision/dual_capture_test.py` checks this in about a
minute and tells you what does coexist.

**Disk — record to the host, copy afterwards.** That is the plan and it is the
right one. Two MJPEG streams (1080p30 + 720p30) is roughly 10–20 MB/s
sustained, and plenty of USB flash drives cannot hold that: they benchmark well
in bursts and collapse on sustained writes, at which point you drop frames
silently and only find out when the footage is unusable and everyone has gone
home. Writing to the host's internal SSD sidesteps that entirely, and the copy
to external storage afterwards is not time-critical.

Two things still worth checking: that the host has **free space** (roughly
1 GB per minute for both streams together, so budget tens of GB for a session),
and that the capture actually sustains frame rate. **Do one 60-second test
recording and count the frames before the real session** — expect 1800 per
stream at 30 fps. If you are short, you are dropping them, and no amount of
labelling fixes footage that was never captured.

**Laptop or desktop.** If it is a laptop, check whether it throttles. A model
that hits frame rate for thirty seconds and then drops as the package heats up
will look like a software bug. Run the detector flat out for five minutes and
watch `nvidia-smi` clocks before believing any benchmark.

### Then measure, don't assume

Once the survey is done, benchmark the actual pipeline rather than the model in
isolation:

```python
# end-to-end, not just model.predict()
# capture -> decode -> preprocess -> infer -> postprocess
```

The number that matters is **end-to-end latency**, because it sets the control
loop's gain ceiling — `AGENT_HANDOFF.md` §8 puts the usable P-loop bandwidth at
roughly 1/(8L). A model that runs in 4 ms inside a pipeline that takes 90 ms
end to end gives you a 90 ms loop, not a 4 ms one.

---

## 11. Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install ultralytics opencv-python
# PyTorch with CUDA for the 4070 -- check the current index URL at pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`ultralytics` pulls in YOLO11, YOLO-World and the pose models. TensorRT export
(`model.export(format='engine', half=True)`) needs a working CUDA install and
takes a few minutes the first time.

Sanity check before anything else — confirm both cameras open and deliver full
frame rate:

```bash
python turret_vision/identify_cameras.py     # resolve by USB identity, not index
python turret_vision/probe_camera.py         # real delivered fps per mode
python turret_vision/dual_capture_test.py    # both at once; USB bandwidth
```

Those exist and are documented. **Camera indices are not stable on Windows** —
resolve by VID/PID, never hardcode an index.

## 12. Suggested order, given five days

This ordering is itself a guess. Reorder it if you see better.

1. **Today** — get the cameras running at full rate, get YOLO-World detecting
   *something* on live video. This tells you how much of §5–§7 you actually
   need.
2. **Then** — capture footage with the variation in §5. This is the only input
   nobody else can substitute for, and it gates everything downstream.
3. **Then** — pre-label, correct, fine-tune YOLO11n. Roughly half a day.
4. **In parallel** — the control integration against `AGENT_HANDOFF.md` §4.8
   and the fail-closed safety logic in §8. This is independent of the detector
   and can start immediately.
5. **Last** — stereo calibration, if you want range at all. §1.5 of the handoff
   shows a fixed 3 m assumption costs at most 7 px of aim error across the
   whole 2–5 m envelope, so this may not be worth doing.

---

**Again: question this document.** It was written by the hardware side, who
has measured a great deal about the mechanism and almost nothing about the
detector. The parts about motion, timing and the physical machine are backed by
measurements. The parts about training and detection are reasoning, and §9 is
an honest list of where that reasoning is thin.
