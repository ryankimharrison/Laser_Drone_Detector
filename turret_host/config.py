"""Measured constants for the tracking stack.

Values marked MEASURED were verified on this hardware. Values marked ESTIMATE
are starting points that a calibration routine overwrites. Do not "tidy" an
ESTIMATE into a MEASURED by guessing -- run the calibration.

Where this file disagrees with AGENT_HANDOFF.md, this file wins: several of the
doc's camera claims were disproved on this machine (see CAMERA notes below).
"""

# ==========================================================================
#   BOARD LINK
# ==========================================================================
PICO_VID, PICO_PID = 0x2E8A, 0x0005     # resolve by VID/PID, never COM number
SERIAL_BAUD = 115200                    # USB CDC: baud is ignored, any value works

# Firmware VEL_WATCHDOG_MS is 400. Every command period must stay well under
# it or the motors park between updates -- that failure looked exactly like a
# tuning problem and cost the previous builder a long debug.
VEL_WATCHDOG_MS = 400
VEL_COMMAND_PERIOD_MAX_MS = 150         # refuse to run if we get slower than this

# ==========================================================================
#   PLATFORM  (from firmware `diff` / `limits`, verified 2026-09-16)
# ==========================================================================
AXIS_STEP_DEG = 0.04875                 # MEASURED: payload deg per axis-step, 1/16
DIFFERENTIAL_N = 2.3077                 # MEASURED: 26T -> 60T belt
MOTOR_MAX_RATE = 4000                   # steps/s per motor, firmware max_rate
PITCH_LIMIT_DEG = (-90.0, +90.0)        # MEASURED: from TRUE LEVEL, not an arbitrary zero
YAW_LIMIT_DEG = (-225.0, +225.0)        # MEASURED: payload wiring loom limit

# MEASURED 2026-09-16 by IMU hysteresis loop. At 3 m this is 30 mm of beam
# wander on every direction reversal -- an order of magnitude past the 2.55 mm
# step resolution. Backlash, not resolution, sets accuracy on a reversing
# target, and a hand-held drone reverses constantly.
BACKLASH_DEG = 0.57
PRELOAD_STEPS = 14                      # 1.2 x the ~12 microsteps of lash

# ==========================================================================
#   CAMERAS
# ==========================================================================
# Both cameras are USB 2.0 and each reserves ~24.6 MB/s of isochronous
# bandwidth at stream start, against a ~48 MB/s periodic budget PER xHCI ROOT
# PORT. Two cameras on one root port => the second stream is refused with a
# lying ERROR_DEVICE_NOT_CONNECTED. Hubs are fine; sharing a root port is not.
# VERIFIED on this machine: separate root ports => both stream concurrently.
#
# MSMF vs DSHOW, both MEASURED here:
#   MSMF  -- delivers full frame rate, but SILENTLY IGNORES exposure writes
#            (set() returns True, value never changes)
#   DSHOW -- exposure writes work, but frame rate collapses (C270: 10 fps)
# So: set exposure through a short DSHOW open, release, reopen on MSMF.
NARROW_SERIAL = "C8258920"              # C270 has a real serial -- fully stable
NARROW_VID_PID = (0x046D, 0x0825)
WIDE_VID_PID = (0x32E4, 0x9230)         # SVPRO USBFHD01M, OV2710 + 2.1 mm M12
WIDE_FINGERPRINT_WIDTH = 1920           # no serial: identify by max resolution

# The HP integrated webcam also caps at 1280, so a max-width fingerprint CANNOT
# distinguish it from the C270. The narrow camera must be resolved by serial.
INTEGRATED_VID_PID = (0x30C9, 0x0069)

NARROW_SIZE = (1280, 720)               # MEASURED: 30.1 fps MJPEG on MSMF; its ceiling
NARROW_FPS = 30
WIDE_SIZE = (1920, 1080)                # MEASURED: 30.3 fps MJPEG
WIDE_FPS = 30

# The wide module also does 1280x720 @ 60 fps MJPEG -- AGENT_HANDOFF says this
# mode delivers 9 fps and calls it a trap. That 9 fps is the YUYV variant;
# forcing MJPEG gives a MEASURED 59 fps. It is a CENTER CROP, not a downscale:
# horizontal field drops ~108 -> ~85 deg while angular resolution is unchanged.
WIDE_FAST_SIZE = (1280, 720)
WIDE_FAST_FPS = 60

# EXPOSURE SETS THE C270's REAL FRAME RATE, AND -6 BLINDS THE DETECTOR.
#
# The C270 cannot start a new frame until the current one has finished
# integrating, so a long exposure halves its true rate while the UVC driver
# pads the stream back to 30 by REPEATING the last image. MEASURED on the rig
# 2026-09-20, distinct-frame rate and detector hit-rate at 1280x720:
#
#     exposure   distinct   brightness   YOLO v4 hits
#     (as found)   ~17 fps      126-140         --
#     -4         13.9 fps          124         --
#     -5         25.4 fps           79      12/12 at conf 0.755
#     -6         27.1 fps           36       0/12  BLIND
#     -7         27.2 fps           20       0/12  BLIND
#
# So -5, not -6. -6 buys 1.7 fps over -5 and costs every detection: the
# detector found the drone on none of twelve frames. The frame-rate plateau is
# already reached at -6, so there is nothing above -5 worth having.
#
# TWO TRAPS, both measured:
#  * The camera RETAINS exposure across process restarts. Whatever the last
#    process set, the next one inherits, which is why observed brightness has
#    wandered 9-140 between runs all day. Always SET it, never assume.
#  * MSMF's exposure READBACK IS A CONSTANT -- it returns the same value
#    whatever you write, while the image responds correctly. Any verification
#    by read-back is worthless. See lock_exposure().
NARROW_EXPOSURE_EV = -5
WIDE_EXPOSURE_EV = -6                   # wide sensor is fast at any of these

MIN_ACCEPTABLE_FPS = 25                 # refuse to start below this, per camera
FPS_PROBE_FRAMES = 40

# DUPLICATE FRAME REJECTION.
#
# MEASURED 2026-09-19 (run_140019): 43 saved frames contained only 28 DISTINCT
# images -- 35% of what the capture thread published was the SAME picture
# again. cv2 grab() returns immediately for an image already sitting in the
# driver buffer, and cameras.py stamps perf_counter() right after grab(), so a
# re-delivery gets a FRESH timestamp and a FRESH frame index. Nothing
# downstream could tell it apart: Slot.put() bumps its sequence number for
# every put, and every consumer guards on that sequence, not on content.
#
# The cost is not a wasted inference. The tracker takes the duplicate as an
# independent measurement at dt ~ 5 ms, where zero innovation collapses P; the
# next genuine frame then reads as a 10-20 sigma outlier and trips the Q spike.
#
# EXACT equality is the test, deliberately. A truly static scene (turret
# parked, nothing moving) legitimately produces near-identical frames and those
# are real observations that must survive. Only a bit-for-bit repeat is a
# re-delivery.
#: Ask the board for its own ticks_ms alongside each IMU sample, so an IMU row
#: is timed by the BOARD rather than by when the host got round to reading it.
#: One extra integer print per sample; clocksync.py fits host<->board offset and
#: skew offline from these. Turn OFF if it ever starves the vel stream -- the
#: IMU loop shares the io lock with the command path.
IMU_LOG_BOARD_TICKS = True

DEDUP_FRAMES = True
#: Rows/cols step for the cheap pre-filter. A 1280x720x3 frame decimates to
#: ~10 KB at 16, which rejects a genuinely new frame in microseconds; the full
#: exact comparison then runs ONLY on candidates that survive it.
DEDUP_STRIDE = 16

# ==========================================================================
#   OPTICS
# ==========================================================================
# Narrow: Logitech C270, IR-CUT FILTER REMOVED, MOUNTED ROTATED 90 degrees.
# The filter removal is a real domain shift for COCO weights and weakens every
# colour-based test; the rotation means its parallax correction lands on the
# goal ROW, not the column.
NARROW_F_PX = 1400.0                    # ESTIMATE -- run chessboard calibration
NARROW_IR_CUT_REMOVED = True
NARROW_REFOCUSED = True                 # builder confirmed sharp at 2-5 m after filter removal

# The C270 is physically mounted rotated. AGENT_HANDOFF 1.5 left the SIGN open
# ("90 vs 270 is not yet verified ... getting it backwards inverts the parallax
# correction rather than merely misplacing it").
#
# RESOLVED 2026-09-16, builder observed the live preview: the raw frame must be
# rotated 90 degrees CLOCKWISE to appear upright.
#   display = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
NARROW_ROTATION_DEG = 90
NARROW_ROTATE_CLOCKWISE = True

# ROTATE FOR DISPLAY ONLY. Never rotate frames on the processing path: it is a
# full-frame copy per frame for no benefit, and the detector does not care which
# way up the world is. The geometry carries the rotation instead.
#
# Consequence of the rotation: the laser/camera offset is horizontal in the
# PAYLOAD frame, but in the C270's own frame it lands on the sensor's VERTICAL
# axis -- so this camera's parallax correction applies to the goal ROW, while
# the wide camera's applies to its COLUMN. Apply it per camera, not globally.
#
# Even now that the direction is known, DO NOT hand-derive the parallax sign
# from it. g(R) is fitted from two measured ranges (see calibrate.py) and that
# fit determines direction and magnitude empirically. The sign ambiguity only
# ever bites someone who hard-codes the correction from the CAD numbers.

# Wide: OV2710, 3.0 um pixels -> 5.76 x 3.24 mm active, with a 2.1 mm lens.
# f_px = 2.1 mm / 3.0 um = 700. Horizontal FOV 2*atan(2.88/2.1) = 107.9 deg.
# At 115 deg diagonal this lens is NOT rectilinear -- expect to need
# cv2.fisheye. Calibrate; M12 focal lengths run +/-10%.
WIDE_F_PX = 700.0                       # DERIVED from sensor geometry, not measured
WIDE_FOV_H_DEG = 107.9

# MEASURED from CAD (sterolaserview.step), all three features on y = 0:
#     wide x=-30.000    laser x=0.000    narrow x=+30.000  (mm)
LASER_TO_NARROW_MM = +30.0
LASER_TO_WIDE_MM = -30.0
STEREO_BASELINE_MM = 60.0

# Aiming at the drone BODY works with no range estimate at all: assuming a flat
# 3 m across the whole 2-5 m envelope costs at most 7 px against a 132 px
# target. Build the loop against this, then add ranging to tighten it.
#: How far the DEAD-RECKONED pitch may exceed what gravity measures before the
#: pose stops being treated as evidence, degrees.
#:
#: Dead reckoning is open-loop and its error is unbounded. MEASURED on
#: run_2026-09-19_132817: over the last 8.9 s the dead-reckoned pitch reached
#: -92.47 deg -- PAST the -90 soft stop -- while gravity said the payload had
#: never left a 35.84 deg total tilt. A ~54 deg disagreement, against a
#: LIMIT_MARGIN_DEG of 3.0. TravelLimits then derated the pitch axis on a pose
#: that never physically happened, on all 51 of the corpus's pure-yaw rows.
#:
#: THE COMPARISON IS AXIS-FREE, and it must stay that way. Gravity's TOTAL
#: angle from the datum is blind to yaw and so is a LOWER BOUND on |true
#: pitch|: a pure yaw reads 0, a pure pitch reads |theta|. So |DR pitch|
#: exceeding the measured total by this much means the DR pose is wrong --
#: which is sound without ever asking WHICH IMU axis a payload pitch lands on.
#: That question depends on yaw and has already produced one sign bug here.
#: THE THRESHOLD IS NOT ARBITRARY -- the disagreement is BIMODAL. Measured
#: over all 6621 rows, |DR pitch| - measured total tilt:
#:     median -4.4    p75 +5.9    p90 +43.4    p95 +56.3    max +69.3
#: So rows either agree within ~6 deg or are wrong by 43-69 deg, with almost
#: nothing between. Any tolerance from 10 to 30 selects the same population
#: (22.0% down to 14.9% of rows); 20 sits on that plateau. The incident above
#: sits at +57 deg.
#:
#: THAT ~19% IS ITSELF THE HEADLINE: the dead-reckoned pose is not
#: occasionally wrong, it is wrong by tens of degrees on a fifth of all rows.
#: The derate has been acting on that. This check is ONE-SIDED by design -- it
#: only refuses a pose claiming MORE pitch than gravity can support, so it can
#: prevent phantom braking but never remove braking that was warranted. The
#: firmware keeps its own step-based pitch limit and the physical endstops
#: underneath all of this, and the measured ATTITUDE_MAX_DEG envelope stays
#: live, so refusing the derate does not leave the axis unprotected.
POSE_DR_DISAGREE_MAX_DEG = 20.0

ASSUMED_RANGE_M = 3.0
RANGE_LIMITS_M = (1.5, 6.0)

# Real width of the drone as seen by the detector, for apparent-size ranging:
#   R = f * W_real / w_px
#
# The aircraft is a DJI MINI 2 (operator, 2026-09-19). Published unfolded
# dimensions WITH propellers are 245 x 289 x 56 mm, so the widest silhouette
# is 289 mm. Propellers included because the detector boxes what it can see
# and the props are in the box; the 213 mm "diagonal distance" in the spec
# sheet is motor-shaft to motor-shaft and is NOT the visible extent.
#
# CROSS-CHECKED AGAINST THIS RIG'S OWN FOOTAGE, and this is why it is trusted
# rather than merely quoted: the drone measures ~132 px across at 3 m, so
# W = R * w_px / f = 3.0 * 132 / 1400 = 283 mm. Published 289, observed 283 --
# 2% apart, from completely independent sources.
#
# THE ERROR BAR IS ORIENTATION, NOT THE NUMBER. An X-quad's silhouette width
# swings by about 1/sqrt(2) between nose-on and 45-degrees-off, so a single
# constant makes range good to roughly +/-20% depending on pose. That is still
# far better than the fixed 3.0 m it replaces: the beam is offset from the
# camera, so the goal pixel genuinely moves with range (see the g_inf + c/r
# model in calibrate.goal_pixel_for_range), and at 5 m a 3 m assumption is
# wrong by the whole parallax term rather than by 20% of it.
#
# RANGE_LIMITS_M still clamps the result, so a wild box cannot produce a wild
# aim point.
DRONE_WIDTH_M = 0.289

# ==========================================================================
#   CONTROL
# ==========================================================================
# ESTIMATE. Calibrated by calibrate.py: move each motor 200 steps and measure
# the pixel shift. This 2x2 absorbs the belt ratio, the differential, the joint
# order, the 11 deg axis tilt, focal length and every sign -- which is why the
# loop tolerates it being 20% wrong.
J_PX_PER_STEP = None                    # set by calibration; None = not calibrated

# 1/s. The procedure in the original note -- "raise until the step response
# overshoots, back off 30%" -- was never run; 2.0 was a starting value.
#
# MEASURED 2026-09-19 at K=2.0, on 562 frames with a real box every frame at
# conf 0.50-0.78: the command reversed sign on 20% of frames and the error
# oscillated 37 -> 157 px without ever settling. That is overshoot, and it is
# why the beam could never fire (MAX_ERROR_TO_FIRE_PX = 25).
#
# L IS NOW MEASURED -- see LATENCY_S below. 0.066 s, so the ceiling 1/(8L) is
# 1.89 /s, and:
#
#     K = 2.0  ->  1.06 x ceiling   ABOVE it. This is the oscillation above.
#     K = 1.2  ->  0.64 x ceiling   inside it.
#
# The "back off 30%" rule wants 0.7 x 1.89 = 1.32, and 1.2 is already there
# within the precision either number deserves. So THE GAIN IS NO LONGER THE
# SUSPECT: do not keep lowering it looking for the glue. Whatever is left
# after K=1.2 -- 235 px mean error, still not settling -- is not loop gain,
# and the association gate is the measured candidate (the detector sees the
# drone on 79% of COAST frames and association rejects 100% of them).
#
# The earlier guess in this comment, "at a true 90 ms the ceiling is 1.4", had
# the direction right and the size wrong: 66 ms, ceiling 1.89, and K=2.0 was
# over by 6% rather than by half.
#
# NOTE the metric named in the old version of this comment is BROKEN. At lower
# gain more commands sit near zero, so a trivial +/-0.5 deg/s reversal counts
# the same as a +/-70 deg/s one -- which is why sign flips went UP (20% ->
# 26%) on the change that improved every error measure. Weight reversals by
# magnitude before using them to judge anything.
CONTROL_GAIN_K = 1.2
ERROR_DEADBAND_PX = 1.0                 # on the P term only
MAX_MOTOR_RATE = 4000                   # steps/s, clipped before send
# 2026-09-20: slew cap while the controlling box is WIDE-SOURCED. Measured on
# run_2026-09-20_122343: the wide box is 100-160 ms old and arrives at 6-12 Hz,
# and at the 50-65 deg/s the P law commanded from it the narrow camera saw only
# motion blur, so it could never take over; the turret overshot the drone and
# tripped the attitude envelope. 600 steps/s on the faster motor is ~29 deg/s of
# payload: ~4 deg of staleness error at 150 ms, ~11 px of blur at exposure -6.
# Direction-preserving scale, applied in app._control_loop. 0 disables.
WIDE_MAX_MOTOR_RATE = 600
# 2026-09-20: while a track is live, the wide fallback may substitute for the
# narrow detector only after this many CONSECUTIVE narrow misses. Measured on
# run_2026-09-20_123719 (static drone): every one of 24 single-frame narrow
# misses was filled with a 50-140 ms old wide box whose mapped position sat
# tens of px from the narrow one, and each produced a 30-80 px excursion on a
# target that was not moving. A one-frame dropout should coast on the filter.
# Acquisition from SEARCH is unaffected: wide still seeds a new track.
WIDE_FALLBACK_AFTER_MISSES = 3

# --------------------------------------------------------------------------
#   COARSE / FINE ACQUISITION CASCADE
# --------------------------------------------------------------------------
# Operator's design: the WIDE camera finds the drone, the turret slews until
# the NARROW camera can see it, narrow then takes over, and only then may the
# laser fire. Two PHASES with a handover, not a measurement preference.
#
# WHY A CASCADE AND NOT "PREFER WIDE": the wide->narrow mapping carries ~0.5
# deg of parallax/range error (60 mm baseline; ASSUMED_RANGE_M is 3.0 while the
# true range is nearer 2.05). Against the narrow capture window of +/-14.4 deg
# that is 3.7% of the budget -- irrelevant. Against the 25 px firing tolerance
# (1.02 deg) it is HALF the budget -- serious. So the wide measurement is used
# ONLY to get the target into the narrow field, and is DROPPED ENTIRELY the
# moment narrow acquires. It must never touch the fine loop.
#
# WHY THE TERMINATION MATTERS MORE THAN THE APPROACH: the 1065 px runaway on
# 2026-09-20 WAS an unterminated coarse slew. The old code substituted a wide
# box mapped ~400 px outside the narrow frame and drove at it forever, until
# the measured-attitude envelope stopped the payload at 68 deg pointing at the
# floor. A coarse phase needs a GOAL and a GIVE-UP or it rebuilds that bug.
CASCADE_COARSE_ENABLED = True

#: Arrival: the mapped box overlaps the narrow frame, so the narrow detector
#: has something to find. Checked with camera_fusion.in_narrow_view, whose
#: margin EXPANDS the acceptance region -- negative values require the box to
#: be that many px INSIDE the edge before the approach is called done.
COARSE_ARRIVE_MARGIN_PX = -40.0

#: Give-up. Stop driving on the wide box after this long without the narrow
#: detector confirming, log once, and let the tracker fall back to SEARCH.
#: Bounds the slew in TIME, which is the guarantee the old code lacked.
COARSE_MAX_S = 3.0

#: After giving up, wait this long before another coarse approach is allowed,
#: so a target the narrow camera cannot resolve does not produce a pumping
#: slew-stop-slew cycle.
COARSE_COOLDOWN_S = 2.0

# MEASURED 2026-09-19, `measure_latency.py --wall`, n=16, 0 misses.
# Laser flashed against a plain wall, room lit; onset is the first frame whose
# GREEN EXCESS breaks a floor measured on the day (26.27, sd 1.92 -> threshold
# 37.80 -- quoted because a latency without its detection threshold cannot be
# checked). Command -> photons visible in a frame: median 51.2 ms, sd 7.2.
# Plus ~15 ms inference, which the loop pays before the controller sees
# anything:  0.051 + 0.015 = 0.066.
#
# Measured at narrow exposure -6, which is what app.NARROW_EXPOSURE uses in
# flight, so the integration time inside this number is the real one. Re-run
# it if that exposure ever changes.
#
# The 51.2 is deliberately NOT de-quantised down to the true photon arrival
# (which the 33 ms frame interval places somewhere above 17.9 ms). The control
# loop is fed by frames, so it genuinely waits out that quantisation too --
# the whole-frames figure is the one the gain ceiling should be built on.
LATENCY_S = 0.066                       # MEASURED (was 0.060, an estimate)

# ==========================================================================
#   PLATFORM COMPENSATION   (DEFAULT OFF -- never validated on hardware)
# ==========================================================================
# What the camera sees move is the DRONE plus the TURRET:
#
#     observed image velocity  =  drone motion  +  platform motion
#
# The filter cannot separate them -- it sees one number. So its velocity
# estimate silently contains however fast the turret happened to be slewing
# while the last detections arrived, and `predict_to(t + LATENCY_S)` carries
# that forward as if it were the drone's.
#
# But the platform term is the one thing here that is KNOWN IN ADVANCE rather
# than measured after the fact: we command it. That makes the command strictly
# better information than the gyro for PREDICTION -- the gyro reports what the
# turret already did, the command says what it is about to do, and the
# VelocityLoop holds a commanded rate until told otherwise.
#
# The correction is smaller than it first looks. The filter's velocity already
# embeds the platform rate that was in effect when the detections arrived, so
# only the CHANGE matters:
#
#     correction_px = J.pixel_rates(omega_now - omega_at_last_detection) * L
#
# Zero while the command is steady -- which is most of the time, and is why
# this is a refinement rather than a rescue. It bites exactly when the command
# has just changed, which is when the loop is turning around and when the
# prediction is worst.
#
# OFF BY DEFAULT because it has never run against a moving drone. It is also
# the half of the deliverable that says "interpolate where the drone will be
# between inference captures", so it should be turned on and MEASURED with
# tools/analyze_run.py -- lead error, against the same recorded baseline --
# rather than turned on and believed.
# TESTED OFFLINE 2026-09-20 AND IT DOES NOT WORK AS BUILT. Replayed against
# three recorded runs: if this correction captured real platform motion, it
# would point along the lead error, i.e. cos(error, correction) near +1.
# Measured cos was +0.001, +0.198, +0.142 -- essentially orthogonal -- and
# subtracting it made the error WORSE in two runs of three.
#
# The diagnosis, which also corrects something claimed earlier in that session:
# stepper.VelocityLoop RAMPS toward a commanded rate rather than jumping to it,
# so omega_commanded != omega_actual exactly during the transitions where this
# correction is non-zero. It corrects with a rate the platform has not reached.
#
# So the command is better information than the gyro only where the platform
# FOLLOWS the command. During the transient it does not, and the gyro -- now at
# 500 Hz, which it was not when these runs were recorded (10 Hz, three times
# coarser than the frame rate) -- is the thing that measures what the platform
# actually did. Re-test this against a run recorded with the 500 Hz stream
# before changing the default.
PLATFORM_COMPENSATION = False

# --------------------------------------------------------------------------
#   FEEDFORWARD ON THE TARGET'S *INERTIAL* VELOCITY
# --------------------------------------------------------------------------
# DEFAULT OFF. Changes the control law; A/B it against a recorded run first.
#
# THE DEFECT THIS FIXES IS IN THE CODE'S OWN DERIVATION. control.py derives:
#
#       J = d(image position of a world-fixed point) / d(motor step)
#       with e = u_hat - goal:   d(e)/dt = J @ omega + v_target
#       so                       omega   = -J_inv @ (K*e + v_target)
#
# The feedforward input it calls for is v_target -- the target's INERTIAL
# velocity. What is actually passed is est.du/dv, which is the filter's
# IMAGE-PLANE velocity, and the camera rides the payload, so
#
#       est.du = v_target + J @ omega          (not v_target)
#
# Those are not the same quantity, and the difference is not small: J @ omega
# is driving toward -v_target by design, so the two CANCEL and the feedforward
# input collapses toward zero exactly when tracking starts working.
#
# CONSEQUENCES, both derivable and both measured:
#   * A STANDING ERROR of v_target / K. Substituting est.du for v_target gives
#     de/dt = -(K/2)e + v_target/2, whose steady state is e = v_target/K, not
#     zero. Measured: 20 of 88 servoing segments hold a large FLAT error while
#     the turret slews -- median error 345 px, median slew 524 px/s, while the
#     filter reports only 110 px/s of target motion (21% of the truth).
#   * HALF THE INTENDED GAIN. The effective closed-loop gain is K/2, not K.
#     So K=1.2 has been acting as 0.6 -- which also means enabling this is not
#     a destabilising change: effective gain goes 0.6 -> 1.2, still below the
#     1.89 ceiling implied by the measured 66 ms latency. Do NOT also raise K.
#
# With the subtraction, de/dt = -K*e exactly, independent of target speed.
#
# NOT THE SAME THING AS PLATFORM_COMPENSATION. That flag corrects POSITION by
# J @ (omega_now - omega_at_box) * LATENCY_S -- only the CHANGE in rate. This
# corrects VELOCITY by subtracting J @ omega_at_box outright. A STEADY platform
# rate contributes nothing to that flag and is exactly what cancels a steady
# target velocity here. Different term, different magnitude; only this one
# removes the standing lag. They are independent and may be enabled separately.
#
# COST: v_target_hat carries all of est.du's noise PLUS Jacobian error on
# omega, so it is noisier than est.du. MAX_TARGET_ACCEL_MPS2 matters MORE with
# this on, not less.
# !! DO NOT ENABLE THIS YET. MEASURED UNSAFE 2026-09-20. !!
#
# The DIAGNOSIS above stands -- the derivation asks for v_target, the line
# passes est.du, and the feedforward really does partially self-cancel. The
# IMPLEMENTATION below is not a usable fix, for two independent reasons, both
# measured after it was written:
#
# 1. THE COMMANDED RATE IS NOT THE ACHIEVED RATE. Lucas-Kanade on background
#    corners (a direct measurement, independent of J and of the control law)
#    puts ACTUAL platform image motion at a median 49 px/s against a commanded
#    |J @ omega| of 119 px/s over the same pairs -- the command over-predicts
#    real ego-motion by 1.6-2.0x by regression slope, ~2.4x by median. Best
#    correlation lag is +80 ms, which is the VEL_ACCEL ramp's signature (see
#    slew ceiling notes: 100 ms from rest to full rate). Subtracting the
#    COMMANDED term therefore removes roughly twice the platform motion that
#    actually happened.
#
# 2. WORSE, AND FOUND HERE: the commanded rate is an EXACT FUNCTION of the
#    measurement it would be subtracted from. The law sets
#        J @ omega = -(K*e + est.du)
#    so  est.du - J @ omega_at_box  reconstructs  est.du + K*e_prev + du_prev,
#    i.e. roughly 2*est.du + K*e. Measured against the logs: the subtraction's
#    output correlates +0.90 / +0.91 with that identity, and its median
#    magnitude is 569 px/s against a median |est.du| of 127 px/s. It does not
#    estimate target velocity; it largely recovers the control law's own
#    output, amplified. For scale, the only direct measurement of true target
#    speed available (optical flow) is ~176 px/s -- so this would feed the
#    loop a target velocity roughly 3x too large.
#
#    This is the SAME circular quantity that was identified and discarded as an
#    ANALYSIS method ("it just recovers 2*v_measured + K*e") and then proposed
#    as a CONTROL fix without the circularity being carried across. Subtracting
#    a MEASURED platform rate would be sound; subtracting a COMMANDED one that
#    the law computed from this very measurement is not.
#
# 3. AND THE MECHANISM'S OWN PREDICTION FAILS A DIRECT TEST. If the observed
#    error is the standing lag v_target/K, then raising K must cut it
#    proportionally. The corpus contains both gains -- recovered exactly by
#    solving the control law as an identity on unsaturated, P-active rows
#    (K = |-J@omega - (du,dv)| / |p|, whose p25/p50/p75 agree to two decimals
#    on every run): K = 2.00 on five runs, K = 1.20 on three. Splitting the
#    error by the recovered gain:
#         K = 2.00 : median error 286 px (n=1931)   v/K predicts  88 px
#         K = 1.20 : median error 275 px (n=1714)   v/K predicts 147 px
#         observed ratio 0.96 (1.01 on P-active rows); mechanism needs 1.67
#    The error does not move with the parameter it is supposed to depend on,
#    and both values are 2-3x the prediction. So the substitution defect is
#    REAL -- it is in the file's own derivation, which is not a matter of
#    opinion -- but it is NOT what sets the observed error magnitude.
#    Caveat: not a controlled experiment. The two gain populations are
#    different sessions on different days. Suggestive, not conclusive -- but
#    it is a test of the mechanism's own prediction, and it fails.
#
#    NOTE FOR ANYONE READING OLD RUNS: 58% of recorded rows were taken at
#    K = 2.00, which EXCEEDS the 1/(8L) = 1.89 stability ceiling implied by
#    the measured 66 ms latency. Any statistic pooled across the corpus mixes
#    two different loops, one of them above its own stability limit. Runs
#    before 2026-09-19 pm have session.json config == null, which is why this
#    had to be recovered rather than read.
#
# WHAT WOULD MAKE IT CORRECT: subtract the ACHIEVED rate, not the commanded
# one. Either integrate the firmware ramp (VEL_ACCEL 40000 microsteps/s^2 at a
# 200 Hz tick) to estimate achieved rate at box time, or measure ego-motion
# directly from image texture. Both are post-demo work, and both need the
# fiducial recordings to validate. A scalar "delivery factor" fudge is NOT
# acceptable here -- it would mask the circularity in (2) rather than remove it.
#
# Leaving the flag in place, OFF, because the diagnosis is worth keeping and
# the logging it added (cmd.v_target_*, cmd.ff_platform_px_s, cmd.omega_at_box)
# is what makes the real fix measurable. Enabling it degrades tracking; it does
# not create a safety hazard, because the laser interlock gates firing on
# MAX_ERROR_TO_FIRE_PX independently of the feedforward.
FEEDFORWARD_SUBTRACT_PLATFORM = False  # 2026-09-20: run_124411 was a feedforward
# RUNAWAY on a static target (est.du -299 px/s of the turret's own slew fed
# forward, error 15 -> 1065 px in 1.6 s). Enabling this was checked on those
# rows: J @ omega_at_box is ANTI-parallel to est.du (cos -0.98, 46/46 rows), so
# the subtraction would double the feedforward. Stays OFF. See FEEDFORWARD_GAIN.
# 2026-09-20: multiplier on the velocity feedforward (est.du/dv). 0.0 = pure
# proportional law, which cannot run away on its own motion; cost is a standing
# lag of v_target / CONTROL_GAIN_K on a moving target. Set for the demo night.
FEEDFORWARD_GAIN = 0.0   # 2026-09-20 demo: run_142953 showed the feedforward flipping +-150-300 px/s frame to frame on a target moving <100 px/s (6-8 reversals/s, beam pulsing); pure P is smoother and cannot chase its own motion
# backwards in velocity mode (firmware iteration 14 fixed the DIR cache), not the
# feedforward. 1.0 is the configuration that held the static drone for 88 s in run_123719.

#: Decay the proportional term's scale on the SAME ramp `authority` uses,
#: instead of cliffing it to zero. DEFAULT OFF, and INDEPENDENT of the flag
#: above -- enabling both at once makes neither A/B interpretable.
#:
#: As shipped, p_scale is a hard step: 1.0 while the evidence is fresher than
#: COAST_HOLD_MS, 0.0 the instant it is not. So the proportional term does not
#: fade, it VANISHES, mid-flight, on a threshold crossing -- while `authority`
#: is computed four lines earlier and decays smoothly over COAST_DECAY_MS. Two
#: staleness responses to the same signal, one smooth and one a cliff.
#:
#: Measured, pooled over all runs, regression of de/dt on e (and confirmed
#: with a lagged instrument, so it is not the shared-noise artifact):
#:     P term ACTIVE   slope_v -0.271 /s   (error decaying)
#:     P term GATED    slope_v +1.052 /s   (error DIVERGING)
#: NOT a causal claim -- "no recent evidence" correlates with "the target was
#: moving fast", and that confound cannot be removed from these logs. The
#: argument for changing it is that a discontinuity in a control law is bad
#: practice on its own terms, and the smooth ramp already exists beside it.
#:
#: This changes only the SHAPE of the scale factor. COAST_HOLD_MS and
#: COAST_DECAY_MS are untouched, so the evidence horizon is unchanged.
#:
#: KNOW THIS BEFORE READING THE A/B: `authority` ALREADY multiplies the FINAL
#: clipped rate, so with this flag on the P term is scaled by it TWICE --
#: once here and once at the output -- and therefore decays as authority^2
#: while the feedforward decays as authority^1. That is a consequence of the
#: pre-existing output scaling, not of this flag, and it is consistent with
#: the intent stated in control.py ("coasting keeps matching the target's
#: speed, and stops arguing with a position nothing has seen"): the P term
#: SHOULD fade faster than the feedforward. But it means the effective P
#: decay is quadratic, not the linear ramp the flag name suggests. Both
#: p_scale and authority are logged per row so this is visible rather than
#: inferred.
P_SCALE_DECAY = True   # 2026-09-20: ON. run_122343 measured the binary gate on
# hardware: wide boxes arrive 150-250 ms old, so the P term flickered 1/0 and
# the command went 930 -> 0 -> 1417 -> 0 with |e| pinned at 804 px. Fade instead.

#: Ceiling on the platform term subtracted from the feedforward, px/s. At the
#: 195 deg/s mechanical top the platform can contribute ~4800 px/s at f=1400,
#: so anything past this is a bad omega or a stale Jacobian, not a slew.
FEEDFORWARD_PLATFORM_MAX_PX_S = 5000.0

#: Ignore corrections below this. Sub-pixel corrections are noise from the
#: Jacobian's own calibration error, not signal, and applying them just adds
#: jitter to the aim point.
PLATFORM_COMP_MIN_PX = 1.0

#: Refuse corrections above this. A correction this large means the command
#: changed enormously between the detection and now -- a saturation event or a
#: state transition -- and extrapolating a jump that big is how the runaway
#: behaved. Clamp rather than trust.
PLATFORM_COMP_MAX_PX = 120.0

# ==========================================================================
#   FILTER  (pixel Kalman, state [u, v, udot, vdot])
# ==========================================================================
MEAS_NOISE_PX = 3.0                     # sigma_m
# q_base chosen so sqrt(q) ~ 3 m/s^2 expressed in px/s^2: 3 * f / R at 3 m.
Q_BASE = (3.0 * NARROW_F_PX / ASSUMED_RANGE_M) ** 2
Q_MAX_MULT = 30.0
Q_SPIKE_MULT = 8.0                      # on NIS > threshold
Q_DECAY = 0.85                          # q <- q_base + decay*(q - q_base) each frame
NIS_THRESHOLD = 9.21                    # chi^2, 2 dof, 99%

#: ACCELERATION CEILING ON THE VELOCITY STATE, in m/s^2 at the target.
#:
#: A 249 g hand-held drone cannot accelerate at 23-77 m/s^2, but the filter's
#: velocity state was measured jumping >400 px/s in a single update on 19-29%
#: of updates, which is exactly that. Those jumps are not target motion: they
#: are the filter reacting to the duplicate-frame outliers above, and the
#: phantom velocity is then fed to the motors as feedforward AND used to
#: extrapolate position through detection gaps.
#:
#: Converted to pixels per update as a_max * NARROW_F_PX / range_m * dt, so it
#: tightens automatically as the target gets closer and a given angular rate
#: costs more pixels.
#:
#: This BOUNDS the correction; it does not replace it. The filter still moves
#: the velocity state as far as the measurement asks, up to what physics allows.
#:
#: WHY 12. The drone is hand-held and was being moved SLOWLY. Two independent
#: checks agree:
#:   * FORCE. The 77 m/s^2 excursions in the logs would be 19 N on a 0.249 kg
#:     drone -- about 2 kgf, a hard snap. Nobody snapped it; those are not real.
#:   * BASELINE DECAY. Measured acceleration falls 6.9 -> 4.1 -> 2.6 m/s^2 as
#:     the differencing window grows 100 -> 200 -> 400 ms. A REAL acceleration
#:     does not depend on the window you measure it over. That decay is
#:     residual measurement noise, so true typical acceleration is <= 2.6.
#: 12 sits above the p90 at every baseline, so it does not clip real motion.
#:
#: This CLAMPS ~37% of updates on the current logs. That is the clamp working,
#: not over-clamping: those updates are the filter being yanked by box-centre
#: noise, which is exactly what it exists to bound. But note the cost -- the
#: clamp limits ACCELERATION, so reaching a genuinely fast 1500 px/s takes ~6
#: updates (~0.26 s) instead of one. If real fast waves start looking damped,
#: that is this constant, and it should be raised rather than removed.
#:
#: THE UNIT CONVERSION INSIDE THIS IS OFF, AND IT ERRS SAFE. The clamp forms
#: a_px = a_max * NARROW_F_PX / range_m with NARROW_F_PX = 1400 and, in
#: practice, range_m = ASSUMED_RANGE_M = 3.0. The logs imply a true range near
#: 2.05 m and f settled at ~1328 px. So the computed 5600 px/s^2 ceiling is
#: really 5600 * 2.05/1328 = 8.6 m/s^2, not 12 -- the clamp is TIGHTER than
#: its nameplate by 1.39x, not looser. It still clears the measured target
#: acceleration (<= 2.6 m/s^2) by 3.3x, so this is a comment and not a change.
#: Fix the conversion when NARROW_F_PX and ASSUMED_RANGE_M are themselves
#: corrected, not before, or the two errors will be chased separately.
#:
#: INTERIM, not measured truth: derived from a noisy chain that cannot yet
#: characterise the target's dynamics. Re-derive from the fiducial sessions.
MAX_TARGET_ACCEL_MPS2 = 12.0
#: FLOOR on the association gate, not the whole gate. A detection this close
#: to the prediction is always accepted, whatever the filter thinks of itself.
#: Kept at 60 so the change below is strictly ADDITIVE -- it can only accept
#: boxes the old gate rejected, never reject one it accepted.
GATE_PX = 60.0

#: Hard ceiling, however uncertain the filter becomes. Without it a long coast
#: inflates P until the gate would admit anything on screen, and a
#: single-target nearest-neighbour associator then locks onto whatever clutter
#: happens to be closest. 300 px covers the measured coast drift
#: (COAST_HOLD_MS + COAST_DECAY_MS = 450 ms at a few hundred px/s) without
#: approaching the 720x1280 frame.
GATE_MAX_PX = 300.0

#: Chi-square, 2 dof, 99% -- the same value NIS_THRESHOLD uses, because it is
#: the same test: "is this innovation consistent with the filter's own claimed
#: uncertainty?" Applied at ASSOCIATION time rather than after it.
#:
#: WHY THE FIXED GATE HAD TO GO. Measured 2026-09-19 over two runs: in COAST
#: the detector saw the drone on 79-92% of frames and association rejected
#: 100% of them. The filter coasts, its prediction drifts past a gate frozen
#: at 60 px, and it then cannot accept the drone while looking straight at it.
#: The cost is not just the miss -- scoring those runs showed the filter's
#: forward prediction was 20-26 px WORSE than a naive "it is where it was last
#: seen", precisely because the naive predictor uses the boxes association
#: threw away. The filter already tracks its own uncertainty in P; this makes
#: the gate ask.
GATE_CHI2 = 9.21

#: Consecutive misses after which a confident detection RESTARTS the track
#: instead of having to agree with the prediction.
#:
#: A wider gate answers "the filter is UNSURE where the target is". It does
#: nothing for "the filter is WRONG" -- if the drone changed direction during
#: a coast, the prediction is in the wrong place and no gate width fixes a bad
#: mean, it only admits more clutter around it.
#:
#: The evidence that this is real: scored over two runs on 2026-09-19, the
#: filter's forward prediction was 20-26 px WORSE than a naive "the drone is
#: where it was last seen". A filter whose state is worse than the last raw
#: measurement should be throwing that state away, and `_begin_track` already
#: knows how.
RESEED_AFTER_MISSES = 6

#: ...but only on a detection good enough to bet the track on. The real drone
#: detects down to 0.350 (the threshold itself), so this cannot be set high
#: without refusing to recover at exactly the ranges that need it. 0.50 is
#: above the median false positive and below most real detections.
RESEED_MIN_CONF = 0.50

# Rolling shutter: the C270 reads out over ~20 ms, so a moving target's
# centroid is displaced along its motion. Subtract k * velocity per frame.
ROLLING_SHUTTER_K_S = 0.010             # ESTIMATE; calibrate by panning past a static object

# Occlusion: the hand covers part of the drone and biases the box.
AREA_DROP_FRAC = 0.30                   # >30% area loss in one frame = occlusion
AREA_RECOVER_FRAC = 0.20
# WAS 0.15 -- "the hand is below the drone, bias the aim up so the beam lands
# on the airframe rather than the fingers". Set to 0 on 2026-09-19 by the
# operator's call: it is a 5 mW demo pointer, fingers are not at risk, and the
# hand is only in the picture because the drone cannot be flown indoors for a
# school demo. It is not part of the deliverable.
#
# It was not a small offset. MEASURED on run_2026-09-19_132817 at 300-500 px
# boxes: a median 41.7 px push, against MAX_ERROR_TO_FIRE_PX of 25. With the
# bias on, the beam could not be centred on the body even in principle -- and
# "centred in the drone's body" is the entire deliverable.
#
# It also cost a session of analysis: comparing the aim point against the raw
# box centre reads the offset as tracking error, which is how "the filter is
# worse than not predicting at all" was produced. The tracker actually holds
# its intended aim point to 2.5 px.
#
# Eye safety is unaffected -- that is the face interlock's job, not this.
AIM_BIAS_UP_FRAC = 0.0

# ==========================================================================
#   TRACK STATE MACHINE
# ==========================================================================
ACQUIRE_FRAMES = 5                      # consecutive hits to enter TRACK
ACQUIRE_MISSES = 3                      # misses to fall back to SEARCH
TRACK_MISSES = 10                       # misses to enter COAST
COAST_HOLD_MS = 150                     # hold last velocity
COAST_DECAY_MS = 300                    # then decay to zero

# ==========================================================================
#   DETECTOR
# ==========================================================================
# THE FINE-TUNED DRONE MODEL, not stock yolo11n. Stock COCO has no `drone`
# class at all, so TARGET_CLASSES silently drops it and the stack runs on
# `bird`/`airplane` -- which fire often enough on a quadcopter to look like the
# detector works, while the class you actually trained is not being used.
# Override per-run with the TURRET_YOLO_WEIGHTS environment variable.
# v2, not v1. v1 reports a higher mAP50 (0.917 vs 0.870) and that comparison is
# MEANINGLESS: both runs name the same dataset/yolo_reviewed/data.yaml, but the
# contents were rebuilt between them, so v1 was scored on a smaller and easier
# val set. v2 was trained and scored on more samples across more backgrounds
# and environments, which is the thing that generalises to a demo room.
YOLO_WEIGHTS = "drone_y11n_v4_1280.pt"    # 2026-09-20: v2 boxed a painting 13/28 frames at up to 0.78 and captured the tracker; v4 boxed it 0/28 and kept the drone
# 1280, and DO NOT "fix" this to match the 640 training size.
#
# The val split says 640 is better. The val split is wrong about this machine.
# Measured on the 383-frame reviewed val set (recall, conf 0.35, IoU>0.5):
#
#       imgsz     512     640     960    1280
#       narrow  0.830   0.812   0.786   0.643
#       wide    0.819   0.852   0.846   0.758
#
# Measured on 1226 paired LIVE frames, drone hand-held at demo range
# (tools/watch_detect.py runs both sizes on the same frame, so this is a
# per-frame comparison, not two separate runs):
#
#       box px      frames   @640    @1280
#       100-140        303   0.927   0.974
#       140-180        545   0.943   0.982
#       180-240        149   0.859   0.960
#       240-400         67   0.313   0.836
#       400+           145   0.193   0.938
#       overall       1226   0.796   0.962
#
# 1280 wins in EVERY bucket live. The val set disagrees because its size
# distribution is not this one: mean narrow box 369 px there against 166 px
# live, because the captured footage is mostly close-ups under ~1 m while the
# demo is at 2-5 m. train_detector.py says it outright -- the split is TEMPORAL
# so train and val share clips, and "the number that counts is live footage".
#
# Trust the live measurement. If the model is retrained on footage at demo
# range, re-run tools/watch_detect.py before touching this again.
YOLO_IMGSZ = 1280
YOLO_CONF = 0.25                        # propose; the gate decides. 0.35 -> 0.25 with v4 (0 clutter boxes at 0.20 on 28 run frames; v4 recall is the weaker side)
YOLO_HALF = True
# The fine-tune has exactly ONE class. `bird`/`airplane` were COCO stand-ins
# for a drone class that did not exist yet; against these weights they can
# never match. Dropping them also makes a regression loud: _resolve_classes()
# tolerates SOME classes missing but raises when ALL of them are, so pointing
# YOLO_WEIGHTS back at stock COCO now fails at startup instead of quietly
# tracking birds.
TARGET_CLASSES = ("drone",)
# 20, not 10. MEASURED: the wide detector delivered a result every 160-190 ms
# against a WIDE_FALLBACK_MAX_AGE_S of 150 ms, so the wide box was already
# STALE on 21-45% of the miss rows -- the exact rows the fallback exists to
# rescue. Raising the rate fixes the staleness at its source. Do NOT raise the
# age cap instead: the cap is a parallax/latency bound, not a throughput knob,
# and widening it admits a box whose mapping is no longer trustworthy.
# The cost is visible in control.jsonl as wide.infer_ms.
WIDE_SEARCH_FPS_TRACKING = 15   # 2026-09-20: 15 not 20 for the demo night; narrow 30 Hz x 22 ms + wide 20 Hz x 22 ms would be ~1.1 s/s of predict time on one GPU. Raise to 20 once a run shows narrow box_t interval median <50 ms and image age <40 ms           # throttle the wide detector while tracking

# Run the wide FACE pass only every Nth wide iteration. 1 = every iteration
# (the old behaviour). Values below 1 are treated as 1.
#
# WHY: the wide loop was COMPUTE-BOUND far below its own throttle, so changing
# WIDE_SEARCH_FPS_TRACKING did nothing at all. MEASURED on --real-detect,
# counting wide frames written (one per loop iteration):
#     synthetic detectors : 2063 imgs / 120 s = 17.2 Hz  (58 ms period)
#     real YOLO on cuda:0 :  950 imgs / 150 s =  6.3 Hz  (158 ms period)
# against a 50 ms throttle period at 20 Hz. The drone pass is only 21.7 ms of
# that 158; the remainder is the wide FACE pass (YuNet, ~100 ms, on the CPU),
# which ran unconditionally every iteration with no throttle of its own. So the
# bottleneck was never the GPU, and no GPU throttle could fix it.
#
# WHY IT MATTERS: WIDE_FALLBACK_MAX_AGE_S is 150 ms. A 158 ms loop period means
# the wide box is ALREADY over the freshness cap the instant it is published,
# before any transport delay -- which is why it was stale on 21-45% of the miss
# rows it exists to rescue. A box cannot be fresher than the loop producing it.
#
# WHY THIS IS SAFE: wide faces are DISPLAY-ONLY and gate NOTHING. The beam's
# face interlock reads app.face_slot, fed exclusively from the NARROW detector
# (app.py, "narrow (SAFETY)"), because there is no wide->narrow mapping on the
# interlock path. VERIFIED by grep: no interlock condition reads a wide face.
# The narrow face interlock is untouched by this constant.
WIDE_FACE_EVERY_N = 4
WIDE_SEARCH_FPS_SEARCHING = 30

# YuNet, not a person detector: a person detector fires on the hand holding the
# drone and the laser would never fire.
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_CONF = 0.6

# ==========================================================================
#   SAFETY  -- see AGENT_HANDOFF 1.6. These are not tuning knobs.
# ==========================================================================
# The laser may be on ONLY when ALL of these hold. Any one failing -> off,
# before anything else happens.
LASER_ENABLED = True                    # master arm -- enabled by the operator 2026-09-18
FACE_INHIBIT_MARGIN_PX = 120            # no face within this of the beam path
MAX_ERROR_TO_FIRE_PX = 60   # 2026-09-20 operator: 25 -> 60 for the moving-drone demo (~2.5 deg, inside a ~200 px airframe)
LASER_MAX_ON_MS = 2000
SETTLED_RATE_DEG_S = 40.0               # 2026-09-20 operator: 5 -> 40 so the beam can stay on while the drone moves
# -- shape gate (2026-09-20, after run_145109 frame 006835) ------------------
# The beam fired on the carrier's HAND: the drone was edge-on behind the fingers,
# YOLO boxed the hand at conf 0.56 and every gate passed. That box was 202x91 px,
# aspect 2.22, against a median of 1.11 over all 638 firing rows of the run and a
# widest legitimate box of 1.52 -- and it was the SMALLEST box of the 638 by area.
# A quadcopter seen by this camera is roughly square; a hand side-on is not. The
# interlock refuses to fire on a box outside this aspect window (w/h). Cost on
# the two armed runs of 2026-09-20: 34/2149 and 6/1844 TRACK rows, 1 firing row
# (the hand). This does NOT protect a hand inside a drone-shaped box; keep the
# grip outside the airframe (hold a landing leg) until a hand detector exists.
FIRE_BOX_ASPECT_MIN = 0.30   # 0.45 -> 0.30: the drone edge-on or folded is 0.39-0.48 (real airframe, frames checked); the hand signature is WIDE, so the floor is only a sanity bound
FIRE_BOX_ASPECT_MAX = 1.6

# -- fail-closed permission ------------------------------------------------
# The beam is granted by POSITIVE evidence, never by the absence of a veto.
# "No face detected" is indistinguishable from "the face pass did not run", so
# clearance alone can never authorise light. Permission comes from a fresh drone
# box lying under the beam path; the head pass only ever takes it away.
#
# A drone box older than this is a statement about the past, not about now. At
# 30 fps this is ~4 frames, enough to ride out a dropped frame without letting a
# tracker coast the beam across a room on a stale measurement.
DRONE_BOX_MAX_AGE_S = 0.150
# Shrink the box per side before testing containment, as a fraction of its own
# width/height so it scales with range. The 283 mm airframe is ~132 px at 3 m,
# so 0.15 leaves ~92 px of permitted region there -- the body, not the rim,
# where a box that is a few px loose would otherwise put the beam past the edge.
DRONE_CONTAINMENT_INSET_FRAC = 0.15
# The head pass must have reported within this for its answer to describe the
# room the beam is in. Matches app.FACE_MAX_AGE_S, which guards the same thing
# one layer out.
HEAD_REPORT_MAX_AGE_S = 0.200

# Dot detection (opportunistic -- never gate anything on it).
# The chroma test is weakened by the missing IR-cut filter: NIR passes every
# Bayer filter roughly equally, so lit surfaces desaturate. Retune on real
# frames and use brightness as well as chroma.
DOT_WINDOW_PX = 60
DOT_CHROMA_TAU = 60                     # ESTIMATE, assumes an IR-cut sensor
DOT_AREA_PX = (3, 40)
DOT_ROUNDNESS_MIN = 0.6
LASER_PULSE_HZ = 15                     # half the 30 fps capture: present/absent alternating

# ==========================================================================
#   DISPLAY
# ==========================================================================
DISPLAY_SIZE = (640, 360)               # draw on a copy, never the inference buffer
DISPLAY_FPS = 30

# 2026-09-20: before the first non-zero `vel` of a link's life, send one
# negative and one positive tick-sized rate so the firmware writes both DIR
# pins from a standstill. Firmware <= iteration 13 cached the direction and
# skipped the write when the sign matched the cache, but position moves
# (homing) changed the pin without the cache: motor A ran BACKWARDS for all of
# run_2026-09-20_124411 and the first lunge of run_122343 -- the 'start
# tracking slews down-right' failure. Iteration 14 fixes the firmware; this
# stays on as the guard for an unflashed board. Costs two round trips once.
VEL_DIR_RESYNC_ON_ENTRY = True
