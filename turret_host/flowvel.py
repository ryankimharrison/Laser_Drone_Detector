"""Target image velocity from optical flow on the box patch.

Feeds the FEEDFORWARD term only. It is not a measurement into the Kalman
filter and it must not become one without a separate decision -- the filter is
the estimator every other item depends on.

WHY THIS EXISTS
---------------
`FEEDFORWARD_GAIN` is 0.0 because the feedforward flipped sign at +/-150-300
px/s frame to frame. The velocity feeding it came from differencing box
CENTRES, and a box centre moves when the detector's idea of the extent shifts
by a few pixels with nothing moving at all. Measured over both 2026-09-20 runs
(tools/flow_velocity_study.py), predicting the box centre one loop-latency
ahead:

    66 ms horizon, median error px   naive  boxdiff  filter  flow
      run_144709 (fast carrier)       22.1     11.2    10.3    6.0
      run_145109 (slow carrier)        7.4      7.3     5.8    4.1

and the frame-to-frame velocity JUMP -- the quantity that flipped the sign --
falls from 190.8/130.7 px/s (boxdiff) to 50.8/26.2 (flow). Box differencing is
beaten by NAIVE at the 100 and 200 ms horizons on the slow run, so it is not
merely noisy; it is worse than assuming the target is stationary.

THE COST WAS MEASURED BEFORE THIS WAS WRITTEN
----------------------------------------------
The obvious implementation does not fit. `goodFeaturesToTrack` with a mask
still scans the whole image, so full-frame LK plus the backward check costs a
MEASURED 27.5 ms median -- most of a 33 ms frame. Cropping to the box plus a
35% margin first costs 2.78 ms median, 3.56 p90, and returns the same answer:
agreement with the full-frame version over 220 frames is a median 0.0003 px,
p90 0.008 px. So the study above still describes this code.

AVAILABILITY IS THE REAL LIMIT, AND IT IS WHY THE FALLBACK IS PURE P
---------------------------------------------------------------------
Flow answers on 96% of consecutive frame pairs when the carrier is slow and
85% when fast; the misses are motion blur eating the corners, which is exactly
when the velocity matters most. The fallback is therefore NOT the filter and
NOT box differencing: the filter's own frame-to-frame jump is still 85/250
px/s median/p90 on the fast run, and a sign flip on the blurred frames is the
failure this whole change exists to remove. Pure P for a few frames costs lag,
which is a bounded error; a wrong-signed feedforward is not bounded.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from turret_host import config

#: LK parameters. winSize 21 and 3 pyramid levels track the 150-250 px airframe
#: at the speeds these runs reached; smaller windows lose lock on motion blur.
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class FlowVelocity:
    """One instance per narrow camera. Not thread-safe; call from one thread.

    `update()` is called once per narrow frame with the RAW image -- never the
    annotated one. The overlay moves with the ESTIMATE, so tracking it would
    feed the loop its own output.
    """

    def __init__(self):
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._last_v: Optional[Tuple[float, float]] = None
        self._last_v_t: Optional[float] = None
        #: Diagnostics for the recorder, read after every update().
        self.source = "none"          # "flow" | "held" | "none"
        self.points = 0
        self.ms = 0.0
        self.rejects = 0

    # ------------------------------------------------------------------
    def update(self, image, box: Optional[Sequence[float]], t: float):
        """Return (vu, vv) px/s for the feedforward, and set `source`.

        `image` is the raw BGR narrow frame; `box` is the detector's box for
        this frame in narrow pixels, or None. Returns (0.0, 0.0) with
        source="none" whenever there is nothing trustworthy to offer -- never
        a guess, and never the previous value held flat.
        """
        t0 = time.perf_counter()
        gray = None
        v = None
        if image is not None:
            # ONE conversion per frame, reused as this frame's current and the
            # next frame's previous. Converting the crop instead would need a
            # second conversion next frame, because the box has moved by then.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if (self._prev_gray is not None and self._prev_t is not None
                    and box is not None):
                dt = t - self._prev_t
                if 0.0 < dt <= config.FLOW_MAX_DT_S:
                    d = self._flow(self._prev_gray, gray, box)
                    if d is not None:
                        v = (d[0] / dt, d[1] / dt)
                        self.points = d[2]
                    else:
                        self.rejects += 1
                        self.points = 0
        if gray is not None:
            self._prev_gray, self._prev_t = gray, t

        if v is not None:
            self._last_v, self._last_v_t = v, t
            self.source = "flow"
            out = v
        else:
            out = self._decayed(t)
        self.ms = (time.perf_counter() - t0) * 1000.0
        return out

    # ------------------------------------------------------------------
    def _decayed(self, t: float):
        """The held tier: the last flow velocity, ramped linearly to zero.

        Linear rather than flat, deliberately. Holding a velocity flat and then
        dropping it to zero puts a step into the feedforward at the moment the
        hold expires, and a step in the feedforward is the same shape as the
        sign flip this change removes. Ramping means the worst the fallback can
        do is fade.
        """
        if self._last_v is None or self._last_v_t is None:
            self.source = "none"
            self.points = 0
            return (0.0, 0.0)
        age = t - self._last_v_t
        if not 0.0 <= age <= config.FLOW_HOLD_S:
            self.source = "none"
            self.points = 0
            self._last_v = None
            return (0.0, 0.0)
        k = 1.0 - age / config.FLOW_HOLD_S
        self.source = "held"
        self.points = 0
        return (self._last_v[0] * k, self._last_v[1] * k)

    # ------------------------------------------------------------------
    @staticmethod
    def _flow(prev_gray, gray, box):
        """Median displacement of features in the box, or None if degenerate.

        Three guards, and a frame failing any of them returns None rather than
        a confident answer built from nothing: a blurred airframe against a
        flat wall gives few corners and LK will still return numbers.
        """
        h, w = gray.shape[:2]
        bw = float(box[2]) - float(box[0])
        bh = float(box[3]) - float(box[1])
        if bw <= 0 or bh <= 0:
            return None
        px, py = bw * config.FLOW_ROI_PAD, bh * config.FLOW_ROI_PAD
        X1 = max(0, int(box[0] - px))
        Y1 = max(0, int(box[1] - py))
        X2 = min(w, int(box[2] + px))
        Y2 = min(h, int(box[3] + py))
        if X2 - X1 < 16 or Y2 - Y1 < 16:
            return None
        pc = prev_gray[Y1:Y2, X1:X2]
        cc = gray[Y1:Y2, X1:X2]
        # Features from INSIDE the box only; the pad exists so the search has
        # somewhere to follow them to, not so the wall can supply corners.
        mask = np.zeros(pc.shape, np.uint8)
        mask[max(0, int(box[1]) - Y1):int(box[3]) - Y1,
             max(0, int(box[0]) - X1):int(box[2]) - X1] = 255
        p0 = cv2.goodFeaturesToTrack(pc, maxCorners=config.FLOW_MAX_CORNERS,
                                     qualityLevel=0.01, minDistance=4,
                                     mask=mask, blockSize=7)
        if p0 is None or len(p0) < config.FLOW_MIN_POINTS:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pc, cc, p0, None, **_LK)
        if p1 is None:
            return None
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cc, pc, p1, None, **_LK)
        if p0r is None:
            return None
        good = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1)
        if int(good.sum()) < config.FLOW_MIN_POINTS:
            return None
        a = p0.reshape(-1, 2)[good]
        b = p1.reshape(-1, 2)[good]
        r = p0r.reshape(-1, 2)[good]
        keep = np.linalg.norm(a - r, axis=1) < config.FLOW_FB_MAX_PX
        if int(keep.sum()) < config.FLOW_MIN_POINTS:
            return None
        d = b[keep] - a[keep]
        med = np.median(d, axis=0)
        # Coherence: do the survivors agree, or is this a spray of noise that
        # happens to average to something?
        #
        # RELATIVE TO THE DISPLACEMENT, not an absolute px budget. The spread
        # between points on a real target grows with how far the target moved
        # between the two frames -- parallax across the airframe's depth, the
        # box rotating in image, and the props, which genuinely do not move
        # with the body. A fixed px cut therefore tightens as the platform
        # speeds up, and rejected the fast rows this measurement exists to
        # serve. See FLOW_COHERENCE_FRAC for the measurement and the check
        # that the recovered rows are usable rather than merely more numerous.
        mad = float(np.median(np.linalg.norm(d - med, axis=1)))
        disp = float(np.hypot(med[0], med[1]))
        # getattr with a default, so applying flowvel.py without config.py
        # degrades to the old absolute guard instead of raising inside the
        # detect thread on the first frame.
        frac = getattr(config, "FLOW_COHERENCE_FRAC", 0.0)
        limit = max(config.FLOW_COHERENCE_PX, frac * disp)
        if mad > limit:
            return None
        return float(med[0]), float(med[1]), int(keep.sum())

    # ------------------------------------------------------------------
    def reset(self):
        """Drop history. Call when the track is lost, so a stale velocity
        cannot survive across a re-acquisition."""
        self._prev_gray = None
        self._prev_t = None
        self._last_v = None
        self._last_v_t = None
        self.source = "none"
        self.points = 0
