# `gui_fused_finish.patch` — the rest of the fused-overlay work

Apply with:

    git apply -p1 staged/gui_fused_finish.patch

Against `turret_host/gui.py` as of 2026-09-20 18:24 (the file the lead
relaunched on). Verified: applies cleanly, +87/−31, result compiles, and an
AST audit finds no `self.` attribute read that is never assigned — which is
the specific check that would have caught the `_w2n_h` breakage.

**Nothing here touches `config.py`.** `FUSED_VIEW_DEFAULT` stays as the lead
set it (`False`). Flipping it is a separate one-line decision for whenever
the panel gets its shakedown.

## What is in it

**1. Frame-skew readout on the fused pane.** Both frame slots already carry a
`types.Frame` with a capture time and `_image_of` was dropping it, so the
pane composited two different moments with no way to know. Now `_draw_panes`
computes the gap and `_render_fused` prints it, in amber past
`config.FUSED_MAX_FRAME_SKEW_MS` (40).

Measured on run_2026-09-20_151635: median 28.4 ms, p90 69.5, max 181, only
41.5% inside 25 ms. At 30 °/s that median displaces the two layers ~21 narrow
px — bigger than the depth-parallax term, and it looks exactly like a
registration error. This does not fix it; it stops it being misdiagnosed.

**2. Overlay metrics scale with the pane.** Fonts, line widths, marker sizes
and label offsets were all absolute pixels chosen for a 640 px pane. On an
800 px+ pane they would have become hairlines and specks — the standard
failure of "just make the window bigger". All now go through `_sz()` / `_lw()`.
Line widths deliberately grow *sub*-linearly (`_lw`), because a 4 px E-STOP
border scaled linearly to a 1400 px pane is a 9 px slab.

**3. Pose inset moved after the blit resize.** As offered by the 3D-model
session. The inset is a fixed-size HUD at `POSE_OVERLAY_SIZE`; pasting before
a resize would resample it with the video. With the raster now sized to the
pane there is usually no resize at all, so this mostly guarantees it stays
1:1.

**4. Telemetry pitch/yaw no longer frozen.** Was reading `link.pitch_deg`,
which `link.command()` refuses to refresh while servoing — so it held the
last blocking command's value for the whole of a track, i.e. wrong exactly
when someone is watching. Now reads `SystemStatus.pose` (the dead-reckoned
`ReckonedPose` added by the 3D-model session) and falls back to the link
value. Tagged `DR` or `link` on the readout so the two cannot be confused,
because the DR value drifts by construction.

## What to watch when it first runs

The Tk path is still unexercised — see the caveat below — but if something
does go wrong, the likely spots in rough order:

- **Pane sizing.** `_fit_panes_to_screen` measures the chrome and gives the
  panes the rest. If the window comes up too tall for the screen or the
  E-STOP is off the bottom, that measurement is the culprit; the log line
  `panes WxH ... chrome measured at N px` says what it decided. Falls back to
  the 640×360 floor and logs a warning when the screen genuinely cannot fit
  more.
- **DPI.** `_claim_dpi_awareness()` runs before `tk.Tk()` and is best-effort;
  a machine that refuses gets 1.0 and the old behaviour. If text looks tiny,
  the `tk scaling` compensation is the thing to look at.
- **The fused pane specifically.** It refuses rather than guesses on any wide
  frame size it cannot account for, and says so on the pane.

## The caveat that has not changed

**The Tk path has never been run.** The geometry under it is well verified
offline — 0.50 vs 3.49 narrow px rms, held-out 0.46 vs 12.28, and 3518 paired
tiles on real recorded frames — but that is the *functions*, not the
*wiring*, and on this project that distinction has cost four bugs in a day.
Treat the first launch as a test, not as a working panel.

Revert is `git checkout turret_host/gui.py`.

## Related, not in this patch

- `tools/fit_wide_narrow_registration.py` — refits the registration offline.
- `turret_host/registration.py` — the runtime model; `python -m
  turret_host.registration` self-tests the geometry with no camera.
- `tools/preview_fused.py` — renders the overlay old-vs-new from a recorded
  run, so the improvement can be looked at without the rig.
- `tools/skew_study.py` — the skew characterisation.
