"""Bind pixel-space measurements to the rig that produced them."""
from turret_host import config


class CalibrationMismatchError(RuntimeError):
    pass


def camera_context(roles=("narrow", "wide")):
    return {role: config.CAMERA_PROFILES[role].geometry() for role in roles}


def stamp(payload, roles=("narrow",)):
    for role in roles:
        p = config.CAMERA_PROFILES[role]
        if p.vid_pid is None or p.lens_id == "unmeasured" or p.mount_id == "unverified":
            raise CalibrationMismatchError(
                f"Commission {role} USB identity, lens_id and mount_id before saving calibration; "
                "see CAMERA_UPGRADE.md")
    return dict(payload, camera_context=camera_context(roles))


def validate(payload, roles=("narrow",), source="calibration"):
    expected = camera_context(roles)
    recorded = payload.get("camera_context")
    if recorded is None:
        # Only the explicitly selected original rig may use its old artifacts.
        legacy = all(config.CAMERA_PROFILES[r].name ==
                     ("c270" if r == "narrow" else "ov2710") for r in roles)
        if legacy:
            from turret_host.camera_profile import C270, WIDE
            original = {"narrow": C270.geometry(), "wide": WIDE.geometry()}
            legacy = all(expected[r] == original[r] for r in roles)
        if legacy:
            return
    elif isinstance(recorded, dict) and all(recorded.get(r) == expected[r] for r in roles):
        return
    raise CalibrationMismatchError(
        f"{source}: camera/lens/mode/mount does not match this calibration "
        "(or identity metadata is missing). Recalibrate this rig; do not relabel an old fit.")


def roles_for(name):
    if name.startswith("intrinsics_"):
        return (name.removeprefix("intrinsics_"),)
    if name in ("jacobian", "goal_pixel", "focal_narrow"):
        return ("narrow",)
    if name in ("wide_to_narrow", "wide_narrow_registration", "wide_narrow_homography"):
        return ("narrow", "wide")
    return ()  # clock and other non-camera calibration stay independent
