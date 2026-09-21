"""Full assm.step  ->  turret_host/assets/wrist_mesh.npz

A minimal, animatable version of the wrist for the live 3D view: structural
bodies only, split into the rigid groups the differential actually moves, with
the joint axes MEASURED off the CAD rather than typed in from a document.

Run it once (or whenever the CAD changes):

    .venv/Scripts/python.exe tools/export_wrist_mesh.py "C:/path/Full assm.step"

What comes out is small enough to load instantly and to rasterise in numpy at
display rate -- about 13k triangles total, down from 409k in the STEP.

Four things this file decides, all of which matter downstream:

**What is dropped.** Bearings, screws, nuts, washers and the whole PCB subtree.
They are half the triangle budget, they are invisible inside the assembly, and
a rolling-element bearing tessellates into thousands of faces that read as
noise at 300 px. `Bearing motor joiner` is a printed structural bracket and is
KEPT -- match the McMaster part numbers, never the word "bearing".

**Which body each part belongs to.** A differential has no parent/child chain
that a scene graph can express: the output gear's axis is carried BY the
carrier, and the two input pulleys spin about the same axis as the carrier but
at their own rates. So the grouping is explicit (`RULES`), by subassembly, and
the renderer composes the transforms itself.

**How it is reduced.** Weld, then decimate, then repair the winding -- in that
order, and the order is the whole trick. See `_weld` and `_cluster`; getting it
wrong turns the frames into floating shards at any budget.

**Where the axes are.** Fitted from the parts that are surfaces of revolution
about them -- the 6808-2RS bearing bores for pitch, the output miter gear and
the differential's own bearings for yaw. For a body of revolution the axis is
the eigenvector of LEAST variance, which is what `_axis_of` returns. Both came
out exact to four decimals (+Z and +Y, intersecting at y = 28.26 mm), so the
model is stored in a frame centred on that intersection and the renderer's
rotations are about the origin.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "turret_host" / "assets" / "wrist_mesh.npz"
DEFAULT_STEP = Path(r"C:\Users\ryank\Downloads\Full assm.step")

# Tessellation fineness handed to OpenCascade, in mm of chord deviation. 1.0 is
# already past the knee: the triangle count is dominated by gear teeth, which
# are an angular feature, so asking for 2.5 mm saves only 10 %. Decimation
# below does the real work.
TOL_LINEAR = 1.0
TOL_ANGULAR = 0.6

# ---------------------------------------------------------------------------
#   what to throw away
# ---------------------------------------------------------------------------
# Substring match, case-folded, against the node name. These are McMaster and
# footprint part numbers; they are precise on purpose. A generic "bearing" or
# "screw" match would also eat `Bearing motor joiner`.
DROP_SUBSTRINGS = (
    "4668k271",              # 440C ball bearing (the big 6808 bore)
    "5972k81",               # ball bearing, differential + input gears
    "6808-2rs",              # bearing subassembly wrapper
    "socket head screw",     # 93705A838, 90042A304
    "locknut",               # 93625A115, 90576A104
    "washer",                # 95211A150
)
# The GY-85 is KEPT, unlike every other PCB. It is the part the pose overlay's
# solid model actually reads, so the one board worth its triangles is the one
# the display is about. It arrives as two solids -- a 1.5 mm substrate and its
# component cluster -- which is why the blue silkscreen can be a real body here
# rather than the synthetic surface patch the brief's viewer had to fake.
# Whole subtrees to skip, by node name. The PCB is 75 parts and 80k triangles
# of KiCad footprints: caps, resistors, JST shrouds, a Pico.
DROP_SUBTREES = ("PCB:1", "Full PCB:1")

# ---------------------------------------------------------------------------
#   which rigid body each part belongs to
# ---------------------------------------------------------------------------
# Checked in order; first prefix match wins, so the specific `Input Gear A`
# entries must precede the `A frame` catch-all.
#
#   base      bolted to the world
#   input_a   60T pulley + bevel driven by motor A, spins about the pitch axis
#   input_b   likewise for motor B
#   pulley_a  26T pulley on motor A's shaft (spins 2.3077x faster)
#   pulley_b  likewise
#   carrier   the differential case -- this is what PITCH rotates
#   head      output gear + payload -- pitch, then yaw on top
#
# The static group and the payload are split further than the kinematics needs,
# purely so they can be COLOURED separately: one flat grey machine is unreadable
# at 260 px, and the parts a viewer needs to find -- the orange optical face the
# beam leaves through, the blue IMU the solid model is measured from -- are
# exactly the ones that vanish into it. Bodies sharing a rotation cost nothing
# extra to animate; the renderer maps several names onto one matrix.
RULES = (
    # ---- payload: pitch, then yaw on top ---------------------------
    ("Payload (1):1/GY85",                  "head_imu"),   # split -> imu + silk
    ("Payload (1):1/Back end and wiring",   "head_back"),
    # The optical faceplate, by name. A geometric search for it -- largest
    # beam-perpendicular plane ahead of the head centroid -- picked the
    # PLATE'S BACK FACE, because the back is the larger of the two (3893 mm2
    # against 3360) exactly as the brief warns: the front is interrupted by
    # the laser bore and both camera apertures. The CAD already separates the
    # part, so take it whole and skip the search.
    ("Payload (1):1/sterolaserview:1/logi c270:1/Camera cover (1):1/Top shell",
                                            "head_face"),
    ("Payload (1):1/sterolaserview",        "head_rest"),
    ("Output gear",                         "head_gear"),
    ("Payload",                             "head_rest"),  # anything else on the payload
    # ---- the differential case: pitch only -------------------------
    ("Differential Case:1",                 "carrier"),
    # ---- input bevels: pitch +/- yaw -------------------------------
    ("A frame:1/Input Gear A:1",            "input_a"),
    ("B frame:1/Input Gear A(Mirror):1",    "input_b"),
    # ---- motor pulleys: N x the input rate -------------------------
    ("A frame:1/=>",                        "pulley_a"),
    ("B frame:1/=>",                        "pulley_b"),
    ("A frame:1/GT2 pully",                 "pulley_a"),
    ("B frame:1/GT2 pully",                 "pulley_b"),
    ("3764N109",                            "pulley_b"),   # loose instance, z < 0
    # ---- static. Specific before the frame catch-alls --------------
    ("A frame:1/Nema 17",                   "motor"),
    ("B frame:1/Nema 17",                   "motor"),
    ("A frame:1/Endcap",                    "endcap"),
    ("B frame:1/Endcap",                    "endcap"),
    ("And and b frame joiner:1",            "base_plate"),
    ("A frame:1",                           "frame"),
    ("B frame:1",                           "frame"),
)

# Triangle budget per body, split across that body's parts in proportion to
# their raw counts. The head and base carry the silhouette so they get the
# most; the pulleys are 20 mm discs that nobody looks at.
BUDGET = {
    "base_plate": 1600,
    "frame":      5000,   # the two side plates carry most of the silhouette
    "motor":      2800,
    "endcap":     1800,
    "carrier":    1400,
    "head_gear":  2600,
    "head_back":  1400,
    "head_imu":   2000,
    "head_silk":   700,   # a flat 1.5 mm slab, but it shreds if starved
    "head_face":  1400,   # the optical plate: flat, needs its outline only
    "head_rest":  4200,
    "input_a":    2200,
    "input_b":    2200,
    "pulley_a":    520,
    "pulley_b":    520,
}
# No part drops below this, however small its share. A bracket reduced to 30
# triangles stops being recognisable and starts being a shard.
MIN_PART_TRIS = 150

# A second, much cruder copy of every body, used only to cast the ground
# shadow. The shadow is rasterised at quarter resolution and then blurred, so
# it cannot resolve detail the display mesh has -- paying full price for it was
# costing more than the shadow is worth.
SHADOW_BUDGET = 340

# Body colours: white structure, orange on every face that means something
# optically (the end caps, the rear plate, the face the beam leaves through),
# dark metal for the motors, and the GY-85 in board blue and IC black.
#
# Stored in LINEAR light, because the renderer finishes with a sqrt as a cheap
# gamma -- so each entry is (hex / 255) ** 2 and `_srgb` does that conversion,
# keeping the table readable as the hex values it was specified in.
def _srgb(hex_rgb: int) -> tuple:
    return tuple(((hex_rgb >> s & 0xFF) / 255.0) ** 2 for s in (16, 8, 0))


COLORS = {
    "base_plate": _srgb(0xEEF1F6),
    "frame":      _srgb(0xEEF1F6),
    "motor":      _srgb(0x6E757F),
    "endcap":     _srgb(0xD98A3A),
    "carrier":    _srgb(0x454B54),
    "head_gear":  _srgb(0x8A93A3),
    "head_back":  _srgb(0xD98A3A),
    "head_silk":  _srgb(0x2F6FD0),
    "head_imu":   _srgb(0x15181D),
    "head_face":  _srgb(0xD98A3A),
    "head_rest":  _srgb(0xEEF1F6),
    "input_a":    _srgb(0x454B54),
    "input_b":    _srgb(0x454B54),
    "pulley_a":   _srgb(0x454B54),
    "pulley_b":   _srgb(0x454B54),
}


# ---------------------------------------------------------------------------
#   scene walking
# ---------------------------------------------------------------------------
def _child_map(graph):
    import collections
    kids = collections.defaultdict(list)
    for a, b in graph.transforms.edge_data.keys():
        kids[a].append(b)
    return kids


def _walk(scene, kids, node, path, parts):
    """Depth-first, accumulating '/'-joined paths for parts that have geometry."""
    if any(node.startswith(d) for d in DROP_SUBTREES):
        return
    here = f"{path}/{node}" if path else node

    geo = scene.graph.transforms.node_data.get(node, {}).get("geometry")
    if geo is not None and geo in scene.geometry:
        # Match the whole path, not the leaf: the GY85's geometry lives in a
        # child called GY87_10DOF_IMU_PCB, and a leaf-only test would keep it.
        if not any(s in here.lower() for s in DROP_SUBSTRINGS):
            import trimesh
            mesh = scene.geometry[geo]
            T = scene.graph.get(node)[0]
            parts.append((here, trimesh.transform_points(mesh.vertices, T) * 1000.0,
                          np.asarray(mesh.faces, dtype=np.int64)))
    for c in sorted(set(kids.get(node, []))):
        _walk(scene, kids, c, here, parts)


def _body_of(path: str) -> str | None:
    """Map an assembly path to a rigid body, ignoring the scene/root nodes."""
    trimmed = path
    for root in ("world/", "Full assm/"):
        if trimmed.startswith(root):
            trimmed = trimmed[len(root):]
    for prefix, body in RULES:
        head = prefix.split("/")
        seg = trimmed.split("/")
        if len(seg) >= len(head) and all(seg[i].startswith(head[i]) for i in range(len(head))):
            return body
    return None


# ---------------------------------------------------------------------------
#   axis fitting
# ---------------------------------------------------------------------------
def _axis_of(parts, predicate):
    """Axis + centroid of the parts matching `predicate`.

    A surface of revolution spreads least along its own axis, so the axis is
    the eigenvector of the smallest eigenvalue of the vertex covariance.
    """
    v = [p[1] for p in parts if predicate(p[0])]
    if not v:
        return None, None
    v = np.vstack(v)
    c = v.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((v - c).T))
    ax = V[:, 0]
    if ax[int(np.argmax(np.abs(ax)))] < 0:
        ax = -ax
    return c, ax


def _weld(v, f):
    """Merge coincident vertices. MUST happen before decimating.

    OpenCascade tessellates each B-rep face independently, so the mesh arrives
    as a soup of disconnected patches that happen to share edges geometrically
    but not topologically. A quadric simplifier cannot collapse an edge it
    cannot see, so it shreds each patch on its own: at 50 % reduction the
    frames came out as floating shards while the same reduction on a welded
    mesh is visually indistinguishable from the original.
    """
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.merge_vertices()
        return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    except Exception as exc:                                   # pragma: no cover
        print(f"    ! weld failed ({exc})")
        return v, f


def _cluster(v, f, cell):
    """Vertex clustering: snap to a grid, merge, drop collapsed triangles.

    Cruder than quadric decimation and it ignores topology completely -- which
    is exactly why it is here. A GT2 pulley's teeth are ~40 sharp features
    around a 20 mm disc, and every edge collapse that would remove one also
    flips a normal, so the quadric simplifier refuses and stalls at 50 %
    whatever budget it is given. Clustering merges the teeth into the rim.
    """
    key = np.floor(v / cell).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    nv = np.zeros((len(counts), 3), np.float64)
    np.add.at(nv, inv, v)
    nv /= counts[:, None]
    nf = inv[f]
    ok = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[ok]
    # DEDUPE. Snapping to a grid folds distinct triangles onto the same vertex
    # triple, and a repeated triangle makes every one of its edges look
    # non-manifold: the 60T input pulley came out of here with 312 such edges
    # from a source that had none. Measured with tools/../edges audit.
    if len(nf):
        _, keep = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)
        nf = nf[np.sort(keep)]
    return nv, nf


def _decimate(v, f, target):
    if len(f) <= target:
        return v.astype(np.float32), f.astype(np.int32)
    try:
        import fast_simplification
        # target_count, not target_reduction: on a welded gear the reduction
        # form stops early at its own quality threshold and hands back five
        # times the requested budget.
        nv, nf = fast_simplification.simplify(v.astype(np.float64),
                                              f.astype(np.int32),
                                              target_count=int(target), agg=8.0)
        if len(nf) > target * 1.5:
            # Quadric decimation stalled (see `_cluster`). Binary-search a
            # clustering cell size for the largest mesh that still fits the
            # budget. Taking a result already at or under target means there
            # is nothing left to hand back to the simplifier.
            span = float(np.linalg.norm(v.max(0) - v.min(0)))
            lo, hi = span * 1e-4, span * 0.5
            best = None
            for _ in range(22):
                cell = math.sqrt(lo * hi)
                cv, cf = _cluster(v, f, cell)
                if len(cf) > target:
                    lo = cell
                else:
                    hi = cell
                    if best is None or len(cf) > len(best[1]):
                        best = (cv, cf)
            if best is not None and len(best[1]) < len(nf):
                nv, nf = best
        return np.asarray(nv, np.float32), np.asarray(nf, np.int32)
    except Exception as exc:                                   # pragma: no cover
        print(f"    ! decimation unavailable ({exc}); keeping {len(f)} tris")
        return v.astype(np.float32), f.astype(np.int32)


def _repair(v, f):
    """Agree on a winding across each connected component, outward-facing.

    Cosmetic rather than structural -- the renderer shades two-sided and does
    not cull, because decimation leaves these solids open. Consistent normals
    still matter for the lighting: neighbouring triangles that disagree shade
    to different greys and the part reads as noise.
    """
    try:
        import trimesh
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        trimesh.repair.fix_normals(m)
        return (np.asarray(m.vertices, np.float32),
                np.asarray(m.faces, np.int32))
    except Exception as exc:                                   # pragma: no cover
        print(f"    ! winding repair failed ({exc})")
        return v.astype(np.float32), f.astype(np.int32)


def _boresight(parts, origin):
    """Direction the laser and both cameras look along, or None.

    The optics plate is flat, so its least-spread direction is the plate
    normal. The normal has no inherent sign; resolve it against the wiring
    bracket, which is bolted to the BACK of the plate, so forward is the way
    that points away from it. (AGENT_HANDOFF calls the boresight +x; in this
    assembly's frame the optics look down -x, with the bracket occupying
    x = +6..+52 behind them.)
    """
    plate = [v - origin for p, v, _ in parts if "sterolaserview" in p]
    if not plate:
        return None
    pv = np.vstack(plate)
    c = pv.mean(0)
    _w, V = np.linalg.eigh(np.cov((pv - c).T))
    bore = V[:, 0]
    back = [v - origin for p, v, _ in parts if "Back end and wiring" in p]
    if back and float((np.vstack(back).mean(0) - c) @ bore) > 0:
        bore = -bore
    return bore


def _muzzle(parts, origin, bore):
    """Where the beam leaves: the front face of the optics plate.

    Centroid of the vertices furthest along the boresight, so the beam starts
    at the glass rather than at the wrist centre.
    """
    pv = np.vstack([v - origin for p, v, _ in parts if "sterolaserview" in p])
    d = pv @ bore
    return pv[d >= np.quantile(d, 0.98)].mean(0)


def _split_silk(group):
    """GY-85 parts -> (components, substrate).

    The board normal is the assembly +Y (CAD, to 0.3 deg), so the bare
    substrate is simply the part with the smallest Y extent -- 1.5 mm against
    the 11.5 mm the component cluster stands off the board. Picked by geometry
    rather than by node name because the exporter's names carry a content hash
    that changes on every re-tessellation.
    """
    if len(group) < 2:
        return group, []
    thin = min(group, key=lambda vf: float(vf[0][:, 1].max() - vf[0][:, 1].min()))
    return [p for p in group if p is not thin], [thin]


def main(argv):
    step = Path(argv[1]) if len(argv) > 1 else DEFAULT_STEP
    if not step.exists():
        raise SystemExit(f"STEP not found: {step}")

    import cascadio
    import trimesh

    tmp = OUT.parent / "_wrist_tmp.glb"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"tessellating {step.name} ...")
    cascadio.step_to_glb(str(step), str(tmp), tol_linear=TOL_LINEAR, tol_angular=TOL_ANGULAR)
    scene = trimesh.load(tmp, process=False)
    print(f"  {time.time() - t0:.1f}s, {len(scene.geometry)} solids, "
          f"{sum(len(m.faces) for m in scene.geometry.values())} triangles")

    parts: list = []
    _walk(scene, _child_map(scene.graph), scene.graph.base_frame, "", parts)
    print(f"  {len(parts)} parts survive the hardware/PCB filter")

    # ---- axes, before re-centring ------------------------------------
    pitch_c, pitch_ax = _axis_of(parts, lambda p: "6808-2RS" in p)
    if pitch_ax is None:                          # bearings were filtered: use the bores
        pitch_c, pitch_ax = _axis_of(parts, lambda p: "Input Gear A" in p)
        pitch_ax = np.array([0.0, 0.0, 1.0])
    yaw_c, yaw_ax = _axis_of(parts, lambda p: "2600N14" in p or "Output gear(Mirror)" in p)
    if yaw_ax is None:
        yaw_c, yaw_ax = _axis_of(parts, lambda p: "Output gear" in p)

    # The two axes are skew by construction only if the CAD is wrong; take the
    # pitch axis' height and the yaw axis' lateral position as the origin.
    origin = np.array([yaw_c[0], pitch_c[1], yaw_c[2]])
    print(f"  pitch axis {np.round(pitch_ax, 4)} through y={pitch_c[1]:.2f}")
    print(f"  yaw   axis {np.round(yaw_ax, 4)} through ({yaw_c[0]:.2f}, *, {yaw_c[2]:.2f})")
    print(f"  origin {np.round(origin, 2)}  (axes cross here)")

    # ---- boresight, BEFORE grouping ----------------------------------
    # The faceplate split needs it, and that runs inside the body loop below.
    bore = _boresight(parts, origin)

    # ---- group, merge, decimate --------------------------------------
    groups: dict[str, list] = {}
    unassigned = []
    for path, v, f in parts:
        b = _body_of(path)
        if b is None:
            unassigned.append((path, len(f)))
            continue
        groups.setdefault(b, []).append((v - origin, f))
    for path, n in unassigned:
        print(f"    ? unassigned, dropped: {path} ({n} tris)")

    # The GY-85 arrives as two solids; separate the bare board from the parts
    # standing on it so one can be silkscreen blue and the other IC black.
    if "head_imu" in groups:
        groups["head_imu"], silk = _split_silk(groups["head_imu"])
        if silk:
            groups["head_silk"] = silk

    out: dict[str, np.ndarray] = {}
    names = []
    finished: dict[str, tuple] = {}
    total_in = total_out = 0
    for body in BUDGET:
        if body not in groups:
            print(f"    ! body '{body}' has no geometry")
            continue
        # Decimate each PART on its own, then merge. Collapsing a merged soup
        # of disconnected solids lets the simplifier spend the whole budget on
        # one part and delete the others outright -- which is exactly what it
        # did: the frames came out as floating specks.
        raw = sum(len(f) for _, f in groups[body])
        total_in += raw
        vs, fs, off = [], [], 0
        for v, f in groups[body]:
            share = max(MIN_PART_TRIS, int(round(BUDGET[body] * len(f) / raw)))
            v, f = _repair(*_decimate(*_weld(v, f), share))
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        v = np.vstack(vs).astype(np.float32)
        f = np.vstack(fs).astype(np.int32)
        total_out += len(f)
        finished[body] = (v, f)
        print(f"  {body:10s} {len(groups[body]):2d} parts -> {len(v):5d} verts, "
              f"{len(f):5d} tris")

    for body, (v, f) in finished.items():
        sv, sf = _decimate(v, f, SHADOW_BUDGET)
        out[f"{body}_v"] = v
        out[f"{body}_f"] = f
        out[f"{body}_c"] = np.asarray(COLORS[body], np.float32)
        out[f"{body}_sv"] = sv
        out[f"{body}_sf"] = sf
        names.append(body)

    if bore is not None:
        out["boresight"] = bore.astype(np.float32)
        out["muzzle"] = _muzzle(parts, origin, bore).astype(np.float32)
        print(f"  boresight {np.round(bore, 4)}  muzzle {np.round(out['muzzle'], 1)}")

    allv = np.vstack([out[f"{b}_v"] for b in names])
    out["bodies"] = np.asarray(names)
    out["pitch_axis"] = pitch_ax.astype(np.float32)
    out["yaw_axis"] = yaw_ax.astype(np.float32)
    out["floor_y"] = np.float32(allv[:, 1].min())
    out["bounds"] = np.vstack([allv.min(0), allv.max(0)]).astype(np.float32)

    np.savez_compressed(OUT, **out)
    tmp.unlink(missing_ok=True)
    print(f"\n{total_in} -> {total_out} triangles")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main(sys.argv)
