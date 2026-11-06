"""The world in Gazebo: SDF models spawned into a running AAS world.

STARTER (owner: Taylor). What works today: every region as a translucent
patch + outline + volume (visual only, as in avians-02), obstacles as solid
prisms, stations and depots as pads, and the person as a model at their
scenario position. avians_sim.gazebo spawns these before takeoff.

To do (see README.md, "Gazebo"):
  - a full world file (terrain, trees, textures) instead of spawning into
    AAS's impalpable_greyness, so the camera and YOLO see something realistic
  - the person as an actual human model (rescue_randy_sitting from Gazebo
    Fuel, as in AAS's worlds) instead of the red box
  - zone safety changes shown in the scene (recolour zone_fire at t=420 s)
  - the payload: a model attached with Gazebo's DetachableJoint system, so
    deliver() releases it and it falls under physics
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

from .world import WorldSpec

RGB = {"search": (0.05, 0.64, 0.05), "no_fly": (0.82, 0.23, 0.23), "hazard": (0.95, 0.55, 0.1),
       "landing": (0.3, 0.3, 0.3), "other": (0.54, 0.53, 0.51)}
GROUND = (0.86, 0.86, 0.86)
UNBOUNDED_HEIGHT = 30.0


def _material(rgb, alpha: float = 1.0) -> str:
    r, g, b = rgb
    return (f"<material><ambient>{r:.3f} {g:.3f} {b:.3f} {alpha}</ambient>"
            f"<diffuse>{r:.3f} {g:.3f} {b:.3f} {alpha}</diffuse></material>"
            + (f"<transparency>{1 - alpha:.2f}</transparency>" if alpha < 1 else ""))


def _model(name: str, anchor, body: str, pose: str = "0 0 0 0 0 0") -> str:
    ax, ay = anchor
    x, y, z, *rest = (float(v) for v in pose.split())
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{escape(name)}'>"
            f"<static>true</static><pose>{ax + x:.3f} {ay + y:.3f} {z:.3f} "
            f"{' '.join(f'{v:g}' for v in rest)}</pose><link name='link'>{body}</link></model></sdf>")


def _poly(points, h: float) -> str:
    pts = "".join(f"<point>{e:.3f} {n:.3f}</point>" for e, n in points)
    return f"<geometry><polyline>{pts}<height>{h:.3f}</height></polyline></geometry>"


def region_models(w: WorldSpec, anchor=(0.0, 0.0)) -> dict[str, str]:
    out = {}
    for rid, r in w.regions.items():
        floor, ceiling = r.altitude or (0.0, UNBOUNDED_HEIGHT)
        rgb = RGB.get(r.kind, RGB["other"])
        tint = tuple(0.45 * c + 0.55 * g for c, g in zip(rgb, GROUND))
        vis = [f"<visual name='patch'><pose>0 0 0.01 0 0 0</pose>{_poly(r.polygon, 0.02)}"
               f"{_material(tint)}<cast_shadows>false</cast_shadows></visual>"]
        for k, ((e1, n1), (e2, n2)) in enumerate(zip(r.polygon, r.polygon[1:] + r.polygon[:1])):
            length, yaw = math.hypot(e2 - e1, n2 - n1), math.atan2(n2 - n1, e2 - e1)
            vis.append(f"<visual name='edge_{k}'><pose>{(e1 + e2) / 2:.3f} {(n1 + n2) / 2:.3f} 0.035 "
                       f"0 0 {yaw:.4f}</pose><geometry><box><size>{length + 0.25:.3f} 0.25 0.05"
                       f"</size></box></geometry>{_material(rgb)}</visual>")
        vis.append(f"<visual name='volume'><pose>0 0 {floor:.3f} 0 0 0</pose>"
                   f"{_poly(r.polygon, ceiling - floor)}{_material(rgb, 0.12)}"
                   f"<cast_shadows>false</cast_shadows></visual>")
        out[f"avians_region_{rid}"] = _model(f"avians_region_{rid}", anchor, "".join(vis))
    return out


def obstacle_models(w: WorldSpec, anchor=(0.0, 0.0)) -> dict[str, str]:
    out = {}
    for oid, o in w.obstacles.items():
        geo = _poly(o.polygon, o.height)
        body = (f"<visual name='v'>{geo}{_material((0.25, 0.35, 0.2))}</visual>"
                f"<collision name='c'>{geo}</collision>")
        out[f"avians_obstacle_{oid}"] = _model(f"avians_obstacle_{oid}", anchor, body)
    return out


def pad_models(w: WorldSpec, anchor=(0.0, 0.0)) -> dict[str, str]:
    out = {}
    colours = {"charge": (0.1, 0.4, 0.9), "swap": (0.6, 0.2, 0.8), "depot": (0.9, 0.8, 0.1)}
    pads = [(s.id, s.kind, s.position) for s in w.stations.values()]
    pads += [(d.id, "depot", d.position) for d in w.depots.values()]
    for pid, kind, (e, n) in pads:
        body = (f"<visual name='v'><geometry><cylinder><radius>1.5</radius><length>0.05</length>"
                f"</cylinder></geometry>{_material(colours[kind])}</visual>")
        out[f"avians_pad_{pid}"] = _model(f"avians_pad_{pid}", anchor, body, f"{e} {n} 0.03 0 0 0")
    return out


def person_model(position, anchor=(0.0, 0.0)) -> tuple[str, str]:
    """A red box the size of a sitting person. TODO(Taylor): rescue_randy."""
    e, n = position
    body = (f"<visual name='v'><geometry><box><size>0.6 0.9 0.9</size></box></geometry>"
            f"{_material((0.85, 0.1, 0.1))}</visual>")
    return "avians_person", _model("avians_person", anchor, body, f"{e} {n} 0.45 0 0 0")


def all_models(w: WorldSpec, person_at=None, anchor=(0.0, 0.0)) -> dict[str, str]:
    out = {**region_models(w, anchor), **obstacle_models(w, anchor), **pad_models(w, anchor)}
    if person_at is not None:
        name, sdf = person_model(person_at, anchor)
        out[name] = sdf
    return out


def world_sdf(w: WorldSpec) -> str:
    """A complete Gazebo world for this WorldSpec. TODO(Taylor)."""
    raise NotImplementedError("Taylor: a full SDF world (terrain, trees, person model); "
                              "until then avians_sim spawns all_models() into AAS's world")
