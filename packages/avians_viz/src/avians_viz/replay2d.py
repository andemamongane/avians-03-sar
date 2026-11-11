"""The sped-up 2D replay: map + drone + belief + timeline, as an mp4.

    avians video <episode_dir> [--speed 20]

Left: the world (regions tinted by kind, closed zones hatched once closed,
obstacles, pads), the drone's trail, the true person (x), the estimate with
its 1-sigma circle, the kit where it landed. Right: battery and distance to
the person over time, and the decision being executed with its reason.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from .episode import Episode  # noqa: E402


def _ffmpeg() -> None:
    """Use ffmpeg from PATH, else the binary bundled with imageio-ffmpeg."""
    import shutil
    if shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]):
        return
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("no ffmpeg: install it (apt/brew) or `pip install imageio-ffmpeg`") from None

COLOURS = {"search": "#2e9d3a", "no_fly": "#d13b3b", "hazard": "#f08a1c", "landing": "#666666",
           "other": "#8a8781"}


def _timeline(ep, world):
    """The episode's own zone timeline (a family member's), else the world's."""
    sc = ep.meta.get("scenario") or {}
    if sc.get("timeline") is not None:
        from avians_env import SafetyEvent
        return [SafetyEvent(**e) for e in sc["timeline"]]
    return world.safety_timeline


def draw_world(ax, world, t: float | None = None, timeline=None):
    for r in world.regions.values():
        c = COLOURS.get(r.kind, COLOURS["other"])
        closed = False
        if timeline is not None and t is not None:
            closed = any(e.zone == r.id and e.t <= t and not e.safe for e in timeline) and \
                not any(e.zone == r.id and e.t <= t and e.safe and e.t > max(
                    (x.t for x in timeline if x.zone == r.id and not x.safe and x.t <= t), default=-1)
                    for e in timeline)
        ax.add_patch(Polygon(r.polygon, closed=True, fc=c, alpha=0.35 if closed else 0.12, ec=c, lw=1.2,
                             hatch="///" if closed else None))
        cx, cy = r.centroid
        ax.text(cx, cy, r.id, ha="center", va="center", fontsize=8, color=c)
    for o in world.obstacles.values():
        ax.add_patch(Polygon(o.polygon, closed=True, fc="#3d4b2f", ec="none", alpha=0.8))
    for s in world.stations.values():
        ax.plot(*s.position, marker="s", color="#1f5fd1" if s.kind == "charge" else "#8a33cc", ms=7)
        ax.text(s.position[0] + 6, s.position[1] - 10, s.id, fontsize=7)
    for d in world.depots.values():
        ax.plot(*d.position, marker="D", color="#c9a400", ms=7)
    if world.bounds:
        xs, ys = zip(*world.bounds)
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")


def render(ep_dir: str | Path, world, out: str | Path | None = None, speed: float = 20.0,
           fps: int = 10, dpi: int = 100) -> Path:
    ep = Episode.load(ep_dir)
    out = Path(out or Path(ep_dir) / "replay.mp4")
    t, s = ep.t, ep.sig
    person = ep.meta.get("person")
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.04, 0.08, 0.55, 0.86])
    ax_b = fig.add_axes([0.66, 0.62, 0.31, 0.3])
    ax_d = fig.add_axes([0.66, 0.26, 0.31, 0.27])
    ax_txt = fig.add_axes([0.62, 0.0, 0.37, 0.2])
    ax_txt.axis("off")
    ax_b.plot(t, s["battery"], color="#1f5fd1")
    ax_b.axhline(10, color="#d13b3b", ls="--", lw=0.8)
    ax_b.set_ylabel("battery %")
    ax_b.set_xlim(0, t[-1])
    ax_d.plot(t, np.minimum(s["dist_person"], 400), color="#555")
    ax_d.axhline(30, color="#d13b3b", ls="--", lw=0.8)
    ax_d.set_ylabel("to person (m)")
    ax_d.set_xlabel("t (s)")
    ax_d.set_xlim(0, t[-1])
    cur_b = ax_b.axvline(0, color="k", lw=0.8)
    cur_d = ax_d.axvline(0, color="k", lw=0.8)
    timeline = _timeline(ep, world)
    trail, = ax.plot([], [], color="#222", lw=1.0)
    drone, = ax.plot([], [], marker="o", color="#000", ms=6)
    est, = ax.plot([], [], marker="+", color="#c0392b", ms=10, mew=2)
    sig_c = Circle((0, 0), 1, fill=False, ec="#c0392b", lw=1, ls="--")
    ax.add_patch(sig_c)
    if person:
        ax.plot(person[0], person[1], marker="x", color="#c0392b", ms=8, alpha=0.7)
    kit, = ax.plot([], [], marker="*", color="#c9a400", ms=12, ls="none")
    title = ax.set_title("")
    txt = ax_txt.text(0, 1, "", va="top", fontsize=9, family="monospace")
    kit_events = [e for e in ep.events if e["kind"] == "kit_landed"]
    n_frames = int(t[-1] / speed * fps) + 1
    drawn_world = {"closed": None}

    def frame(k):
        tk = k * speed / fps
        i = int(np.searchsorted(t, tk, side="right")) - 1
        closed = tuple(e.zone for e in timeline if e.t <= tk and not e.safe)
        if closed != drawn_world["closed"]:
            for p in list(ax.patches):
                if p is not sig_c:
                    p.remove()
            for tx in list(ax.texts):
                tx.remove()
            draw_world(ax, world, tk, timeline)
            drawn_world["closed"] = closed
        trail.set_data(s["east"][: i + 1], s["north"][: i + 1])
        drone.set_data([s["east"][i]], [s["north"][i]])
        est.set_data([s["est_east"][i]], [s["est_north"][i]])
        sig_c.center = (s["est_east"][i], s["est_north"][i])
        sig_c.set_radius(max(1.0, s["est_sigma"][i]))
        landed = [e for e in kit_events if e["t"] <= tk]
        if landed:
            import re
            pts = [tuple(map(float, re.findall(r"\(([-\d.]+), ([-\d.]+)\)", e["text"])[0])) for e in landed]
            kit.set_data([p[0] for p in pts], [p[1] for p in pts])
        cur_b.set_xdata([tk, tk])
        cur_d.set_xdata([tk, tk])
        title.set_text(f"{ep.meta.get('episode', '')}   t = {tk:5.0f} s   alt {s['alt'][i]:.0f} m   "
                       f"{'FOUND' if s['found'][i] > 0 else 'searching'}"
                       + ("   HELP ARRIVED" if s['help'][i] > 0 else ""))
        dec = [d for d in ep.decisions if (d["t_apply"] or 0) <= tk and d["tool"] != "keep"]
        if dec:
            d = dec[-1]
            args = ", ".join(f"{a}={v}" for a, v in d["args"].items())
            body = f"t={d['t_apply']:.0f}  {d['tool']}({args})\n" + "\n".join(textwrap.wrap(d["reason"], 48)[:3])
            txt.set_text(body)
        return trail, drone, est

    _ffmpeg()
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    with writer.saving(fig, str(out), dpi):
        for k in range(n_frames):
            frame(k)
            writer.grab_frame()
    plt.close(fig)
    return out


def summary_png(ep_dir, world, out=None) -> Path:
    ep = Episode.load(ep_dir)
    out = Path(out or Path(ep_dir) / "summary.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_world(ax, world, ep.t[-1], _timeline(ep, world))
    ax.plot(ep.sig["east"], ep.sig["north"], color="#222", lw=0.8)
    if ep.meta.get("person"):
        ax.plot(*ep.meta["person"], marker="x", color="#c0392b", ms=9)
    ax.set_title(f"{ep.meta.get('episode', '')}: robustness {ep.score.get('robustness')}, "
                 f"{'satisfied' if ep.score.get('satisfied') else 'violated'}")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
