"""The sponsor showcase: one episode as a polished 1080p film.

    avians showcase EPISODE [--out showcase.mp4] [--seconds 90]

What it shows, left to right:
    the valley       shaded terrain, search regions, the power line, the fire
                     zone (hatched red once closed), the pine stand, the radio
                     tower, the stations and depot
    the search       a heat map of where the person probably is (the prior,
                     cooled cell by cell as each camera pass fails to see them)
    the drones       an icon each with heading and its camera footprint, a trail
    the person       once found: the estimate, pulsing; the kit where it landed;
                     the rescue helicopter flying in at its arrival time
    the pilots       each drone's battery and current skill; the latest decision
                     and its reason as a caption
    the promises     the mission's STL clauses as chips that turn green when the
                     online monitor decides them (red if broken)

Time runs fast through the routine parts and slows down around the moments
that matter (found, report, drop, zone closure, help arriving), with a title
card and a results card. Rendering is matplotlib to ffmpeg (as the 2D replay).
"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

from .episode import Episode  # noqa: E402
from .replay2d import _ffmpeg, _timeline  # noqa: E402

BG = "#0f1419"
PANEL = "#171d24"
INK = "#e6e9ec"
QUIET = "#8b96a3"
ACCENT = "#ffb020"
GOOD = "#3ccf7a"
BAD = "#ff5c5c"
DRONE_COLS = ["#ffffff", "#5cc8ff", "#ff9f43", "#c084fc"]
KEY_EVENTS = ("found", "kit_landed", "released", "zone", "help")


# ---------------------------------------------------------------- time
def _frame_times(ep: Episode, seconds: float, fps: int, slow: float = 4.0) -> np.ndarray:
    """Sim times for each frame: a base speed chosen so the film lasts `seconds`,
    `slow`x slower within 15 s of a key moment."""
    t_end = float(ep.t[-1])
    keys = sorted({float(e["t"]) for e in ep.events if e["kind"] in KEY_EVENTS}
                  | {float(d["t_apply"]) for d in ep.decisions if d["tool"] in ("report", "deliver")
                     and d.get("t_apply") is not None})
    grid = np.linspace(0, t_end, 4000)
    weight = np.ones_like(grid)
    for k in keys:
        weight[np.abs(grid - k) < 15] = slow
    # film time spent per sim second is proportional to weight
    film = np.concatenate([[0], np.cumsum(weight[1:] * np.diff(grid))])
    film *= (seconds * fps) / film[-1]
    return np.interp(np.arange(int(seconds * fps)), film, grid)


# ---------------------------------------------------------------- terrain
def _terrain(world, res: float = 2.0):
    xs, ys = zip(*world.bounds)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    X, Y = np.meshgrid(np.arange(x0, x1, res), np.arange(y0, y1, res))
    rng = np.random.default_rng(7)
    z = np.zeros_like(X)
    for k in range(6):                               # a few smooth hills (deterministic)
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        s, h = rng.uniform(60, 160), rng.uniform(10, 40)
        z += h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    z += 0.004 * (Y - y0) * 20                       # the valley rises to the north (the ridge)
    gy, gx = np.gradient(z, res)
    shade = np.clip(0.55 + 0.9 * (-0.6 * gx + 0.8 * gy) / (1 + np.hypot(gx, gy)), 0, 1)
    base = np.array([0.20, 0.27, 0.19])
    img = base[None, None, :] * (0.6 + 0.6 * shade[..., None])
    return np.clip(img, 0, 1), (x0, x1, y0, y1)


def _heat_raster(world, coverage):
    """Map every coverage cell to a raster pixel once: (raster shape, extent, per region index arrays)."""
    xs, ys = zip(*world.bounds)
    x0, y0 = min(xs), min(ys)
    cell = coverage.cell
    nx, ny = int((max(xs) - x0) / cell) + 1, int((max(ys) - y0) / cell) + 1
    idx = {}
    for rid, g in coverage.grids.items():
        ix = ((g["X"] - x0) / cell).astype(int)
        iy = ((g["Y"] - y0) / cell).astype(int)
        idx[rid] = (iy, ix)
    return (ny, nx), (x0, x0 + nx * cell, y0, y0 + ny * cell), idx


# ---------------------------------------------------------------- render
def render(ep_dir, world, out=None, seconds: float = 90.0, fps: int = 24, dpi: int = 120) -> Path:
    from avians_perception import CameraModel
    from avians_skills.coverage import CoverageMap

    ep = Episode.load(ep_dir)
    out = Path(out or Path(ep_dir) / "showcase.mp4")
    ids = ep.drones
    tracks = {did: ep.drone(did) for did in ids}
    t = ep.t
    timeline = _timeline(ep, world)
    sc = ep.meta.get("scenario") or {}
    prior = sc.get("region_weights") or world.person.region_weights
    person = ep.meta.get("person")
    found_t = ep.summary.get("found_t")
    help_t = ep.summary.get("help_arrival_t")
    kit_events = [e for e in ep.events if e["kind"] == "kit_landed"]
    online = [json.loads(line) for line in open(Path(ep_dir) / "stl_online.jsonl")] \
        if (Path(ep_dir) / "stl_online.jsonl").exists() else []
    clauses = [c["clause"] for c in ep.score.get("clauses", [])]
    final = {c["clause"]: c["satisfied"] for c in ep.score.get("clauses", [])}
    says = {}
    try:
        import yaml
        mission = yaml.safe_load((Path(__file__).resolve().parents[4] / "missions" /
                                  f"{ep.meta['mission']}.yaml").read_text())
        says = {k: (v.get("says") if isinstance(v, dict) else "") for k, v in mission["spec"]["clauses"].items()}
        brief = mission.get("brief", "")
    except Exception:
        brief = ""

    # coverage, recomputed from the drones' paths at the signal rate
    cam = CameraModel()
    cov = CoverageMap(world.search_regions(), camera=cam)
    shape, hext, hidx = _heat_raster(world, cov)
    cov_t = 0

    times = _frame_times(ep, seconds, fps)
    title_frames, end_frames = int(3.5 * fps), int(5 * fps)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0.015, 0.06, 0.585, 0.88])
    ax.set_facecolor(BG)
    terrain, ext = _terrain(world)
    ax.imshow(terrain, extent=ext, origin="lower", interpolation="bilinear", zorder=0)
    heat = np.full(shape, np.nan)
    from matplotlib.colors import ListedColormap
    base = plt.get_cmap("inferno")(np.linspace(0.35, 1.0, 256))
    base[:, 3] = np.linspace(0.0, 0.7, 256) ** 0.8          # cold cells fade out: searched ground clears
    him = ax.imshow(heat, extent=hext, origin="lower", cmap=ListedColormap(base), vmin=0, vmax=1,
                    interpolation="nearest", zorder=1)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # static world
    zone_patches = {}
    for r in world.regions.values():
        if r.kind == "search":
            ax.add_patch(Polygon(r.polygon, closed=True, fill=False, ec="#d7f5d0", lw=1.2, alpha=0.8, zorder=2))
            cx, cy = r.centroid
            ax.text(cx, r.shape.bounds[3] - 8, f"{r.id}  {r.name}", color="#d7f5d0", fontsize=8.5,
                    ha="center", va="top", alpha=0.9, zorder=6)
        elif r.kind == "no_fly":
            ax.add_patch(Polygon(r.polygon, closed=True, fc=BAD, ec=BAD, alpha=0.35, hatch="////", lw=1, zorder=2))
            ax.text(r.centroid[0], r.centroid[1] + 14, "power line: no-fly", color="#ffb3b3", fontsize=8,
                    ha="center", zorder=6)
        elif r.kind == "hazard":
            p = Polygon(r.polygon, closed=True, fc=ACCENT, ec=ACCENT, alpha=0.12, lw=1.2, zorder=2)
            ax.add_patch(p)
            zone_patches[r.id] = p
            ax.text(r.centroid[0], r.shape.bounds[1] + 6, r.name.lower(), color=ACCENT, fontsize=8,
                    ha="center", zorder=6)
    for o in world.obstacles.values():
        if o.height >= 40:
            c = o.shape.centroid
            ax.plot(c.x, c.y, marker="^", color="#cfd8dc", ms=9, zorder=5)
            ax.text(c.x + 8, c.y, o.name.lower(), color="#cfd8dc", fontsize=7.5, va="center", zorder=6)
        else:
            x0, y0, x1, y1 = o.shape.bounds
            rng = np.random.default_rng(3)
            for _ in range(28):
                ax.add_patch(Circle((rng.uniform(x0, x1), rng.uniform(y0, y1)), rng.uniform(2, 4),
                                    fc="#2f5d34", ec="#173018", lw=0.5, zorder=4))
    for s in world.stations.values():
        ax.plot(*s.position, marker="s", color="#5cc8ff" if s.kind == "charge" else "#c084fc", ms=8, zorder=5)
        ax.text(s.position[0] + 7, s.position[1] - 9, s.name.lower(), color=QUIET, fontsize=7.5, zorder=6)
    for d in world.depots.values():
        ax.plot(*d.position, marker="P", color=ACCENT, ms=9, zorder=5)
        ax.text(d.position[0] + 7, d.position[1] + 6, "kit depot", color=ACCENT, fontsize=7.5, zorder=6)

    # dynamic artists
    trails = {did: ax.plot([], [], color=DRONE_COLS[k % 4], lw=1.1, alpha=0.75, zorder=7)[0]
              for k, did in enumerate(ids)}
    icons = {did: ax.plot([], [], marker=(3, 0, 0), color=DRONE_COLS[k % 4], ms=13, mec=BG, mew=1, zorder=9)[0]
             for k, did in enumerate(ids)}
    fps_circ = {did: Circle((0, 0), 1, fill=False, ec=DRONE_COLS[k % 4], lw=0.8, alpha=0.6, zorder=8)
                for k, did in enumerate(ids)}
    labels = {did: ax.text(0, 0, did or "drone", color=DRONE_COLS[k % 4], fontsize=8, zorder=9)
              for k, did in enumerate(ids)}
    for c in fps_circ.values():
        ax.add_patch(c)
    est_ring = Circle((0, 0), 1, fill=False, ec=BAD, lw=2.0, zorder=10, visible=False)
    ax.add_patch(est_ring)
    est_dot, = ax.plot([], [], marker="o", color=BAD, ms=6, zorder=10)
    person_lbl = ax.text(0, 0, "", color="#ffd0d0", fontsize=9, fontweight="bold", zorder=10)
    kit_dots, = ax.plot([], [], marker="s", ls="none", color=ACCENT, ms=8, mec="#3b2a00", zorder=10)
    heli, = ax.plot([], [], marker="+", color="#ffffff", ms=18, mew=3, zorder=11)
    heli_lbl = ax.text(0, 0, "", color="#ffffff", fontsize=8.5, zorder=11)

    # right column
    fig.text(0.625, 0.925, "AVIANS", color=INK, fontsize=24, fontweight="bold")
    fig.text(0.735, 0.932, "language-model pilots, search and rescue", color=QUIET, fontsize=12)
    clock = fig.text(0.625, 0.875, "", color=INK, fontsize=15, family="monospace")
    phase = fig.text(0.705, 0.877, "", color=ACCENT, fontsize=12, fontweight="bold")
    pilot_txt = {}
    y = 0.815
    for k, did in enumerate(ids):
        fig.text(0.625, y, f"● {did or 'drone'}", color=DRONE_COLS[k % 4], fontsize=11, fontweight="bold")
        pilot_txt[did] = (fig.text(0.68, y, "", color=INK, fontsize=10.5, family="monospace"),
                          fig.text(0.625, y - 0.03, "", color=QUIET, fontsize=10))
        y -= 0.075
    fig.text(0.625, y + 0.005, "THE PROMISES (signal temporal logic, checked live)", color=QUIET, fontsize=9.5)
    chips = {}
    cy0 = y - 0.035
    for i, c in enumerate(clauses):
        col, row = i % 2, i // 2
        x, yy = 0.625 + col * 0.18, cy0 - row * 0.047
        box = FancyBboxPatch((x, yy - 0.018), 0.172, 0.041, boxstyle="round,pad=0.002,rounding_size=0.006",
                             transform=fig.transFigure, fc=PANEL, ec="#2a333d", lw=1)
        fig.patches.append(box)
        lbl = "\n".join(textwrap.wrap(says.get(c) or c, 38)[:2])
        chips[c] = (box, fig.text(x + 0.007, yy + 0.003, lbl, color=QUIET, fontsize=7.6, va="center",
                                  linespacing=1.15))
    yfeed = cy0 - ((len(clauses) + 1) // 2) * 0.047 - 0.02
    fig.text(0.625, yfeed, "RADIO AND EVENTS", color=QUIET, fontsize=9.5)
    feed = fig.text(0.625, yfeed - 0.02, "", color=INK, fontsize=9, va="top", linespacing=1.5)
    caption_bg = FancyBboxPatch((0.03, 0.075), 0.555, 0.085, boxstyle="round,pad=0.004,rounding_size=0.01",
                                transform=fig.transFigure, fc="#000000", ec="none", alpha=0.55)
    fig.patches.append(caption_bg)
    caption = fig.text(0.045, 0.145, "", color=INK, fontsize=12, va="top", linespacing=1.4)

    # title / end cards
    card = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    card.set_zorder(100)                             # above every figure-level text and panel
    card.set_xticks([])
    card.set_yticks([])
    for sp in card.spines.values():
        sp.set_visible(False)
    card_t1 = card.text(0.5, 0.6, "", color=INK, fontsize=34, fontweight="bold", ha="center")
    card_t2 = card.text(0.5, 0.48, "", color=QUIET, fontsize=15, ha="center", va="top", linespacing=1.6)

    decisions = [d for d in ep.decisions if d["tool"] != "keep" and d.get("t_apply") is not None]
    feed_events = [e for e in ep.events if e["kind"] in ("found", "kit_landed", "zone", "help", "message", "news",
                                                         "reflex")]
    zone_closed_at = sorted(e.t for e in timeline if not e.safe)

    def set_card(visible, t1="", t2=""):
        card.set_visible(visible)
        card_t1.set_text(t1)
        card_t2.set_text(t2)

    def draw(tk: float):
        nonlocal cov_t
        i = int(np.searchsorted(t, tk, side="right")) - 1
        i = max(0, i)
        # coverage up to tk
        while cov_t <= i:
            for did in ids:
                s = tracks[did]
                if s["landed"][cov_t] < 0.5:
                    cov.mark(t[cov_t], s["east"][cov_t], s["north"][cov_t], s["alt"][cov_t],
                             cam.footprint(s["alt"][cov_t]), frame_s=float(t[1] - t[0]) if len(t) > 1 else 0.5)
            cov_t += 1
        heat[:] = np.nan
        dens = {}
        for rid, (iy, ix) in hidx.items():
            g = cov.grids[rid]
            # probability per cell: the region's prior share spread over its cells, times P(missed)
            dens[rid] = prior.get(rid, 0.0) / max(1, len(g["missed"])) * g["missed"]
        top = max((float(v.max()) for v in dens.values() if len(v)), default=1.0) or 1.0
        for rid, (iy, ix) in hidx.items():
            heat[iy, ix] = np.sqrt(dens[rid] / top)        # sqrt: cold cells stay visible
        if found_t is not None and tk >= found_t:
            heat[:] = np.nan                         # found: the search is over
        him.set_data(heat)
        # zones
        for zid, p in zone_patches.items():
            closed = any(e.zone == zid and e.t <= tk and not e.safe for e in timeline) and not any(
                e.zone == zid and e.safe and e.t <= tk and e.t > max(
                    (x.t for x in timeline if x.zone == zid and not x.safe and x.t <= tk), default=-1)
                for e in timeline)
            p.set_facecolor(BAD if closed else ACCENT)
            p.set_alpha(0.32 + 0.08 * math.sin(tk * 2) if closed else 0.12)
            p.set_hatch("xx" if closed else None)
        # drones
        for did in ids:
            s = tracks[did]
            lo = max(0, i - 400)
            trails[did].set_data(s["east"][lo: i + 1], s["north"][lo: i + 1])
            e, n, a = s["east"][i], s["north"][i], s["alt"][i]
            j = max(0, i - 2)
            hd = math.degrees(math.atan2(n - s["north"][j], e - s["east"][j])) - 90 if i > 2 else 0
            icons[did].set_data([e], [n])
            icons[did].set_marker((3, 0, hd))
            fps_circ[did].center = (e, n)
            fps_circ[did].set_radius(max(0.5, cam.footprint(a)))
            fps_circ[did].set_visible(s["landed"][i] < 0.5)
            labels[did].set_position((e + 6, n + 6))
            bat = s["battery"][i]
            bars = int(round(bat / 10))
            pilot_txt[did][0].set_text(f"{'█' * bars}{'░' * (10 - bars)} {bat:3.0f}%  alt {a:3.0f} m")
        # pilots' current skills
        for did in ids:
            mine = [d for d in decisions if (d.get("drone") or "") == did and d["t_apply"] <= tk]
            pilot_txt[did][1].set_text(textwrap.shorten(f"{mine[-1]['tool']}: {mine[-1]['reason']}", 62)
                                       if mine else "on the pad")
        # person
        if found_t is not None and tk >= found_t:
            ee, nn, sg = ep.sig["est_east"][i], ep.sig["est_north"][i], ep.sig["est_sigma"][i]
            est_ring.set_visible(True)
            est_ring.center = (ee, nn)
            est_ring.set_radius(max(4.0, sg) + 3 * (1 + math.sin(tk * 4)))
            est_dot.set_data([ee], [nn])
            person_lbl.set_position((ee + 10, nn - 4))
            person_lbl.set_text("hiker found")
        landed = [e for e in kit_events if e["t"] <= tk]
        if landed:
            import re
            pts = [tuple(map(float, re.findall(r"\(([-\d.]+), ([-\d.]+)\)", e["text"])[0])) for e in landed]
            kit_dots.set_data([p[0] for p in pts], [p[1] for p in pts])
        if help_t is not None and person is not None and help_t - 60 <= tk:
            f = min(1.0, (tk - (help_t - 60)) / 60)
            hx = ext[0] + (person[0] - ext[0]) * f
            hy = person[1] + 60 * (1 - f)
            heli.set_data([hx], [hy])
            heli_lbl.set_position((hx + 9, hy + 6))
            heli_lbl.set_text("rescue helicopter" if f < 1 else "help has arrived")
        # clock and phase
        clock.set_text(f"T+{int(tk // 60):02d}:{int(tk % 60):02d}")
        if help_t is not None and tk >= help_t:
            ph = "HELP ON SCENE"
        elif any(e["t"] <= tk for e in kit_events):
            ph = "SUPPLIES DELIVERED, WAITING FOR HELP"
        elif found_t is not None and tk >= found_t:
            ph = "FOUND: REPORTING, FETCHING THE KIT"
        else:
            ph = "SEARCHING"
        phase.set_text(ph)
        # chips
        dec = {}
        for rep in online:
            if rep["t"] <= tk:
                for c, v in rep.get("decided", {}).items():
                    dec[c] = v["verdict"]
        if tk >= t[-1] - 1:
            dec = {c: ("satisfied" if ok else "violated") for c, ok in final.items()}
        for c, (box, lbl) in chips.items():
            v = dec.get(c)
            box.set_facecolor({"satisfied": "#123d26", "violated": "#4a1414"}.get(v, PANEL))
            box.set_edgecolor({"satisfied": GOOD, "violated": BAD}.get(v, "#2a333d"))
            lbl.set_color({"satisfied": "#bff5d3", "violated": "#ffc9c9"}.get(v, QUIET))
        # feed
        recent = [e for e in feed_events if e["t"] <= tk][-6:]
        feed.set_text("\n".join(textwrap.shorten(f"T+{int(e['t'] // 60):02d}:{int(e['t'] % 60):02d}  "
                                                 f"{(e.get('drone') + ' ') if e.get('drone') and len(ids) > 1 else ''}"
                                                 f"{e['text']}", 70) for e in recent))
        # caption: the latest decision, held for a while
        last = [d for d in decisions if d["t_apply"] <= tk]
        if last and tk - last[-1]["t_apply"] < 45:
            d = last[-1]
            who = f"{d['drone']} pilot" if d.get("drone") else "pilot"
            caption.set_text(f"{who}: {d['tool'].replace('_', ' ')}\n"
                             + "\n".join(textwrap.wrap(d["reason"], 86)[:2]))
        else:
            caption.set_text("")
        caption_bg.set_visible(bool(caption.get_text()))

    _ffmpeg()
    writer = FFMpegWriter(fps=fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    pilot = ep.meta.get("pilot", "")
    with writer.saving(fig, str(out), dpi):
        set_card(True, "Find the hiker. Get them help.",
                 "\n".join(textwrap.wrap(brief, 90)[:5]) + f"\n\n{len(ids)} drone(s), pilot: {pilot}")
        for _ in range(title_frames):
            writer.grab_frame()
        set_card(False)
        for tk in times:
            draw(float(tk))
            writer.grab_frame()
        ok = sum(final.values())
        res = (f"{ok} of {len(final)} promises kept"
               + (f"\nhiker found at T+{int(found_t // 60):02d}:{int(found_t % 60):02d}" if found_t else "")
               + (f"\nkit landed {ep.summary.get('drop_err'):.1f} m from them" if ep.summary.get("drop_err") else "")
               + (f"\nhelp on scene at T+{int(help_t // 60):02d}:{int(help_t % 60):02d}" if help_t else "")
               + (f"\n{ep.summary.get('decisions')} pilot decisions, ${ep.summary.get('cost_usd', 0):.3f} of model calls"))
        set_card(True, "Mission complete" if ep.score.get("satisfied") else "Mission result", res)
        for _ in range(end_frames):
            writer.grab_frame()
    plt.close(fig)
    return out
