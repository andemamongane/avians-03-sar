"""The sponsor showcase: one episode as a 1080p film that starts in the air.

    avians showcase EPISODE [--out showcase.mp4] [--seconds 90]

Left, the valley: its zones (disjoint, tiling the whole area) coloured by
live status (open, caution, closed by a storm, burning, no-fly), the storm
cells drifting across, the search heat map (where the hiker probably is,
cooling as the cameras sweep), the drones with their camera footprints and
trails, the hiker once found, the kit where it landed, the helicopter flying
in; the mission clock and phase in the corner; radio traffic and zone
changes as a caption.

Right, the pilots and the promises:
    a card per drone: battery, altitude, the skill it runs and its
    arguments, when it decided, how long it thought, when the call took
    effect, its reason, and an excerpt of its thinking summary (or
    "thinking..." while a call is in flight)
    the mission spec in SFO notation (quantifiers over drones and zones),
    each clause with its robustness from the online monitor: an interval
    while undecided, a value with a check or a cross once decided; a
    quantified clause names its tightest zone

Time runs fast through routine flight and slows around the moments that
matter (found, report, drop, closures, help arriving). No title or results
card: the film opens on the flight.
"""
from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from .episode import Episode  # noqa: E402
from .replay2d import _ffmpeg  # noqa: E402

BG = "#0f1419"
PANEL = "#171d24"
INK = "#e6e9ec"
QUIET = "#8b96a3"
ACCENT = "#ffb020"
GOOD = "#3ccf7a"
BAD = "#ff5c5c"
STORM = "#7aa2ff"
FIRE = "#ff6a3d"
DRONE_COLS = ["#ffffff", "#5cc8ff", "#ff9f43", "#c084fc"]
KEY_EVENTS = ("found", "kit_landed", "released", "help")


# ---------------------------------------------------------------- helpers
def _strip_outer(t: str) -> str:
    """Drop one pair of brackets that encloses the whole string."""
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        for k, ch in enumerate(t):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and k < len(t) - 1:
                return t
        t = t[1:-1].strip()
    return t


def sfo(text: str) -> str:
    """A spec formula in compact mathematical notation: D drones, Z zones, S search zones;
    nested quantifiers merged, redundant outer brackets dropped."""
    t = text.strip()
    m = re.match(r"^([GFHO]\[[^\]]*\])\s*(.*)$", t)
    head, body = (m.group(1) + " ", m.group(2)) if m else ("", t)
    body = _strip_outer(body)
    # forall x in drones: (forall z in zones: (...))  ->  forall x in D forall z in Z: ...
    q = re.match(r"^forall (\w+) in (\w+):\s*(.*)$", body)
    if q:
        inner = _strip_outer(q.group(3))
        q2 = re.match(r"^forall (\w+) in (\w+):\s*(.*)$", inner)
        if q2:
            body = f"forall {q.group(1)} in {q.group(2)} forall {q2.group(1)} in {q2.group(2)}: {_strip_outer(q2.group(3))}"
        else:
            body = f"forall {q.group(1)} in {q.group(2)}: {inner}"
    if head and not body.startswith(("forall", "exists")) and any(op in body for op in ("->", " & ", " | ")):
        body = f"({body})"                     # G (p → q), never G p → q
    t = head + body
    for a, b in (("forall ", "∀"), ("exists ", "∃"), (" in drones", "∈D"), (" in search_zones", "∈S"),
                 (" in zones", "∈Z"), ("->", "→"), (">=", "≥"), ("<=", "≤"), ("!", "¬"),
                 (" & ", " ∧ "), (" | ", " ∨ "), ("[0,1500]", ""), ("[0,1300]", "[0,1300]")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).replace("G ", "G ").strip()


def _frame_times(ep: Episode, seconds: float, fps: int, slow: float = 4.0) -> np.ndarray:
    """Sim times for each frame: base speed so the film lasts `seconds`, `slow`x slower
    within 15 s of a key moment."""
    t_end = float(ep.t[-1])
    keys = sorted({float(e["t"]) for e in ep.events if e["kind"] in KEY_EVENTS}
                  | {float(d["t_apply"]) for d in ep.decisions if d["tool"] in ("report", "deliver")
                     and d.get("t_apply") is not None})
    grid = np.linspace(0, t_end, 4000)
    weight = np.ones_like(grid)
    for k in keys:
        weight[np.abs(grid - k) < 15] = slow
    film = np.concatenate([[0], np.cumsum(weight[1:] * np.diff(grid))])
    film *= (seconds * fps) / film[-1]
    return np.interp(np.arange(int(seconds * fps)), film, grid)


def _terrain(world, res: float = 2.0):
    xs, ys = zip(*world.bounds)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    X, Y = np.meshgrid(np.arange(x0, x1, res), np.arange(y0, y1, res))
    rng = np.random.default_rng(7)
    z = np.zeros_like(X)
    for _ in range(6):
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        s, h = rng.uniform(60, 160), rng.uniform(10, 40)
        z += h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * s * s))
    z += 0.08 * (Y - y0)
    gy, gx = np.gradient(z, res)
    shade = np.clip(0.55 + 0.9 * (-0.6 * gx + 0.8 * gy) / (1 + np.hypot(gx, gy)), 0, 1)
    base = np.array([0.20, 0.27, 0.19])
    return np.clip(base[None, None, :] * (0.6 + 0.6 * shade[..., None]), 0, 1), (x0, x1, y0, y1)


def _heat_raster(world, coverage):
    xs, ys = zip(*world.bounds)
    x0, y0 = min(xs), min(ys)
    cell = coverage.cell
    nx, ny = int((max(xs) - x0) / cell) + 1, int((max(ys) - y0) / cell) + 1
    idx = {rid: (((g["Y"] - y0) / cell).astype(int), ((g["X"] - x0) / cell).astype(int))
           for rid, g in coverage.grids.items()}
    return (ny, nx), (x0, x0 + nx * cell, y0, y0 + ny * cell), idx


def _mission(ep: Episode, world):
    """The episode's MissionSpec, rebuilt from the mission file (formulas, domains, regions)."""
    import yaml
    from avians_spec import MissionSpec, sar_registry
    root = Path(__file__).resolve().parents[4]
    raw = yaml.safe_load((root / "missions" / f"{ep.meta['mission']}.yaml").read_text())
    ids = ep.meta.get("drones") or []
    reg = sar_registry(world, drones=ids if len(ids) > 1 else None)
    return MissionSpec.from_dict(raw["spec"], reg, world.regions)


def _tightest(ep: Episode, spec, every_s: float = 10.0) -> dict:
    """For quantified clauses (over zones or search zones): at sampled times, the member
    with the smallest robustness upper bound so far. {clause: [(t, member), ...]}."""
    from avians_spec import Trace, stl
    rows = [json.loads(line) for line in open(ep.dir / "signals.jsonl")]
    times = [r["t"] for r in rows]
    full = Trace.from_samples(times, rows, spec.dt, spec.registry.defaults())
    out = {}
    for c in spec.clauses:
        dom = "search_zones" if " in search_zones" in c.formula else ("zones" if " in zones" in c.formula else None)
        if dom is None:
            continue
        members = spec.members(c.name, dom)
        series = []
        for tk in np.arange(every_s, times[-1] + 1e-9, every_s):
            n = int(tk / spec.dt) + 1
            pre = Trace(spec.dt, {k: v[:n] for k, v in full.signals.items()})
            best = min(members, key=lambda m: stl.interval(m[1], pre, spec.regions, spec.registry, 15.0)[1])
            series.append((float(tk), best[0]))
        out[c.name] = series
    return out


# ---------------------------------------------------------------- render
def render(ep_dir, world, out=None, seconds: float = 90.0, fps: int = 24, dpi: int = 120) -> Path:
    from avians_env.dynamics import Storm
    from avians_env.world import SafetyEvent
    from avians_perception import CameraModel
    from avians_skills.coverage import CoverageMap

    ep = Episode.load(ep_dir)
    out = Path(out or Path(ep_dir) / "showcase.mp4")
    ids = ep.drones
    tracks = {did: ep.drone(did) for did in ids}
    t = ep.t
    sc = ep.meta.get("scenario") or {}
    timeline = [SafetyEvent(**e) for e in sc.get("timeline", [])]
    storms = [Storm(**s) for s in sc.get("storms", [])]
    burning = {f["zone"]: f["t"] for f in sc.get("fires", [])}
    caution_m = float((world.dynamics.get("storms") or {}).get("caution_m", 60))
    prior = sc.get("region_weights") or world.person.region_weights
    person = ep.meta.get("person")
    found_t = ep.summary.get("found_t")
    help_t = ep.summary.get("help_arrival_t")
    kit_events = [e for e in ep.events if e["kind"] == "kit_landed"]
    online = [json.loads(line) for line in open(ep.dir / "stl_online.jsonl")] \
        if (ep.dir / "stl_online.jsonl").exists() else []
    spec = _mission(ep, world)
    tight = _tightest(ep, spec)
    final = {c["clause"]: c for c in ep.score.get("clauses", [])}

    cam = CameraModel()
    cov = CoverageMap(world.search_regions(), camera=cam)
    shape, hext, hidx = _heat_raster(world, cov)
    cov_i = [0]
    times = _frame_times(ep, seconds, fps)

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0.01, 0.03, 0.56, 0.94])
    ax.set_facecolor(BG)
    terrain, ext = _terrain(world)
    ax.imshow(terrain, extent=ext, origin="lower", interpolation="bilinear", zorder=0)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.axis("off")

    # zones (one patch each, recoloured per frame)
    zp = {}
    for k, r in enumerate(world.regions.values()):
        base_fc = {"no_fly": BAD, "landing": "#5b6b78", "search": "#3f5a36"}.get(r.kind, "#34492d" if k % 2 else "#2e4228")
        p = Polygon(r.polygon, closed=True, fc=base_fc, ec="#cfe8c8", lw=0.9,
                    alpha=0.45 if r.kind == "no_fly" else 0.55, hatch="////" if r.kind == "no_fly" else None, zorder=1)
        ax.add_patch(p)
        cx, cy = r.shape.representative_point().coords[0]
        lbl = ax.text(cx, cy + 5, f"{r.id if r.kind == 'search' else ''}  {r.name}".strip(), color="#e8f5e3",
                      fontsize=7.6, ha="center", fontweight="bold" if r.kind == "search" else "normal", zorder=6)
        st = ax.text(cx, cy - 9, "", color=ACCENT, fontsize=7, ha="center", zorder=6)
        zp[r.id] = (p, base_fc, lbl, st)
    heat = np.full(shape, np.nan)
    base = plt.get_cmap("inferno")(np.linspace(0.35, 1.0, 256))
    base[:, 3] = np.linspace(0.0, 0.7, 256) ** 0.8
    him = ax.imshow(heat, extent=hext, origin="lower", cmap=ListedColormap(base), vmin=0, vmax=1,
                    interpolation="nearest", zorder=2)
    storm_patches = {s.id: Circle((0, 0), s.radius, fc=STORM, ec=STORM, alpha=0.22, lw=1.5, ls="--", zorder=3,
                                  visible=False) for s in storms}
    for c in storm_patches.values():
        ax.add_patch(c)
    storm_lbl = {s.id: ax.text(0, 0, "", color="#c9d8ff", fontsize=7.5, ha="center", zorder=7) for s in storms}
    for o in world.obstacles.values():
        if o.height >= 40:
            c = o.shape.centroid
            ax.plot(c.x, c.y, marker="^", color="#cfd8dc", ms=8, zorder=5)
        else:
            x0, y0, x1, y1 = o.shape.bounds
            rng = np.random.default_rng(3)
            for _ in range(24):
                ax.add_patch(Circle((rng.uniform(x0, x1), rng.uniform(y0, y1)), rng.uniform(2, 4),
                                    fc="#2f5d34", ec="#173018", lw=0.5, zorder=4))
    for s_ in world.stations.values():
        ax.plot(*s_.position, marker="s", color="#5cc8ff" if s_.kind == "charge" else "#c084fc", ms=7, zorder=5)
    for d in world.depots.values():
        ax.plot(*d.position, marker="P", color=ACCENT, ms=8, zorder=5)

    trails = {did: ax.plot([], [], color=DRONE_COLS[k % 4], lw=1.1, alpha=0.8, zorder=7)[0] for k, did in enumerate(ids)}
    icons = {did: ax.plot([], [], marker=(3, 0, 0), color=DRONE_COLS[k % 4], ms=13, mec=BG, mew=1, zorder=9)[0]
             for k, did in enumerate(ids)}
    fps_c = {did: Circle((0, 0), 1, fill=False, ec=DRONE_COLS[k % 4], lw=0.8, alpha=0.6, zorder=8)
             for k, did in enumerate(ids)}
    for c in fps_c.values():
        ax.add_patch(c)
    dl = {did: ax.text(0, 0, did or "drone", color=DRONE_COLS[k % 4], fontsize=8.5, zorder=9) for k, did in enumerate(ids)}
    est_ring = Circle((0, 0), 1, fill=False, ec=BAD, lw=2.0, zorder=10, visible=False)
    ax.add_patch(est_ring)
    person_lbl = ax.text(0, 0, "", color="#ffd0d0", fontsize=9, fontweight="bold", zorder=10)
    kit_dots, = ax.plot([], [], marker="s", ls="none", color=ACCENT, ms=8, mec="#3b2a00", zorder=10)
    heli, = ax.plot([], [], marker="+", color="#ffffff", ms=18, mew=3, zorder=11)
    heli_lbl = ax.text(0, 0, "", color="#ffffff", fontsize=8.5, zorder=11)
    clock = ax.text(ext[0] + 8, ext[3] - 8, "", color=INK, fontsize=15, family="monospace", va="top", zorder=12)
    phase = ax.text(ext[0] + 8, ext[3] - 30, "", color=ACCENT, fontsize=10.5, fontweight="bold", va="top", zorder=12)
    cap_bg = FancyBboxPatch((0.02, 0.04), 0.53, 0.06, boxstyle="round,pad=0.004,rounding_size=0.01",
                            transform=fig.transFigure, fc="#000000", ec="none", alpha=0.6)
    fig.patches.append(cap_bg)
    caption = fig.text(0.03, 0.085, "", color=INK, fontsize=10.5, va="top", linespacing=1.4)

    # ---- sidebar: decision cards
    X = 0.585
    fig.text(X, 0.955, "PILOT DECISIONS", color=QUIET, fontsize=10, fontweight="bold")
    card_h = 0.215 if len(ids) > 1 else 0.3
    cards = {}
    y = 0.935
    for k, did in enumerate(ids):
        col = DRONE_COLS[k % 4]
        fig.patches.append(FancyBboxPatch((X, y - card_h), 0.405, card_h - 0.01, boxstyle="round,pad=0.003,rounding_size=0.008",
                                          transform=fig.transFigure, fc=PANEL, ec=col, lw=1.2))
        fig.text(X + 0.01, y - 0.025, f"● {did or 'drone'}", color=col, fontsize=12, fontweight="bold")
        cards[did] = {"stat": fig.text(X + 0.065, y - 0.025, "", color=QUIET, fontsize=9.5, family="monospace"),
                      "call": fig.text(X + 0.01, y - 0.058, "", color=INK, fontsize=10.2, family="monospace"),
                      "when": fig.text(X + 0.01, y - 0.08, "", color=QUIET, fontsize=8.5),
                      "why": fig.text(X + 0.01, y - 0.098, "", color=INK, fontsize=9, va="top", linespacing=1.3),
                      "think": fig.text(X + 0.01, y - 0.145 - (0.04 if len(ids) == 1 else 0), "", color="#b8c2cc",
                                        fontsize=8.4, va="top", style="italic", linespacing=1.3)}
        y -= card_h + 0.015
    # ---- spec panel
    fig.text(X, y - 0.005, "SPEC (SFO)  ·  robustness ρ from the live monitor", color=QUIET, fontsize=9.5,
             fontweight="bold")
    fig.text(X, y - 0.027, "D drones · Z zones · S search zones · G always, F eventually, H historically, O once"
             " · unmarked G/F: the whole mission", color=QUIET, fontsize=7.6)
    rows_y = y - 0.055
    step = min(0.032, (rows_y - 0.03) / max(1, len(spec.clauses)))
    spec_rows = {}
    for c in spec.clauses:
        f = sfo(c.formula)
        fs = min(8.6, 450.0 / max(1, len(f)))           # fit the formula column, room left for ρ
        spec_rows[c.name] = (fig.text(X, rows_y, f, color=INK, fontsize=fs, family="monospace"),
                             fig.text(0.99, rows_y, "", color=QUIET, fontsize=8.4, family="monospace", ha="right"))
        rows_y -= step

    decisions = sorted([d for d in ep.decisions if d.get("t_apply") is not None], key=lambda d: d["t_obs"] or 0)
    radio = [e for e in ep.events if e["kind"] in ("message", "zone", "news", "found", "kit_landed", "help", "reflex")]

    def fmt(v):
        if v in ("inf", "-inf"):
            return "∞" if v == "inf" else "-∞"
        v = float(v)
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"

    def zone_state(zid, tk):
        safe = True
        for e in timeline:
            if e.zone == zid and e.t <= tk:
                safe = e.safe
        return safe

    def draw(tk: float):
        i = max(0, int(np.searchsorted(t, tk, side="right")) - 1)
        while cov_i[0] <= i:
            j = cov_i[0]
            for did in ids:
                s = tracks[did]
                if s["landed"][j] < 0.5:
                    cov.mark(t[j], s["east"][j], s["north"][j], s["alt"][j], cam.footprint(s["alt"][j]),
                             frame_s=float(t[1] - t[0]) if len(t) > 1 else 0.5)
            cov_i[0] += 1
        # heat map (until found)
        heat[:] = np.nan
        if found_t is None or tk < found_t:
            dens = {rid: prior.get(rid, 0.0) / max(1, len(g["missed"])) * g["missed"] for rid, g in cov.grids.items()}
            top = max((float(v.max()) for v in dens.values() if len(v)), default=1.0) or 1.0
            for rid, (iy, ix) in hidx.items():
                heat[iy, ix] = np.sqrt(dens[rid] / top)
        him.set_data(heat)
        # zones
        active = [s for s in storms if s.active(tk)]
        for zid, (p, fc, lbl, st) in zp.items():
            r = world.regions[zid]
            if r.kind == "no_fly":
                st.set_text("NO-FLY")
                st.set_color("#ffb3b3")
                continue
            closed = not zone_state(zid, tk)
            fire = closed and zid in burning and burning[zid] <= tk
            caution = not closed and any(s.radius <= r.shape.distance(Point(s.center(tk))) < s.radius + caution_m
                                         for s in active)
            if fire:
                p.set_facecolor(FIRE)
                p.set_alpha(0.45 + 0.1 * math.sin(tk * 3 + hash(zid) % 7))
                p.set_hatch("xx")
                st.set_text("CLOSED: FIRE")
                st.set_color("#ffc2b0")
            elif closed:
                p.set_facecolor(STORM)
                p.set_alpha(0.42)
                p.set_hatch("xx")
                st.set_text("CLOSED: STORM")
                st.set_color("#c9d8ff")
            else:
                p.set_facecolor(fc)
                p.set_alpha(0.55)
                p.set_hatch(None)
                st.set_text("CAUTION" if caution else "")
                st.set_color(ACCENT)
            p.set_edgecolor(ACCENT if caution else "#cfe8c8")
            p.set_linewidth(1.8 if caution else 0.9)
        for s in storms:
            vis = s.active(tk)
            storm_patches[s.id].set_visible(vis)
            storm_lbl[s.id].set_text("storm cell" if vis else "")
            if vis:
                c = s.center(tk)
                storm_patches[s.id].center = c
                storm_lbl[s.id].set_position((c[0], c[1] + s.radius + 4))
        # drones
        for did in ids:
            s = tracks[did]
            lo = max(0, i - 300)
            trails[did].set_data(s["east"][lo: i + 1], s["north"][lo: i + 1])
            e, n, a = s["east"][i], s["north"][i], s["alt"][i]
            j = max(0, i - 2)
            hd = math.degrees(math.atan2(n - s["north"][j], e - s["east"][j])) - 90 if i > 2 else 0
            icons[did].set_data([e], [n])
            icons[did].set_marker((3, 0, hd))
            fps_c[did].center = (e, n)
            fps_c[did].set_radius(max(0.5, cam.footprint(a)))
            fps_c[did].set_visible(s["landed"][i] < 0.5)
            dl[did].set_position((e + 6, n + 6))
            mine = [d for d in decisions if (d.get("drone") or "") == did]
            done = [d for d in mine if d["t_apply"] <= tk and d["tool"] != "keep"]
            pending = [d for d in mine if d["t_obs"] <= tk < d["t_apply"]]
            bat = s["battery"][i]
            bars = int(round(bat / 10))
            cd = cards[did]
            cd["stat"].set_text(f"{'█' * bars}{'░' * (10 - bars)} {bat:3.0f}%  alt {a:3.0f} m  "
                                f"{len([d for d in mine if d['t_apply'] <= tk])} decisions")
            if done:
                d = done[-1]
                args = ", ".join(f"{k}={round(v, 1) if isinstance(v, float) else v}" for k, v in d["args"].items())
                cd["call"].set_text(textwrap.shorten(f"▶ {d['tool']}({args})", 64))
                lat = d.get("latency_s")
                cd["when"].set_text(f"decided T+{int(d['t_obs'] // 60):02d}:{int(d['t_obs'] % 60):02d}"
                                    + (f" · thought {lat:.1f} s" if lat is not None else "")
                                    + f" · applied T+{int(d['t_apply'] // 60):02d}:{int(d['t_apply'] % 60):02d}"
                                    + ("   ·   ⋯ thinking about the next call" if pending else ""))
                cd["why"].set_text("why:  " + "\n      ".join(textwrap.wrap(d["reason"] or "", 74)[:2]))
                th = d.get("thinking") or ""
                nl = 3 if len(ids) > 1 else 5
                cd["think"].set_text(("thinking:  " + "\n           ".join(textwrap.wrap(th, 72)[:nl])) if th else "")
            else:
                cd["call"].set_text("⋯ on the pad" if not pending else "⋯ thinking about the first call")
        # hiker, kit, helicopter
        if found_t is not None and tk >= found_t:
            ee, nn, sg = ep.sig["est_east"][i], ep.sig["est_north"][i], ep.sig["est_sigma"][i]
            est_ring.set_visible(True)
            est_ring.center = (ee, nn)
            est_ring.set_radius(max(4.0, sg) + 3 * (1 + math.sin(tk * 4)))
            person_lbl.set_position((ee + 10, nn - 4))
            person_lbl.set_text("hiker")
        landed = [e for e in kit_events if e["t"] <= tk]
        if landed:
            pts = [tuple(map(float, re.findall(r"\(([-\d.]+), ([-\d.]+)\)", e["text"])[0])) for e in landed]
            kit_dots.set_data([p[0] for p in pts], [p[1] for p in pts])
        if help_t is not None and person is not None and help_t - 60 <= tk:
            f = min(1.0, (tk - (help_t - 60)) / 60)
            hx, hy = ext[0] + (person[0] - ext[0]) * f, person[1] + 60 * (1 - f)
            heli.set_data([hx], [hy])
            heli_lbl.set_position((hx + 9, hy + 6))
            heli_lbl.set_text("rescue helicopter" if f < 1 else "help has arrived")
        clock.set_text(f"T+{int(tk // 60):02d}:{int(tk % 60):02d}")
        if help_t is not None and tk >= help_t:
            ph = "HELP ON SCENE"
        elif any(e["t"] <= tk for e in kit_events):
            ph = "KIT DELIVERED · WAITING FOR HELP"
        elif found_t is not None and tk >= found_t:
            ph = "FOUND · FETCHING THE KIT"
        else:
            ph = "SEARCHING"
        phase.set_text(ph)
        # spec rows
        cur = [r for r in online if r["t"] <= tk]
        iv = {}
        if cur:
            for c in cur[-1]["clauses"]:
                iv[c["clause"]] = (c["lo"], c["hi"], c["verdict"])
            for c, v in cur[-1].get("decided", {}).items():
                iv[c] = (v["lo"], v["hi"], v["verdict"])
        end = tk >= t[-1] - 1
        for name, (ft, vt) in spec_rows.items():
            if end and name in final:
                r = final[name]
                vd, lo, hi = ("satisfied" if r["satisfied"] else "violated"), r["robustness"], r["robustness"]
            else:
                lo, hi, vd = iv.get(name, (None, None, "undecided"))
            who = ""
            if name in tight and vd == "undecided":
                past = [m for (tt, m) in tight[name] if tt <= tk]
                if past:
                    who = f" {past[-1]}"
            if vd == "satisfied":
                vt.set_text(f"ρ = {fmt(lo)}  ✓")
                vt.set_color(GOOD)
            elif vd == "violated":
                vt.set_text(f"ρ = {fmt(hi)}  ✗")
                vt.set_color(BAD)
            elif lo is None:
                vt.set_text("")
            else:
                vt.set_text(f"ρ ∈ [{fmt(lo)}, {fmt(hi)}]" + (f" · {who.strip()}" if who else ""))
                vt.set_color(QUIET)
        # caption: the latest radio / zone event, held 25 s
        rec = [e for e in radio if e["t"] <= tk and tk - e["t"] < 25]
        if rec:
            e = rec[-1]
            who = f"{e['drone']} · " if e.get("drone") and len(ids) > 1 else ""
            cap = {"zone": "zone", "message": "radio", "news": "radio", "found": "found", "kit_landed": "kit",
                   "help": "help", "reflex": "safety"}.get(e["kind"], e["kind"])
            caption.set_text(textwrap.shorten(f"{cap}  T+{int(e['t'] // 60):02d}:{int(e['t'] % 60):02d}  {who}{e['text']}", 110))
        else:
            caption.set_text("")
        cap_bg.set_visible(bool(caption.get_text()))

    _ffmpeg()
    writer = FFMpegWriter(fps=fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(out), dpi):
        for tk in times:
            draw(float(tk))
            writer.grab_frame()
    plt.close(fig)
    return out


def still(ep_dir, world, t: float, out) -> Path:
    """One frame of the film at sim time t, as a PNG (for review)."""
    import matplotlib.pyplot as _plt
    ep_dir = Path(ep_dir)
    tmp = Path(out).with_suffix(".mp4")
    # render a one-frame film is wasteful; reuse render with a frame list instead
    global _frame_times
    keep = _frame_times
    try:
        _frame_times = lambda ep, s, f, slow=4.0: np.array([t])  # noqa: E731
        render(ep_dir, world, tmp, seconds=1, fps=1)
    finally:
        _frame_times = keep
    import subprocess

    from .replay2d import matplotlib as _m
    subprocess.run([_m.rcParams["animation.ffmpeg_path"], "-y", "-loglevel", "error", "-i", str(tmp),
                    "-frames:v", "1", str(out)], check=True)
    tmp.unlink()
    _plt.close("all")
    return Path(out)
