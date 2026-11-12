"""Draws the architecture graphics (python docs/make_architecture.py):

    docs/architecture_decisions.png/.svg   1. who makes each decision (router, onboard SLM, cloud LLM)
    docs/architecture_apicall.png/.svg     2. what one pilot API call sends and gets back
    docs/architecture_toolcalls.png/.svg   3. what happens to the tool call: skill, safety, PX4
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BG, INK, QUIET, BODY = "#0f1419", "#e6e9ec", "#9aa5b1", "#cfd6dd"
ROUTER, SLM, LLM, KEEP, SKILL, SAFE, PX4, PROMPT = ("#ffb020", "#5cc8ff", "#c084fc", "#9aa5b1",
                                                    "#3ccf7a", "#ff6b6b", "#cfd8dc", "#7aa2ff")
OUT = Path(__file__).parent


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    ax.text(5, 84, title, color=INK, fontsize=24, fontweight="bold", va="top")
    ax.text(5, 78.5, subtitle, color=QUIET, fontsize=13.5, va="top")
    return fig, ax


def box(ax, x, y, w, h, col, title, lines=(), badge=None, title_fs=17, line_fs=12.5, mono=False, step=3.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=col + "22", ec=col, lw=2.2))
    ax.text(x + 2, y + h - 2.2, title, color=INK, fontsize=title_fs, fontweight="bold", va="top")
    for k, ln in enumerate(lines):
        ax.text(x + 2, y + h - 7.2 - step * k, ln, color=BODY, fontsize=line_fs, va="top",
                family="monospace" if mono else None)
    if badge:
        ax.text(x + w - 1.8, y + h - 2.4, badge, color=col, fontsize=13, fontweight="bold", va="top", ha="right")


def arrow(ax, p, q, col=QUIET, text=None, tpos=None, rad=0.0, ls="-", fs=12.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, color=col, lw=2.4, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3))
    if text:
        x, y = tpos or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(x, y, text, color=col, fontsize=fs, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="none"))


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=120, facecolor=BG)
    plt.close(fig)
    print(OUT / f"{name}.png")


# ------------------------------------------------------------------ 1. who decides
fig, ax = canvas("1. Who makes each decision",
                 "A small, fast model on the drone picks the cheapest decision-maker that is good enough.")
box(ax, 5, 36, 27, 22, KEEP, "Decision point", ["something happened", "(person found, zone closed,",
                                                "skill done, radio message)", "or 45 s have passed"])
box(ax, 42, 36, 30, 22, ROUTER, "Router", ["System One model", "(JEV-type: Kev or Laya)", "on the drone"], badge="~50 ms*")
box(ax, 92, 56, 62, 17, KEEP, "Keep going", ["no model call: the running skill continues", "most decision points end here"],
    badge="0 s")
box(ax, 92, 34, 62, 17, SLM, "Onboard SLM pilot", ["small language model on the drone", "routine decisions; works without a link"],
    badge="~0.5-1 s*")
box(ax, 92, 12, 62, 17, LLM, "Cloud LLM pilot", ["large model over the radio link (Claude today)", "new or high-stakes situations"],
    badge="~2-6 s")
arrow(ax, (32, 47), (42, 47))
arrow(ax, (72, 53), (92, 64.5), ROUTER, "commit", tpos=(80, 61.5))
arrow(ax, (72, 47), (92, 42.5), ROUTER, "local", tpos=(82, 46.8))
arrow(ax, (72, 41), (92, 20.5), ROUTER, "escalate", tpos=(79, 29))
arrow(ax, (148, 34), (148, 29), SLM, "unsure", tpos=(140.5, 31.5), ls="--")
ax.text(5, 24, "No link?", color=INK, fontsize=15, fontweight="bold", va="top")
ax.text(5, 19.5, "The router chooses only between keep going", color=QUIET, fontsize=13.5, va="top")
ax.text(5, 15.5, "and the onboard SLM: the drone keeps flying.", color=QUIET, fontsize=13.5, va="top")
ax.text(5, 5, "* targets. The router and the onboard SLM are trained from recorded cloud-LLM decisions.",
        color=QUIET, fontsize=11.5, va="top")
save(fig, "architecture_decisions")

# ------------------------------------------------------------------ 2. one API call
fig, ax = canvas("2. One pilot call",
                 "Every decision is one stateless request. The pilot sees everything it needs in it, and answers with one tool call.")
ax.text(5, 71, "REQUEST", color=QUIET, fontsize=13, fontweight="bold")
box(ax, 5, 45, 62, 23, PROMPT, "System prompt", ["the pilot's role and the rules of the game", "the mission brief and its spec (STL)",
                                                 "the map: zones, obstacles, pads"],
    badge="same all flight: cached")
box(ax, 5, 26, 62, 16, SKILL, "Tools", ["27 skills (31 with a swarm), plus keep", "every tool also takes reason and notes"],
    title_fs=17)
box(ax, 5, 5, 62, 18, ROUTER, "Decision prompt", ["now: time, position, battery, closed zones",
                                                  "what happened since the last decision",
                                                  "promise margins (ρ) and the pilot's notes"], badge="new each time")
box(ax, 78, 30, 22, 22, LLM, "Pilot model", ["Claude Haiku", "(or the onboard", "SLM, same format)"], title_fs=17)
ax.text(89, 26.5, "effort: medium\nthinking: adaptive", color=QUIET, fontsize=11.5, ha="center", va="top")
for y in (56, 34, 14):
    arrow(ax, (67, y), (78, 41), QUIET)
ax.text(111, 71, "RESPONSE", color=QUIET, fontsize=13, fontweight="bold")
box(ax, 111, 51, 44, 17, KEEP, "Thinking (summary)", ["\"Storm is closing r2; the", "kit is safer via the creek.\""],
    title_fs=16, line_fs=12.5)
box(ax, 111, 18, 44, 29, ROUTER, "Exactly one tool call", ["deliver(", "  target=\"person\",", "  reason=\"person found,",
                                                            "    kit on board\",", "  notes=\"r3 not searched\")"],
    title_fs=16, line_fs=12.5, mono=True)
arrow(ax, (100, 45), (111, 58))
arrow(ax, (100, 38), (111, 33))
ax.text(111, 13.5, "No tool call, or an API error → keep.", color=QUIET, fontsize=12.5, va="top")
ax.text(111, 9.5, "Notes are the pilot's only memory.", color=QUIET, fontsize=12.5, va="top")
save(fig, "architecture_apicall")

# ------------------------------------------------------------------ 3. what the tool call does
fig, ax = canvas("3. From tool call to flight",
                 "The pilot never flies the drone directly. It picks a skill; the skill does the flying.")
box(ax, 5, 46, 30, 18, LLM, "Tool call", ["from the pilot", "search_region(r3)"], title_fs=17)
box(ax, 45, 46, 30, 18, SAFE, "Pre-check", ["rejects bad calls,", "says why"], title_fs=17)
box(ax, 85, 42, 38, 26, SKILL, "Skill", ["runs for minutes on its own:", "search patterns, A* paths,", "tracking, ballistic drop,",
                                         "battery service"], title_fs=17, badge="10 Hz")
box(ax, 133, 46, 23, 18, PX4, "PX4", ["autopilot,", "via AAS"], title_fs=17)
arrow(ax, (35, 55), (45, 55))
arrow(ax, (75, 55), (85, 55), text="ok", tpos=(80, 58.5))
arrow(ax, (123, 55), (133, 55), text="setpoints", tpos=(128, 60.5), fs=11)
arrow(ax, (60, 46), (20, 46), SAFE, "rejected: why", tpos=(40, 41), rad=-0.35, ls="--")
box(ax, 85, 18, 71, 17, SAFE, "Safety reflex (always on)", ["brakes before no-fly zones, leaves closed zones,",
                                                             "keeps drones apart, lands on low battery"],
    title_fs=16, badge="10 Hz")
arrow(ax, (104, 35), (104, 42), SAFE, "overrides", tpos=(111, 38.5), fs=11)
arrow(ax, (104, 68), (20, 64), SKILL, "done · failed · person found · zone closed  →  the pilot's next decision",
      tpos=(62, 72.5), rad=0.12, ls="--", fs=12)
ax.text(5, 26, "One skill runs at a time; a new call replaces it.", color=QUIET, fontsize=13.5, va="top")
ax.text(5, 21.5, "Instant tools (report, set_speed) do not.", color=QUIET, fontsize=13.5, va="top")
save(fig, "architecture_toolcalls")
