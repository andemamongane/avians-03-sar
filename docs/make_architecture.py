"""Draws docs/architecture.png and docs/architecture.svg: the planned autonomy
architecture (System One router vs onboard SLM vs cloud LLM). python docs/make_architecture.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BG, INK, QUIET = "#0f1419", "#e6e9ec", "#8b96a3"
ONB, CLOUD, LEARN = "#14202b", "#1d1830", "#172219"
C_ROUTER, C_SLM, C_LLM, C_SAFE, C_SKILL, C_PX4 = "#ffb020", "#5cc8ff", "#c084fc", "#ff6b6b", "#3ccf7a", "#cfd8dc"

fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def panel(x, y, w, h, fc, title, sub=""):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="#2c3a47", lw=1.2))
    ax.text(x + 1.5, y + h - 2.2, title, color=QUIET, fontsize=11, fontweight="bold", va="top")
    if sub:
        ax.text(x + 1.5, y + h - 5.0, sub, color=QUIET, fontsize=8.6, va="top")


def box(x, y, w, h, col, title, lines=(), badge=None, title_fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=col + "1f", ec=col, lw=1.8))
    ax.text(x + 1.4, y + h - 1.6, title, color=INK, fontsize=title_fs, fontweight="bold", va="top")
    for k, ln in enumerate(lines):
        ax.text(x + 1.4, y + h - 4.6 - 2.45 * k, ln, color="#c7cfd6", fontsize=8.7, va="top")
    if badge:
        ax.text(x + w - 1.2, y + h - 1.6, badge, color=col, fontsize=9, fontweight="bold", va="top", ha="right")


def arrow(p, q, col=QUIET, text=None, tpos=None, rad=0.0, lw=1.6, ls="-", fs=8.6, tcol=None, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=col, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        x, y = tpos or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.2)
        ax.text(x, y, text, color=tcol or col, fontsize=fs, ha=ha, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.85))


# ---- title
ax.text(3, 87, "AVIANS autonomy: who decides, and how fast", color=INK, fontsize=20, fontweight="bold", va="top")
ax.text(3, 82.6, "At every decision point a fast router picks the cheapest decision-maker that is good enough; "
        "a safety layer below every one of them keeps the drone inside the envelope.", color=QUIET, fontsize=10.5, va="top")

# ---- panels
panel(2, 22, 104, 57, ONB, "ON THE DRONE", "onboard computer (e.g. Jetson-class): works without a link")
panel(109, 22, 49, 57, CLOUD, "IN THE CLOUD", "needs the radio / LTE link")
panel(2, 2, 156, 18, LEARN, "THE LEARNING LOOP (offline, between flights)")

# ---- onboard boxes
box(5, 56, 25, 13, QUIET, "Decision point", ["an event (skill done, person found,", "zone closed, radio message, reflex)", "or a timer (every 45 s)"])
box(36, 54, 30, 17, C_ROUTER, "System One router", ["JEV-type decision model: Kev 0.8B or", "Laya 0.4B, fine-tuned on our labels", "(Jev through its API as the reference)",
                                                    "answers one typed question:", "commit / local / escalate"], badge="~50 ms")
box(73, 54, 30, 17, C_SLM, "Onboard SLM pilot", ["the distilled student (~1-4B)", "reads the full decision prompt:", "state, memory, spec (SFO), world",
                                                   "returns one tool call", "may itself call escalate"], badge="~0.5-1 s")
box(5, 40, 25, 12, QUIET, "commit", ["no model call: the running", "skill continues (most", "decision points)"], badge="0 s")
box(36, 25, 30, 25, C_SKILL, "Skill library", ["one tool call runs for minutes:", "search_region, deliver, track_person,", "recharge, report, assign_tasks, ...",
                                                 "built on standard algorithms:", "A* on a visibility graph, IAMSAR", "patterns, Kalman tracking, Ruckig,", "ballistic drop, Hungarian assignment"],
    badge="10 Hz")
box(73, 36, 30, 12, C_SAFE, "Safety layer", ["pre-check: rejects bad calls, says why", "reflex: brakes before no-fly zones,", "leaves closed zones, lands on low battery"], badge="10 Hz")
box(73, 25, 30, 9.5, C_PX4, "PX4 autopilot (via AAS)", ["geofence and failsafes enforced in firmware"], title_fs=10.5)
box(5, 25, 25, 12, "#7aa2ff", "Online monitor", ["checks the mission spec (STL/SFO)", "live; ρ goes into every prompt,", "a broken promise wakes the pilot"])

# ---- cloud boxes
box(112, 51, 43, 20, C_LLM, "Cloud LLM pilot", ["the teacher: Claude Haiku 5.5 today", "(larger models for the hardest calls)", "same prompt, same tools as the SLM",
                                                  "handles novel or high-stakes situations:", "unfamiliar hazards, conflicting promises,", "swarm re-planning"], badge="~2-6 s")
box(112, 31, 43, 14, "#9aa5b1", "Ground station", ["operator: tasking, answers, news", "rescue crew: dispatch and ETA", "shared state between the drones"])
ax.text(113.5, 27.0, "If the link drops, the router can only choose", color=QUIET, fontsize=8.6)
ax.text(113.5, 24.8, "commit or local: the drone keeps flying safely.", color=QUIET, fontsize=8.6)

# ---- decision flow
def poly(pts, col, text=None, tpos=None, ls="-", fs=8.6):
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-1][0]], list(ys) + [pts[-1][1]], color=col, lw=1.6, ls=ls, solid_capstyle="round")
    arrow(pts[-2], pts[-1], col, ls=ls)
    if text:
        ax.text(*tpos, text, color=col, fontsize=fs, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none", alpha=0.9))


arrow((30, 62.5), (36, 62.5), QUIET)
arrow((51, 54), (17.5, 52), C_ROUTER, "commit", tpos=(40, 52.6), rad=-0.12)
arrow((66, 62.5), (73, 62.5), C_ROUTER, "local", tpos=(69.5, 64.6))
poly([(55, 71), (55, 75.2), (128, 75.2), (128, 71)], C_ROUTER, "escalate", tpos=(91, 75.2))
arrow((103, 58), (112, 58), C_SLM, "escalate", tpos=(107.5, 60), fs=8)
arrow((86, 54), (60, 50), C_SLM, "tool call", tpos=(76, 53.6), rad=0.05)
arrow((112, 53), (64, 50), C_LLM, "tool call", tpos=(106, 50.6), rad=-0.12)
arrow((66, 40), (73, 40), C_SKILL, "setpoints", tpos=(69.5, 43.2), fs=8)
arrow((88, 36), (88, 34.6), C_SAFE)
arrow((112, 38), (103, 41), "#9aa5b1", "radio", tpos=(108, 41.6), fs=8)
poly([(5, 31), (3.6, 31), (3.6, 62.5), (5, 62.5)], "#7aa2ff", ls="--")
ax.text(3.6, 38.5, "ρ, warnings", color="#7aa2ff", fontsize=8, rotation=90, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc=ONB, ec="none"))

# ---- learning loop
steps = [(5, "Record", "every decision as ADP: prompt,", "tool call, latency, outcome"),
         (43, "Distill", "teacher decisions become SFT", "data: cloud LLM -> onboard SLM"),
         (81, "Label the router", "commit / local / escalate from", "SLM-teacher agreement and outcomes"),
         (119, "Reinforce", "RL with STL robustness as the", "reward, in fast simulation")]
for x, t, a, b in steps:
    box(x, 4, 34, 11.5, "#3ccf7a", t, [a, b], title_fs=10.5)
for k in range(3):
    arrow((steps[k][0] + 34, 9.7), (steps[k + 1][0], 9.7), "#3ccf7a")
ax.text(156, 17.8, "each round: a better SLM and router, so more decisions stay onboard and fewer go to the cloud",
        color="#9fe3b7", fontsize=9, ha="right", va="top")

out = Path(__file__).parent
fig.savefig(out / "architecture.png", dpi=120, facecolor=BG)
fig.savefig(out / "architecture.svg", facecolor=BG)
print(out / "architecture.png")
