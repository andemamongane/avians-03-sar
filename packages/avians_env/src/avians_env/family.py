"""Families of environments: what varies between episodes of one world.

A world file's `family:` block declares ranges; `Family.draw(world, rng)`
samples one member for a seed. Everything not listed stays as the file says,
so a world without a family is one fixed environment (the benchmark).

    family:
      zone_safety:                       # per hazard zone: when it closes, for how long
        zone_fire: {p_close: 0.8, t_close: [150, 700], duration: [240, 900], p_reopen: 0.4,
                    notes: ["Fire crews closed the zone", "Smoke over the ridge; crews closed it"]}
      descriptions:                      # per region: alternative texts (one is drawn)
        r2: ["Rocky creek bed; the phone's last ping.", "The creek is in spate after the rain; ..."]
      person:
        region_weights: {dirichlet: 3.0} # resample the prior around the file's (concentration)
        motion: {static: 0.7, random_walk: 0.3}
        speed: [0.2, 0.6]
      decoys: {count: [1, 4], strength: [0.2, 0.5]}   # placed in search regions
      wind: {speed: [0, 5], gust_std: [0, 1.5]}
      help: {delay_mean_s: [420, 780]}
      news:                              # ground-station messages during the mission
        - {p: 0.5, t: [120, 500], about: zone_fire, text: "Rangers report smoke drifting over {name}"}
        - {p: 0.4, t: [60, 400], about: person_region, text: "The hiker's phone pinged once near {name}",
           description: "{old} A phone ping was reported here."}

A news item is delivered at time t (the ground station, avians_sim) and may
rewrite the region's description from then on (`description`, with `{old}`
for the previous text). `about: person_region` names the region the person is
actually in: a weak hint, like a real tip. Ranges `[a, b]` are uniform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .world import Decoy, SafetyEvent, Wind, WorldSpec


@dataclass
class News:
    t: float
    about: str              # region id, or ""
    text: str
    description: str = ""   # new description for the region, "" = unchanged


@dataclass
class Draw:
    """One member of the family: the overrides for a Scenario."""
    timeline: list[SafetyEvent]
    descriptions: dict[str, str]
    decoys: list[Decoy]
    wind: Wind
    help_delay_mean_s: float
    region_weights: dict[str, float]
    motion: str
    speed: float
    news: list[News] = field(default_factory=list)

    def summary(self) -> dict:
        return {"timeline": [vars(e) for e in self.timeline], "descriptions": self.descriptions,
                "decoys": [{"id": d.id, "position": d.position, "strength": d.strength} for d in self.decoys],
                "wind": vars(self.wind), "help_delay_mean_s": self.help_delay_mean_s,
                "region_weights": self.region_weights, "motion": self.motion, "speed": self.speed,
                "news": [vars(n) for n in self.news]}


def _u(rng, rng_or_value):
    if isinstance(rng_or_value, (list, tuple)):
        lo, hi = float(rng_or_value[0]), float(rng_or_value[1])
        return float(rng.uniform(lo, hi))
    return float(rng_or_value)


@dataclass
class Family:
    spec: dict

    @classmethod
    def from_dict(cls, d: dict | None) -> "Family | None":
        return cls(dict(d)) if d else None

    def draw(self, w: WorldSpec, rng: np.random.Generator, horizon_s: float, person_region: str) -> Draw:
        f = self.spec
        # zone safety
        timeline = list(w.safety_timeline)
        zs = f.get("zone_safety", {})
        if zs:
            timeline = [e for e in timeline if e.zone not in zs]
            for zid, z in zs.items():
                if zid not in w.regions:
                    continue
                if rng.random() < float(z.get("p_close", 1.0)):
                    t0 = _u(rng, z.get("t_close", [0, horizon_s / 2]))
                    notes = z.get("notes") or [""]
                    note = notes[int(rng.integers(len(notes)))]
                    timeline.append(SafetyEvent(round(t0, 1), zid, False, note))
                    if "duration" in z and rng.random() < float(z.get("p_reopen", 0.0)):
                        timeline.append(SafetyEvent(round(t0 + _u(rng, z["duration"]), 1), zid, True,
                                                    "crews reopened the zone"))
            timeline.sort(key=lambda e: e.t)
        # descriptions
        descriptions = {rid: r.description for rid, r in w.regions.items()}
        for rid, options in f.get("descriptions", {}).items():
            if rid in descriptions and options:
                descriptions[rid] = str(options[int(rng.integers(len(options)))])
        # person
        pf = f.get("person", {})
        weights = dict(w.person.region_weights) or {r.id: 1.0 for r in w.search_regions()}
        if "region_weights" in pf and "dirichlet" in pf["region_weights"]:
            conc = float(pf["region_weights"]["dirichlet"])
            ids = sorted(weights)
            base = np.array([weights[i] for i in ids]) / sum(weights.values())
            drawn = rng.dirichlet(np.maximum(base * conc * len(ids), 0.05))
            weights = {i: float(v) for i, v in zip(ids, drawn)}
        motion = w.person.motion
        if "motion" in pf:
            opts = pf["motion"]
            names = sorted(opts)
            p = np.array([float(opts[n]) for n in names])
            motion = names[int(rng.choice(len(names), p=p / p.sum()))]
        speed = _u(rng, pf["speed"]) if "speed" in pf else w.person.speed
        # decoys
        decoys = list(w.decoys)
        df = f.get("decoys")
        if df:
            n = int(rng.integers(int(df["count"][0]), int(df["count"][1]) + 1))
            decoys = []
            regions = w.search_regions()
            for k in range(n):
                r = regions[int(rng.integers(len(regions)))]
                x0, y0, x1, y1 = r.shape.bounds
                for _ in range(100):
                    e, n_ = rng.uniform(x0, x1), rng.uniform(y0, y1)
                    if r.contains(e, n_):
                        break
                decoys.append(Decoy(f"decoy_{k}", (float(e), float(n_)), _u(rng, df.get("strength", 0.3))))
        # wind
        wind = w.wind
        wf = f.get("wind")
        if wf:
            s, th = _u(rng, wf.get("speed", [0, 0])), rng.uniform(0, 2 * math.pi)
            wind = Wind(round(s * math.cos(th), 2), round(s * math.sin(th), 2),
                        round(_u(rng, wf.get("gust_std", 0.0)), 2))
        help_mean = _u(rng, f["help"]["delay_mean_s"]) if "help" in f and "delay_mean_s" in f["help"] \
            else w.help.delay_mean_s
        # news
        news = []
        for item in f.get("news", []):
            if rng.random() >= float(item.get("p", 1.0)):
                continue
            about = item.get("about", "")
            if about == "person_region":
                about = person_region
            name = w.regions[about].name if about in w.regions else about
            text = str(item.get("text", "")).format(name=name, id=about)
            desc = str(item.get("description", "")).format(old=descriptions.get(about, ""), name=name)
            news.append(News(round(_u(rng, item.get("t", [0, horizon_s / 2])), 1), about, text, desc))
        news.sort(key=lambda n: n.t)
        return Draw(timeline, descriptions, decoys, wind, help_mean, weights, motion, speed, news)
