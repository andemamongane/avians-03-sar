# avians_env: the world (owner: Taylor)

Everything a mission flies in, described in one YAML file
(`worlds/<name>/world.yaml`), and the world's behaviour over time. Other
packages only read it, so this file is the contract.

```python
from avians_env import WorldSpec, WorldState
world = WorldSpec.load("worlds/sar_valley")      # static: regions, obstacles, stations, ...
scenario = world.sample(seed=3)                  # one draw: person position/motion, help delay
state = WorldState(scenario)                     # running: zone safety, help, person now
state.zone_safe("zone_fire", t=500)              # False after 420 s in sar_valley
state.report_found(t, east, north)               # the drone's report dispatches help
state.signals(t)                                 # help, help_dispatched, unsafe_<zone>
world.describe(); state.status(t)                # the words the pilot reads
```

## The world file

| Key | What |
|---|---|
| `regions` | `id`, `kind` (search, no_fly, hazard, landing, other), `polygon` [[e, n], ...], optional `altitude` [floor, ceiling], `name`, `description` (read by the pilot) |
| `obstacles` | solid prisms: `polygon` or `circle {center, radius}`, `height` |
| `stations` | `kind` charge (`charge_rate_pct_s`) or swap (`swap_s`), `position` |
| `depots` | where kits are loaded: `supplies`, `load_s` |
| `person` | `region_weights` (the prior: where they probably are), `motion` (static, random_walk + `speed`), `description` |
| `decoys` | person-like objects: `position`, `strength` (0..1) |
| `dynamics.zone_safety` | the timeline: `{t, zone, safe, note}` (time-varying predicates in the spec read `unsafe_<zone>`) |
| `dynamics.help` | delay from the report to arrival (mean, std), and the penalty for a wrong position |
| `wind`, `bounds`, `max_alt`, `origin` | mean wind and gust std; the operating area (geofence); lat/lon of the launch point |

## Families (what varies between episodes)

A `family:` block declares ranges; `WorldSpec.sample(seed)` draws one member
into the `Scenario`: the zone-safety timeline, region descriptions, the
person's prior, motion and position, decoys, wind, the help delay, and news
(ground-station messages during the mission that may rewrite a region's
description). `WorldState` reads the scenario, not the file, so the pilot
sees that member's world. A world file can `extends:` another and override
keys: `worlds/sar_valley_family` is the benchmark valley as a family;
`worlds/sar_valley` stays one fixed environment. The format is documented in
`family.py`.

## What exists (a working reference; change it freely)

- `world.py`: loading and validation, the scenario draw (seeded), the
  running state, and the text descriptions.
- `gazebo.py`: SDF models spawned into AAS's running world: regions
  (translucent), obstacles (solid), pads, and the person (a red box).
- `worlds/sar_valley/world.yaml`: the benchmark world.
- `tests/test_world.py`.

## Your tasks

1. **The Gazebo world** (`gazebo.world_sdf`): a complete SDF world for a
   WorldSpec. Include terrain or a textured ground, trees as obstacles, and
   the person as a real human model (Gazebo Fuel's `rescue_randy_sitting`,
   as in AAS's worlds), so the camera and YOLO see something realistic. Keep
   the coordinates identical to the YAML. Wire it in through AAS's `WORLD`
   (see how AAS's `simulation_worlds/` are loaded).
2. **Dynamics you can see:** show zone closures in Gazebo (recolour or add a
   visual), and move a walking person (`random_walk`) in Gazebo along
   `Scenario.person_track`.
3. **The payload in Gazebo:** a kit model attached to the x500 with Gazebo's
   DetachableJoint system, and released on a topic. `avians_sim.gazebo`
   currently only places a kit where the drop model says it landed.
4. **Variety:** the family mechanism exists (above). Extend it with layout
   variation: region shapes and positions, obstacle placement, extra stations,
   drawn per seed, so that rollouts cover more than one map. Keep
   `sar_valley` itself fixed: it is the benchmark.
5. **Descriptions:** make region and hazard descriptions useful to a pilot
   (terrain, visibility, where a hiker would go). The pilot reads
   `describe()` and `status()` at every decision.

## Rules for changes

- **Names are a contract.** Keep `WorldSpec.load`, `.regions`,
  `.search_regions()`, `.hazard_zones()`, `.sample()`, `Region.robustness()`,
  `WorldState.zone_safe()`, `.changes()`, `.report_found()`, `.signals()`,
  `.status()`, or change them together with their users (grep the
  repository).
- **The same seed gives the same episode.** All randomness goes through the
  scenario's generator.
- **Run `python -m pytest` at the repo root** before pushing.
