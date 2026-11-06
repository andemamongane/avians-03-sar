# avians_skills: the tools the pilot calls

Each skill is one tool call that runs for many control ticks, producing
setpoints, until it succeeds, fails or is replaced. Skills don't know which
simulator they're in. `base.py` defines the contracts: `VehicleState`,
`Setpoint`, `Vehicle`, `Limits`, `Context`, `Skill`, and `create` / `tools`
for the JSON-schema tool definitions.

| Skill | Built on |
|---|---|
| `goto`, and every transit inside other skills | visibility graph + A* (exact shortest paths among polygons), re-derived from the world at that moment (closed zones); replans when a zone closes |
| path following | pure pursuit with corner- and end-braking speed limits; the vehicle's controller (PX4's, or the fast tier's) does the rest |
| `search_region` | IAMSAR patterns: boustrophedon "parallel track" (lanes along the minimum rotated rectangle, spaced by the camera footprint), expanding square, sector |
| `inspect`, `track_person` (orbit / follow), `orbit`, `intercept` | the tracker's belief; constant-velocity intercept for moving targets |
| `pick_up`, `deliver` | depot loading; a drag-factored fall time and wind compensation for the drop point; refuses on a loose estimate |
| `recharge`, `swap_battery` | land on the pad; the battery model's charge rate or swap time |
| `report`, `request_help`, `escalate`, `capture_image`, `drop_marker` | instant (radio, operator, router hand-off, camera, map) |
| `takeoff`, `land`, `precision_land`, `rtl`, `hover`, `wait_until`, `set_speed`, `emergency_stop` | basics |
| `handoff`, `formation`, `assign_tasks` | swarm only: `swarm.assign_tasks` (Hungarian method) works; the skills refuse until there is a multi-vehicle backend |

Optional, Linux only (no macOS wheels): OMPL (RRT*) can replace
`planning.PathPlanner` for 3D or kinodynamic planning, and Fields2Cover can
replace `coverage.lawnmower`. Each needs only a thin adapter behind the same
interface.

To add a skill: subclass `Skill` in `library/`, set `name`, `summary`,
`params` (JSON schema, with defaults) and `required`, then implement
`start` and `tick`. It is registered on import and appears in the pilot's
tools (`BENCHMARK_TOOLS`).
