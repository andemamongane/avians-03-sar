# avians_skills: the tools the pilot calls

Each skill is one tool call that runs for many control ticks, producing
setpoints, until it succeeds, fails or is replaced. Skills don't know which
simulator they're in. `base.py` defines the contracts: `VehicleState`,
`Setpoint`, `Vehicle`, `Limits`, `Context`, `Skill`, and `create` / `tools`
for the JSON-schema tool definitions.

| Skill | Built on |
|---|---|
| `goto`, `fly_waypoints`, `path_plan`, and every transit inside other skills | visibility graph + A* (exact shortest paths among polygons), re-derived from the world at that moment (closed zones); a start inside a zone that just closed leaves it by the shortest way first; replans when a zone closes |
| path following | pure pursuit picks the carrot and the speed profile (corner and end braking); `reference.py` turns it into jerk-limited motion with Ruckig, the same reference both tiers fly |
| `search_region`, `scan` | IAMSAR patterns: boustrophedon "parallel track" (lanes along the minimum rotated rectangle, spaced by the camera footprint), crosshatch (two perpendicular passes), expanding square, sector; a straight low scan |
| `inspect`, `track_person` (orbit / follow), `orbit`, `intercept` | the tracker's belief; constant-velocity intercept for moving targets |
| `pick_up`, `deliver` | depot loading; `ballistics.py`: a point mass with quadratic drag in the estimated wind, the release point by fixed-point iteration (the kit's mass and drag area come from the world file); refuses on a loose estimate |
| `recharge`, `swap_battery` | land on the pad; the battery model's charge rate or swap time |
| `report`, `request_help`, `escalate`, `capture_image`, `drop_marker` | instant (radio, operator, router hand-off, camera, map) |
| `takeoff`, `land`, `precision_land`, `rtl`, `hover`, `wait_until`, `set_speed`, `emergency_stop` | basics |
| `claim_region`, `assign_tasks`, `handoff`, `formation` | swarm only (they refuse with one drone): claims shared through the world, Hungarian assignment (scipy) radioed to each drone, a tasking request, relay/wing offsets on a leader |

Optional, Linux only (no macOS wheels): OMPL (RRT*) can replace
`planning.PathPlanner` for 3D or kinodynamic planning, and Fields2Cover can
replace `coverage.lawnmower`. Each needs only a thin adapter behind the same
interface.

The full table, generated from the registry, is `docs/action_space.md`
(`python -m avians_skills.docs > docs/action_space.md`; a test fails when it is stale).

To add a skill: subclass `Skill` in `library/`, set `name`, `summary`,
`params` (JSON schema, with defaults) and `required`, then implement
`start` and `tick`. It is registered on import and appears in the pilot's
tools (`BENCHMARK_TOOLS`).
