# The action space

Generated from the skill registry by `python -m avians_skills.docs`; do not edit by hand.
Every tool also takes `reason` (one sentence) and `notes` (the pilot's memory). `keep` lets
the running skill continue.

## movement

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `emergency_stop` (runs) | - | Brake now and hover in place (stops whatever was running). | jerk-limited brake to a hold | benchmark |
| `fly_waypoints` (runs) | waypoints, speed? | Fly through a list of waypoints in order (each with optional alt), planning around no-fly zones, closed zones and obstacles between them; stops at the last one. | as goto, leg by leg | benchmark |
| `goto` (runs) | east?, north?, place?, alt?, speed? | Fly to a point or a named place (station, depot, region centre, 'launch') on the shortest path around no-fly zones, closed zones and obstacles. | visibility graph + A* (exact shortest path among polygons); Ruckig jerk-limited reference; pure pursuit | benchmark |
| `hover` (runs) | seconds?=10.0 | Hold position for some seconds (0: until the next command). | position hold | benchmark |
| `land` (runs) | place? | Land where you are (or at a named pad with place). | vertical reference; AAS land action in Gazebo | benchmark |
| `orbit` (runs) | around? ∈ {point, person}="point", east?, north?, radius?=20.0, alt?, speed?=5.0, laps?=0.0 | Circle a point (or the person's current estimate) at a radius, altitude and speed; laps 0 circles until replaced. | circular carrot with tangential feed-forward | benchmark |
| `path_plan` (instant) | east?, north?, place?, alt? | Plan a route to a point or place without flying it: returns its length, time and battery cost, and whether anything blocks it. Free: use it to compare options. | visibility graph + A*; the battery model | benchmark |
| `precision_land` (runs) | place | Land exactly on a named pad (station, depot or 'launch'), aligning over it first. | transit + vertical reference | benchmark |
| `rtl` (runs) | - | Return to the launch point at cruise altitude and land. | as precision_land to the launch point | benchmark |
| `set_speed` (instant) | speed?, alt? | Change the cruise speed (and optionally cruise altitude) used by later commands. | changes the envelope the planners use | benchmark |
| `takeoff` (runs) | alt?=30.0 | Arm and climb straight up to the given altitude. | vertical reference | benchmark |
| `wait_until` (runs) | condition ∈ {help_arrived, found, time, battery_below}, value?=0.0, timeout_s?=900.0 | Hold position (or stay on the ground) until something happens: help arrives, the person is found, a time, or the battery drops below a value. | position hold / idle on the ground | benchmark |

## search

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `inspect` (runs) | east?, north?, alt?=15.0, seconds?=8.0 | Fly low over the person estimate (or a point) and hover there to look closely: raises confidence in a real person, lowers it for a decoy. | hover over the estimate; the tracker's log-odds update | benchmark |
| `probability_map` (instant) | - | Where the person probably is now, per search region: the prior updated by what the cameras have covered (search theory). Free. |  | benchmark |
| `scan` (runs) | east, north, east2, north2, alt?=15.0, speed?=4.0 | Scan a straight line (a road, a creek, a ridge) at low altitude and slow speed with the camera: from (east, north) to (east2, north2). | straight low-altitude leg | benchmark |
| `search_around` (runs) | around? ∈ {person, point}="person", east?, north?, radius?, alt?=25.0, speed? | Datum search: an expanding square around the person estimate (or a point) out to a radius, for a person who was seen and then lost (their estimate has drifted), or a tip about a place. Lane spacing from the camera footprint at the altitude. |  | benchmark |
| `search_region` (runs) | region, pattern? ∈ {lawnmower, crosshatch, expanding_square, sector}="lawnmower", alt?=30.0, overlap?=0.2, speed?, datum_east?, datum_north? | Search a region with a standard pattern: lawnmower (a lap around the edge, then parallel tracks over the whole region), crosshatch (lawnmower twice, the second pass perpendicular: slower, best detection), expanding_square or sector (around the region centre or a datum). Lane spacing comes from the camera footprint at the altitude. Finishes when the pattern is flown; a sighting does not stop it (you decide). | IAMSAR parallel track (boustrophedon on the minimum rotated rectangle), expanding square, sector; crosshatch = two perpendicular passes | benchmark |

## targets

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `capture_image` (instant) | note?="" | Take a photo with the camera and get what the detector sees in it. | the detector on one frame | benchmark |
| `drop_marker` (instant) | label, east?, north? | Mark a spot on the shared map (here, or at a point) with a label, for later or for the crew. | shared map annotation | benchmark |
| `intercept` (runs) | alt?=20.0, speed? | Fly to where a moving person will be (constant-velocity intercept from the tracker's estimate), re-aiming every few seconds; finishes within 5 m of them. | constant-velocity intercept (closed form), re-aimed every 3 s | benchmark |
| `track_person` (runs) | mode? ∈ {orbit, follow}="orbit", standoff?=10.0, alt?=25.0, speed?=4.0, until? ∈ {none, help_arrived}="none" | Stay with the person: orbit them at a standoff radius (mode orbit) or hover over them (mode follow), following the estimate as it updates. With until=help_arrived it finishes when help arrives; otherwise it runs until replaced. | Kalman estimate + orbit / follow carrot | benchmark |

## payload

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `deliver` (runs) | target? ∈ {person, point}="person", east?, north?, drop_alt?=8.0, max_sigma?=8.0 | Drop the kit next to the person: fly over the estimate (or a point), descend to the drop altitude, hold at the computed release point (ballistic drop with drag and the estimated wind), release, climb back. Refuses when the estimate is too uncertain: inspect first. | ballistic drop with quadratic drag and the wind estimate (release point by fixed-point iteration) | benchmark |
| `pick_up` (runs) | depot?="base" | Fly to a depot, land, load a medical kit (takes the depot's load time), take off again. | transit, land, depot load time | benchmark |

## energy

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `recharge` (runs) | station, until_pct?=95.0 | Fly to a charging pad, land, charge to a target level, take off again. | land on the pad; the battery model's charge rate | benchmark |
| `swap_battery` (runs) | station | Fly to a battery swap station, land, get a full battery (fixed time), take off again. | land on the pad; the station's swap time | benchmark |

## comms

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `escalate` (instant) | reason | Hand this decision to the larger remote model (onboard pilots only): use when the situation is unfamiliar or the stakes are high. | hands the decision to the remote model | student |
| `report` (instant) | kind ∈ {person_found, status, hazard}, message?="", east?, north? | Radio a report to the ground station. kind person_found dispatches the rescue crew to the given position (default: the current estimate): a wrong position delays them. The acknowledgement (and the crew's ETA) comes back as a message. | ground station message; dispatches help | benchmark |
| `request_help` (instant) | question | Ask the operator at the ground station a question; the answer, if any, arrives later as a message (they may be slow, or not answer at all). | ground station question | benchmark |

## swarm

| tool | parameters | what it does | built on | set |
|---|---|---|---|---|
| `assign_tasks` (instant) | regions? | Split search regions between all drones at minimum total flight distance (Hungarian method); records the claims and radios each drone its region. | Hungarian method (scipy linear_sum_assignment) (swarm) | swarm |
| `claim_region` (instant) | region | Claim a search region as yours (the other drones see claims); release with region ''. |  | swarm |
| `formation` (runs) | leader, role? ∈ {relay, wing}="wing" | Hold a position relative to another drone until replaced: relay (20 m behind and 10 m above it, a radio relay) or wing (20 m beside it). | offset setpoints relative to a leader (swarm) | swarm |
| `handoff` (instant) | to, task | Ask another drone, by radio, to take over a task (e.g. tracking the person); it decides. | transfers the running skill (swarm) | swarm |
