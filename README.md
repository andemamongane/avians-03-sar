# AVIANS stage 3: search and rescue (avians-03-sar)

AVIANS teaches small language models to fly drones. A capable LLM (the teacher,
Claude Haiku 5.5) flies missions through **high-level skills**. Every
decision it makes is recorded in the Agent Data Protocol (ADP). That data is
distilled into a small model that can run on the drone. A fast "System One"
router decides, at each decision point, whether to keep going, ask the
onboard model, or escalate to the remote teacher.

Missions are specified in signal temporal logic (STL). Every flight is scored
by its robustness, which is also the reinforcement-learning reward. Funded by
AeroVironment.

Stage 2 (avians-02-aas) flew single waypoints and trajectories. Stage 3 adds:

- **Skills:** search a region, track or intercept the person, pick up and
  deliver a kit, recharge or swap the battery, report, and more. Each is built
  on a standard algorithm; the list is `docs/action_space.md`.
- **The benchmark:** a search-and-rescue mission.
- **Pilot memory.**
- **Decisions under simulated latency**, not lockstep.
- **Recording** of observation-action pairs, ready for training.

## The benchmark: `missions/sar_supply.yaml`

A hiker is missing in `worlds/sar_valley`. The drone must:

1. **Find them.** Until then, every search region must be visited at least
   every 10 minutes.
2. **Report their position.** This dispatches a helicopter about 10 minutes
   out; a wrong position delays it.
3. **Fetch the medical kit** from the base depot and drop it within 5 m of
   them.
4. **Stay within 30 m** until help arrives.

Throughout, it must stay out of the power-line corridor and the fire zone
(which closes at t = 420 s), keep at least 10 % battery, and keep clear of
obstacles. The spec is twelve STL clauses, all shown to the pilot before
takeoff.

| Pilot | Satisfied | Notes |
|---|---|---|
| `scripted` (rule-based baseline) | 40 / 40 seeds | |
| `scripted` on the family (`sar_supply_family`) | 35 / 40 seeds | misses: a walker who wandered into searched ground (3), a walker lost after the drop (2) |
| `haiku` (the teacher, Claude Haiku 5.5) | 1 / 1 (seed 3) | 38 decisions, $0.018, 110 s of wall time |
| `scripted`, two drones (`sar_supply_2drones`) | 36 / 40 seeds | one pilot per drone; the lead fetches the kit, the wingman keeps eyes on the person |

## Layout

```
packages/                    each one installable on its own (pip install -e packages/<name>)
  avians_spec/       STL over named signals, past-time operators (O, H), ROSI online monitor, signal registry
  avians_env/        [Taylor]    the world: regions, hazards, person, stations, depots, timelines; Gazebo models
  avians_perception/ simulated detector, YOLO ground projection, person tracker (belief always defined)
  avians_skills/     the skill library: path planning, coverage, tracking, payload, energy, comms, swarm stubs
  avians_safety/     envelope -> PX4 parameters, explanatory pre-checks, reflex layer (runtime assurance)
  avians_sim/        fast kinematic tier, Gazebo/AAS tier, skill executor, Gazebo recording and replay
  avians_adp/        [McGlashan] Agent Data Protocol recording; SFT and router datasets (to do)
  avians_agent/      the pilot loop: decision points, simulated latency, memory, pilots, router, `avians` CLI
  avians_viz/        2D sped-up replays, QGroundControl .tlog and KML exports
worlds/sar_valley/   the benchmark world (world.yaml); sar_valley_family/: the same map as a family
missions/            sar_supply.yaml: brief + STL spec; sar_supply_family.yaml: the same on the family
configs/             limits (envelope), agent (decision loop), latency (per model), pilots, aas (Gazebo tier)
docker/              the avians container for the Gazebo tier; aas/: avians' patch to AAS (trajectory controller)
scripts/             install.sh, up_remote.sh, shell.sh, aas_plan_ahead.sh
external/            aerial-autonomy-stack (git submodule, the same commit as avians-02)
data/                episodes land here (gitignored)
```

How one episode runs:

```
world.yaml --sample(seed)--> Scenario (person, help delay)
                                  |
 decision point (event, or every 45 s)                 every 0.1 s of sim time
   prompt = spec + world + state + memory  ---------->  Executor: current skill -> setpoint
   pilot (Haiku | scripted | student + router)            -> reflex layer -> vehicle (fast | PX4)
   decision applies after a simulated latency             -> camera -> tracker -> belief
                                  |                        -> world timeline (zones, help)
 ADP trajectory + signals + events + STL score  <-------  signals -> online monitor (ROSI)
```

## Quick start (any machine, no simulator, no ROS)

```bash
git clone --recursive git@github.com:andemamongane/avians-03-sar.git && cd avians-03-sar
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.10
./scripts/install.sh
cp .env.example .env                                     # then paste ANTHROPIC_API_KEY into .env

avians doctor                                            # what is installed
avians check missions/sar_supply.yaml                    # exactly what the pilot sees; nothing flies
avians fly missions/sar_supply.yaml --seed 3 --video     # scripted pilot, ~5 s, plus a 2D replay video
avians fly missions/sar_supply.yaml --pilot haiku --seed 3 --video --qgc   # the teacher (about $0.02)
avians rollouts missions/sar_supply.yaml --seeds 0-99 --workers 8          # a sweep, with a results table
python -m pytest                                         # all packages' tests
```

Each episode is written to `data/episodes/<mission>/<pilot>/s<seed>/`:

| File | What |
|---|---|
| `trajectory.json` | the ADP trajectory: every decision prompt, the pilot's text and thinking summary, each tool call (with latency, tokens, state snapshot), each result |
| `signals.jsonl`, `events.jsonl` | every spec signal at 2 Hz; every event (skill done, found, zone closed, reflex, ...) |
| `score.json`, `summary.json`, `stl_online.jsonl` | final robustness per clause; outcome, cost, timings; live ROSI verdicts |
| `system_prompt.txt`, `meta.json` | the prompt, the scenario (true person position, help delay), pilot, latency models |
| `replay.mp4`, `summary.png` | with `--video`: the sped-up 2D replay |
| `flight.tlog`, `world.kml`, `flight.kml` | with `--qgc`: QGroundControl replay and map overlay |
| `gazebo.mp4`, `gazebo_overview.mp4`, `px4.ulg` | Gazebo tier only (below) |

## Flying and recording in Gazebo (Linux GPU machine)

The Gazebo tier flies the same pilot loop, skills and reflex layer against
PX4 in Gazebo through the aerial-autonomy-stack, in real time. **It is ported
from avians-02 but not yet tested in avians-03.** The AAS vehicle adapter
(`avians_sim/aas.py`) was written without access to the Linux machine. Run
step 6 first: it exercises everything except the PX4 control path.

The commands are marked **host** (a normal terminal on the machine) or
**container** (inside the avians container, entered with `./scripts/shell.sh`).

1. **Once per machine (host).** You must be in the docker group, and the GPU
   must be visible:
   ```bash
   docker ps >/dev/null && echo docker ok; nvidia-smi -L; command -v tmux
   ```
   Without a display, create the `~/bin/xterm` and `~/bin/xhost` stand-ins
   described in avians-02's `docs/running_simulations.md`, section 2.
2. **Once per machine (host): build AAS, and add avians' trajectory controller to it:**
   ```bash
   git clone --recursive git@github.com:andemamongane/avians-03-sar.git && cd avians-03-sar
   cp .env.example .env                                   # add ANTHROPIC_API_KEY
   (cd external/aerial-autonomy-stack/tools_and_docs && ./sim_build.sh)    # long, once
   ./scripts/aas_plan_ahead.sh                            # a minute; again after AAS rebuilds
   ```
3. **Each session (host, terminal 1): start AAS headless, in tmux.**
   ```bash
   cd external/aerial-autonomy-stack/tools_and_docs
   tmux new -s aas-launch "PATH=~/bin:\$PATH XDG_CURRENT_DESKTOP=GNOME HEADLESS=true GND_CONTAINER=false NUM_QUADS=1 WORLD=impalpable_greyness ./sim_run.sh; read -p 'sim_run.sh exited; Enter to close'"
   ```
   Wait for `Fly, my pretties, fly!`, then detach with `Ctrl-b d`. Don't press
   any other key in that window: any key stops AAS. sar_valley is drawn into
   AAS's plain world (`impalpable_greyness`), so `WORLD` must be exactly that.
4. **Each session (host, terminal 2): start the avians container and enter it.**
   ```bash
   ./scripts/up_remote.sh          # builds the first time (~15 min), then runs `avians doctor`
   ./scripts/shell.sh              # prompt becomes root@...:/workspace#
   ```
5. **Fly and record (container).**
   ```bash
   avians fly missions/sar_supply.yaml --tier gazebo --seed 3 --video --qgc               # scripted pilot
   avians fly missions/sar_supply.yaml --tier gazebo --seed 3 --pilot haiku --video --qgc  # the teacher
   ```
   Before takeoff, avians draws the world into Gazebo: regions, obstacles, pads
   and the person (a red box until Taylor adds a person model). It also sets
   PX4's envelope parameters and starts the wind and two cameras. The chase
   camera follows the drone; the overview camera frames the whole valley.
   After landing, the episode folder also holds `gazebo.mp4`,
   `gazebo_overview.mp4`, `gazebo_poses.jsonl` and PX4's `px4.ulg`. Decisions
   happen in real time here: the model call runs while PX4 flies, and the
   latency is whatever it really is.
6. **Replay any episode in Gazebo, fast-tier ones included (container).**
   ```bash
   avians gazebo-replay data/episodes/sar_supply/haiku/s0003
   ```
   This flies the recorded path as a visual-only drone, with kits appearing
   where they landed, and records the same two videos. It is the cheap way to
   get Gazebo footage of rollouts that ran at 300× real time.
7. **Stop.** In the container: `exit`. On the host:
   `docker compose -f docker/compose.yml down`, then `tmux attach -t aas-launch`
   and press any key.

**Watching in QGroundControl:**

- **Live (Gazebo tier):** run AAS with its ground container
  (`GND_CONTAINER=true`) or point a desktop QGC at it.
- **Afterwards (any tier):** QGC → Analyze Tools → Log Replay → `flight.tlog`.
  For the regions, use Plan view → import `world.kml`.
- **PX4's own log:** `px4.ulg` opens in PX4 Flight Review.

If something goes wrong, the symptom table in avians-02's
`docs/running_simulations.md` applies unchanged (AAS not starting, no
`/Drone1` topics, empty videos, the missing `avians_trajectory` controller).

## Several drones, one pilot each

`drones: 2` in a mission file gives two drones, each with its own pilot,
memory, decision timer, latency and ADP trajectory (`trajectory_d1.json`,
`trajectory_d2.json`), flying in one shared world (`avians_sim.multi`): one
person belief fused from every camera, one coverage map, one depot, one
radio. Each prompt lists the other drones (position, skill, battery, claimed
regions). Coordination is through the skills: `search_region` claims its
region, `assign_tasks` splits regions optimally (Hungarian method) and
radios each drone its share, `handoff` asks another drone to take over,
`formation` holds a relay or wing position. Pads are exclusive, and the
reflex layer keeps the drones apart by altitude (the higher id flies 12 m
above the other while within range, triggered early enough for the closing
speed).

The spec quantifies over drones: `forall x in drones: x.battery >= 10`,
`exists x in drones: x.in(r1)`, plus `min_sep` and `min_dist_person`
(`missions/sar_supply_2drones.yaml`). Quantifiers expand at parse time, so
scoring and the online monitor are unchanged (the finite fragment of SFO).

## The ground station

Every report, status and question the drone sends goes to a `GroundStation`
(`avians_sim/ground.py`) over a modelled link (latency, loss, range), and
everything coming back arrives as a radio **message event** that wakes the
pilot: the crew's acknowledgement with their ETA, an operator's answer to a
question, tasking, and the scenario's news. The mission file's `ground:`
block sets the link and scripts the operator (timed tasking, regex-matched
answers). Questions are asynchronous: `request_help` returns "sent" and the
answer, if any, comes later.

**Live operator.** `avians fly missions/sar_supply.yaml --ground live
--realtime` (or `--tier gazebo`) opens a console at http://localhost:8765:
the drone's state, the radio log, the pilot's pending questions, and a box to
send tasking, answers or news. Messages typed there reach the drone through
the same link, so the episode records like any other (`messages.jsonl`).
The JSON endpoints (`GET /state`, `GET /log`, `POST /message`) let a program,
such as a second model, play the operator.

## Decisions, latency and memory

- **Decision points:** any waking event (a skill finished or failed, the
  person was found, a zone changed, a battery mark, a reflex, help arrived,
  a rejected call), and at least every `wake_s` (45 s). See
  `configs/agent.yaml`.
- **Simulated latency (fast tier):** the prompt is built at `t_obs`, and the
  call applies at `t_obs + latency`. The latency is sampled per model from
  `configs/latency.yaml`, and the drone keeps flying meanwhile. Fit the
  models to the measured wall times with `avians latency-fit data/episodes`.
- **Memory:** each prompt is self-contained. It holds structured mission
  memory (searches, sightings, reports, deliveries, zone changes), the
  pilot's own `notes`, and the last 4 decisions. The full history is kept in
  the ADP trajectory for training on longer windows later.

## What is not done yet

- **The Gazebo tier is untested** (see above). The YOLO detection path is
  written (`avians_perception/projection.py`) but not wired in: the Gazebo
  tier still uses the simulated detector, fed with the true person position.
- **Taylor's tasks** (`packages/avians_env/README.md`): a full Gazebo world
  file, the person model, procedurally varied scenarios, and the payload's
  detachable joint.
- **McGlashan's tasks** (`packages/avians_adp/README.md`): validate against
  the upstream ADP schema, build the SFT and router datasets, and the
  dataset manifests.
- **Router:** the interfaces exist (`avians_agent/router.py`). The System One
  request format must be checked against the API reference before pointing it
  at Jev, Kev or Laya. The router needs McGlashan's routing dataset to train.
- **Swarm in Gazebo:** the fast tier flies n drones; the Gazebo tier still
  drives one `/Drone1`. Full SFO (quantifiers over time-varying domains,
  counting) is not ported; the finite `forall`/`exists` covers the benchmark.
- **aas-gym:** stepped, faster-than-real-time Gazebo for RL. The `Vehicle`
  interface is the seam for it.
