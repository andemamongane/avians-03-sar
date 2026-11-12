# Linux test: AAS / Gazebo / QGroundControl

Goal: show the SAR mission in simulation and the drone swarm in AAS's Gazebo,
with recorded video and QGroundControl logs. Mission success is not the point
here.

What runs, from safest to riskiest:

| Stage | What | Needs PX4 in the loop? |
|---|---|---|
| Gazebo replay | the committed example episodes (one drone in a storm; two Haiku pilots in a storm) re-flown in AAS's Gazebo as visual drones, filmed by a chase and an overview camera | no |
| QGC logs | `.tlog` of each episode (every drone, the pilots' decisions as status messages) | no |
| Live flight (`LIVE=1`) | the scripted pilot flies `sar_supply` through PX4 + AAS in Gazebo | yes (untested code) |

Commands marked **host** run in a normal terminal on the Linux machine;
**container** means inside the avians container (`./scripts/shell.sh`).

## 1. Get the code (host)

```bash
git clone --recursive -b andema/stage3-completion git@github.com:andemamongane/avians-03-sar.git
cd avians-03-sar
cp .env.example .env            # optional: ANTHROPIC_API_KEY, only for --pilot haiku
```

Already cloned: `git fetch && git checkout andema/stage3-completion && git pull && git submodule update --init --recursive`.

## 2. Build AAS and add the trajectory controller (host, once)

```bash
(cd external/aerial-autonomy-stack/tools_and_docs && ./sim_build.sh)   # long, once
./scripts/aas_plan_ahead.sh                                           # a minute; needed for LIVE=1 only
```

Headless machine: create the `~/bin/xterm` and `~/bin/xhost` stand-ins from
avians-02's `docs/running_simulations.md`, section 2.

## 3. Start AAS (host, terminal 1)

```bash
cd external/aerial-autonomy-stack/tools_and_docs
tmux new -s aas-launch "PATH=~/bin:\$PATH XDG_CURRENT_DESKTOP=GNOME HEADLESS=true GND_CONTAINER=false NUM_QUADS=1 WORLD=impalpable_greyness ./sim_run.sh; read -p 'sim_run.sh exited; Enter to close'"
```

Wait for `Fly, my pretties, fly!`, then detach with `Ctrl-b d`. `WORLD` must be
`impalpable_greyness`: avians draws the valley's 17 zones, obstacles, pads and
the hiker into it.

## 4. Start the avians container (host, terminal 2)

```bash
./scripts/up_remote.sh          # first run builds the image (~15 min), then runs `avians doctor`
./scripts/shell.sh              # prompt becomes root@...:/workspace#
```

## 5. Run the demo (container)

```bash
./scripts/linux_demo.sh                  # checks, Gazebo replays (3x speed), QGC logs, 2D replays: ~10 min
LIVE=1 ./scripts/linux_demo.sh           # also the live PX4 flight
SPEED=1 ./scripts/linux_demo.sh          # replays in real time (the swarm episode is ~14 min)
```

The script never stops on a failure: it logs each stage and ends with
`passed: ...`, `failed: ...`, the list of videos, and an archive to send back:
`data/linux_demo/<stamp>.tar.gz` (also visible on the host, `data/` is mounted).

Success looks like, per example episode in `data/linux_demo/<stamp>/`:

| File | What |
|---|---|
| `gazebo.mp4` | chase camera behind the first drone, in Gazebo |
| `gazebo_overview.mp4` | the whole valley from above: zones, obstacles, pads, the hiker, every drone, kits landing |
| `flight.tlog`, `world.kml` | QGroundControl: Analyze Tools → Log Replay; Plan view → import KML |
| `replay.mp4` | the 2D replay of the same episode |

Swarm footage: `swarm_haiku_storm/gazebo_overview.mp4` shows both drones.

## 6. QGroundControl

- **Replay (any machine with QGC):** open `flight.tlog` in Analyze Tools → Log
  Replay. Both drones of the swarm episode appear as vehicles 1 and 2; the
  pilots' decisions scroll in the message panel.
- **Live, PX4's own stream:** run AAS with `GND_CONTAINER=true` (needs a display).
- **Live, from avians:** `avians fly missions/sar_supply_2drones.yaml --realtime --qgc-live <QGC host IP>:14550`
  streams MAVLink from the fast simulator (both drones) to a desktop QGC.

## 7. More, if time allows (container)

```bash
avians fly missions/sar_supply_2drones.yaml --seed 5 --out data/more     # a new swarm episode (fast, no PX4)
avians gazebo-replay data/more/s0005 --speed 3                           # ...and its Gazebo footage
avians fly missions/sar_supply.yaml --tier gazebo --pilot haiku --seed 0 --qgc   # live, LLM pilot (needs .env)
avians showcase examples/episodes/swarm_haiku_storm --out data/showcase.mp4      # the 1080p sponsor film
```

## When something fails

| Symptom | Likely cause |
|---|---|
| `aas-running` failed | AAS not started, or it exited: `tmux attach -t aas-launch` |
| `gazebo-world` failed | Gazebo still loading: wait 30 s and rerun |
| replay ran but no `gazebo*.mp4` / tiny files | the camera did not render headless: see `sim/record*.log` in the episode; `tmux attach -t aas-Simulation` for Gazebo errors |
| replay: `could not copy the Gazebo helper` | the container cannot reach the docker socket: rerun `./scripts/up_remote.sh` |
| `expected Gazebo world 'impalpable_greyness'` | AAS was started with another `WORLD`: restart it (step 3) |
| live: `no avians_trajectory controller` | run `./scripts/aas_plan_ahead.sh` on the host, restart AAS |
| live: `already armed` | the previous flight did not land: restart AAS |

Send back the `.tar.gz` either way: the logs say what to fix.
