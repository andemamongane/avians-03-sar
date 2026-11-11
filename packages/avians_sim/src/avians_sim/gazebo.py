"""Gazebo around a flight (AAS's simulation container on the Linux machine).

Ported from avians-02 (sim.py, geo.py), for WorldSpec worlds:

    prepare()   check the running world, copy the helper (simside/avians_gz.py)
                into the container, swap AAS's scenery for a plain ground, set
                PX4 parameters from the envelope, spawn the world's models
                (regions, obstacles, pads, the person) at the drone's spawn
    start()     play the wind (mean + gusts), start the recordings (chase and
                overview cameras, every model's pose)
    kit(at)     show a released kit where it landed
    stop()      stop all of that, remove the models, copy gazebo.mp4,
                gazebo_overview.mp4, gazebo_poses.jsonl and px4.ulg into the episode
    replay()    fly a recorded episode (any tier, fast included) again in
                Gazebo as a visual-only drone, recording the same two videos:
                "view flight recordings in Gazebo" without re-flying PX4

Everything runs through `docker exec` into the simulation container, so the
avians container needs the docker socket (docker/compose.remote.yml).
"""
from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from avians_env.gazebo import all_models, person_model
from avians_safety import px4_params

HELPER = Path(__file__).parent / "simside" / "avians_gz.py"
WORK = "/tmp/avians"
VIDEOS = {"chase": "gazebo.mp4", "overview": "gazebo_overview.mp4"}
CAMERA_HFOV = 1.2
REPLAY_MODEL = "avians_replay_drone"


class SimError(RuntimeError):
    pass


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=kw.get("timeout", 60))


# ---------------------------------------------------------------- models
def _camera_model(name, pose, width, height, fps, topic) -> str:
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<pose>{pose}</pose><link name='link'><sensor name='camera' type='camera'>"
            f"<topic>{topic}</topic><update_rate>{fps}</update_rate><always_on>true</always_on>"
            f"<camera><horizontal_fov>{CAMERA_HFOV}</horizontal_fov>"
            f"<image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>"
            f"<clip><near>0.1</near><far>1500</far></clip></camera></sensor></link></model></sdf>")


def chase_camera_model(name, width=1280, height=720, fps=15, topic="/avians/chase/image") -> str:
    return _camera_model(name, "-8 0 6 0 0.45 0", width, height, fps, topic)


def overview_camera_model(world, anchor, name, width=1280, height=720, fps=15,
                          topic="/avians/overview/image", pitch=1.05) -> str:
    """Fixed camera south of the operating area, looking north and down, far
    enough back to frame every region and the launch point."""
    pts = [world.launch] + [p for r in world.regions.values() for p in r.polygon]
    es, ns = [p[0] for p in pts], [p[1] for p in pts]
    ce, cn = (min(es) + max(es)) / 2, (min(ns) + max(ns)) / 2
    w, h = 1.15 * (max(es) - min(es)) + 4, 1.15 * (max(ns) - min(ns)) + 4
    half_h = math.tan(CAMERA_HFOV / 2)
    half_v = half_h * height / width
    dist = max(w / 2 / half_h, h * math.sin(pitch) / 2 / half_v)
    x, y, z = anchor[0] + ce, anchor[1] + cn - dist * math.cos(pitch), dist * math.sin(pitch)
    return _camera_model(name, f"{x:.3f} {y:.3f} {z:.3f} 0 {pitch:.4f} {math.pi / 2:.4f}",
                         width, height, fps, topic)


def plain_ground_model(name="avians_ground", size_m=1200.0) -> str:
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<link name='link'><collision name='collision'><pose>0 0 -5 0 0 0</pose>"
            f"<geometry><box><size>100000 100000 10</size></box></geometry></collision>"
            f"<visual name='visual'><geometry><plane><normal>0 0 1</normal><size>{size_m:g} {size_m:g}"
            f"</size></plane></geometry><material><ambient>0.86 0.86 0.86 1</ambient><diffuse>0.86 0.86 "
            f"0.86 1</diffuse></material></visual></link><link name='lighting_link'><light name='sun' "
            f"type='directional'><pose>0 0 500 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>1"
            f"</intensity><direction>0.001 0.625 -0.78</direction><diffuse>0.904 0.904 0.904 1</diffuse>"
            f"<specular>0.271 0.271 0.271 1</specular></light></link></model></sdf>")


def replay_drone_model(name=REPLAY_MODEL) -> str:
    """A visual-only quadrotor (body + four rotor discs) for replays."""
    arms = "".join(
        f"<visual name='rotor{k}'><pose>{0.25 * math.cos(a):.3f} {0.25 * math.sin(a):.3f} 0.05 0 0 0</pose>"
        f"<geometry><cylinder><radius>0.12</radius><length>0.01</length></cylinder></geometry>"
        f"<material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual>"
        for k, a in enumerate([math.pi / 4 + i * math.pi / 2 for i in range(4)]))
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<link name='link'><visual name='body'><geometry><box><size>0.35 0.35 0.1</size></box>"
            f"</geometry><material><ambient>0.15 0.3 0.8 1</ambient><diffuse>0.15 0.3 0.8 1</diffuse>"
            f"</material></visual>{arms}</link></model></sdf>")


def kit_model(name, at, anchor) -> str:
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<pose>{anchor[0] + at[0]:.3f} {anchor[1] + at[1]:.3f} 0.15 0 0 0</pose><link name='link'>"
            f"<visual name='v'><geometry><box><size>0.4 0.3 0.3</size></box></geometry><material>"
            f"<ambient>0.95 0.8 0.1 1</ambient><diffuse>0.95 0.8 0.1 1</diffuse></material></visual>"
            f"</link></model></sdf>")


def wind_schedule(wind, duration: float, dt: float = 0.5, tau: float = 4.0, seed: int = 0):
    """[[t, east, north], ...]: the mean wind plus first-order (Ornstein-Uhlenbeck)
    gusts with std gust_std and correlation time tau."""
    rng = np.random.default_rng(seed)
    a = math.exp(-dt / tau)
    ge = gn = 0.0
    out = []
    for k in range(int(duration / dt) + 1):
        if wind.gust_std > 0:
            ge = a * ge + math.sqrt(1 - a * a) * rng.normal(0, wind.gust_std)
            gn = a * gn + math.sqrt(1 - a * a) * rng.normal(0, wind.gust_std)
        out.append([round(k * dt, 2), round(wind.east + ge, 3), round(wind.north + gn, 3)])
    return out


# ---------------------------------------------------------------- control
class GazeboControl:
    def __init__(self, cfg: dict, episode: Path, run=_run, popen=subprocess.Popen):
        self.cfg, self.episode = cfg, Path(episode)
        self._run, self._popen = run, popen
        self.container = cfg.get("container", "simulation-container-inst0")
        self.model = cfg.get("model", "x500_0")
        self.world_name: str | None = None
        self.notes: list[str] = []
        self.applied_params: dict = {}
        self._spawned: list[str] = []
        self._recordings: list = []
        self._anchor = (0.0, 0.0)
        self._world = None
        self._t_start: float | None = None
        self._plain: list[str] = []

    # -- plumbing (as avians-02)
    def sh(self, args: list[str], check: bool = True) -> str:
        res = self._run(["docker", "exec", self.container, "bash", "-lc", shlex.join(args)])
        if check and res.returncode != 0:
            raise SimError(f"{' '.join(args)} failed: {(res.stderr or res.stdout).strip()}")
        return res.stdout

    def put(self, text: str, name: str) -> str:
        local = self.episode / "sim" / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text)
        self._run(["docker", "exec", self.container, "mkdir", "-p", WORK])
        res = self._run(["docker", "cp", str(local), f"{self.container}:{WORK}/{name}"])
        if res.returncode != 0:
            raise SimError(f"docker cp {name} failed: {res.stderr.strip()}")
        return f"{WORK}/{name}"

    def _px4(self, tool: str) -> list[str]:
        b = self.cfg.get("px4_bin", "/aas/github_apps/PX4-Autopilot/build/px4_sitl_default/bin")
        return [f"{b}/px4-{tool}", "--instance", str(self.cfg.get("px4_instance", 0))]

    def _helper(self, *args: str) -> list[str]:
        return ["python3", f"{WORK}/avians_gz.py", *args]

    def _background(self, name: str, args: list[str], stdout=None, stdin=None):
        inner = f"echo $$ > {WORK}/{name}.pid; exec {shlex.join(args)}"
        log = self.episode / "sim" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as err:
            return self._popen(["docker", "exec", "-i", self.container, "bash", "-lc", inner],
                               stdout=stdout or subprocess.DEVNULL, stdin=stdin, stderr=err)

    def _kill(self, name: str) -> None:
        self._run(["docker", "exec", self.container, "bash", "-lc",
                   f"kill -TERM $(cat {WORK}/{name}.pid) 2>/dev/null; rm -f {WORK}/{name}.pid"])

    def spawn(self, name: str, sdf: str) -> None:
        path = self.put(sdf, f"{name}.sdf")
        self.sh(self._helper("spawn", "--world", self.world_name, "--sdf", path), check=False)
        self._spawned.append(name)

    # -- before takeoff
    def prepare(self, world, limits=None, person_at=None, set_params: bool = True) -> None:
        self._world = world
        topics = self.sh(["gz", "topic", "-l"])
        running = sorted({t.split("/")[2] for t in topics.split() if t.startswith("/world/")})
        if not running:
            raise SimError("Gazebo is not running in the simulation container (start AAS first)")
        want = self.cfg.get("world")
        if want and want not in running:
            raise SimError(f"expected Gazebo world {want!r}, running {running}: restart AAS with WORLD={want}")
        self.world_name = want or running[0]
        self._run(["docker", "exec", self.container, "mkdir", "-p", WORK])
        res = self._run(["docker", "cp", str(HELPER), f"{self.container}:{WORK}/avians_gz.py"])
        if res.returncode != 0:
            raise SimError(f"could not copy the Gazebo helper: {res.stderr.strip()}")
        self._make_plain_world()
        if set_params and limits is not None:
            params = px4_params(limits)
            for k, v in params.items():
                self.sh(self._px4("param") + ["set", k, f"{v:g}"], check=False)
            self.applied_params = params
        out = self.sh(self._helper("pose", "--world", self.world_name, "--model", self.model),
                      check=False).split()
        self._anchor = (float(out[0]), float(out[1])) if len(out) >= 2 else (0.0, 0.0)
        if len(out) < 2:
            self.notes.append(f"pose of {self.model} unknown: world models placed at the origin")
        for name, sdf in all_models(world, person_at, self._anchor).items():
            self.spawn(name, sdf)

    def _make_plain_world(self) -> None:
        scenery = self.cfg.get("plain_world", {}).get(self.world_name, [])
        if not scenery:
            return
        path = self.put(plain_ground_model(), "avians_ground.sdf")
        self.sh(self._helper("spawn", "--world", self.world_name, "--sdf", path), check=False)
        if not self.sh(self._helper("pose", "--world", self.world_name, "--model", "avians_ground"),
                       check=False).strip():
            self.notes.append("kept the world's scenery: could not add a plain ground")
            return
        for name in scenery:
            self.sh(self._helper("remove", "--world", self.world_name, "--name", name), check=False)
        self._plain = list(scenery)

    # -- during
    def start(self, horizon_s: float, follow_model: str | None = None, seed: int = 0, wind=None) -> None:
        self._t_start = time.time()
        w = wind or self._world.wind
        if w.east or w.north or w.gust_std:
            sched = wind_schedule(w, horizon_s, seed=seed)
            (self.episode / "wind.json").write_text(json.dumps(sched))
            path = self.put(json.dumps(sched), "wind.json")
            self._background("wind", self._helper("wind", "--world", self.world_name, "--schedule", path))
        self._start_recording(follow_model or self.model)

    def _start_recording(self, model: str) -> None:
        rec = self.cfg.get("record_video", {})
        w, h, fps = rec.get("width", 1280), rec.get("height", 720), rec.get("fps", 15)
        if not shutil.which("ffmpeg"):
            self.notes.append("no ffmpeg: the Gazebo videos were not recorded")
            return
        stamp = int(time.time())
        for view in rec.get("views", list(VIDEOS)):
            name = f"avians_{view}_cam_{self.episode.name}_{stamp}"
            topic = f"/avians/{view}/{name}/image"
            if view == "chase":
                sdf = chase_camera_model(name, w, h, fps, topic)
                extra = ["--poses", f"{WORK}/poses.jsonl"]
            else:
                sdf = overview_camera_model(self._world, self._anchor, name, w, h, fps, topic)
                extra = ["--fixed", "--track"]
            cam = self.put(sdf, f"{view}_cam.sdf")
            job = "record" if view == "chase" else f"record_{view}"
            recorder = self._background(job, self._helper(
                "record", "--world", self.world_name, "--model", model, "--camera-sdf", cam,
                "--camera-name", name, "--topic", topic, *extra, "--width", str(w), "--height", str(h),
                "--fps", str(fps)), stdout=subprocess.PIPE)
            ffmpeg = self._popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 str(self.episode / VIDEOS[view])], stdin=recorder.stdout)
            self._recordings.append((view, job, recorder, ffmpeg))

    def kit(self, at: tuple[float, float], k: int = 0) -> None:
        self.spawn(f"avians_kit_{k}", kit_model(f"avians_kit_{k}", at, self._anchor))

    # -- after
    def stop(self, copy_ulog: bool = True) -> None:
        self._kill("wind")
        for _, job, _, _ in self._recordings:
            self._kill(job)
        for view, job, recorder, ffmpeg in self._recordings:
            for p in (recorder, ffmpeg):
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
            v = self.episode / VIDEOS[view]
            if not v.exists() or v.stat().st_size < 1024:
                self.notes.append(f"the Gazebo {view} video has no frames (see sim/{job}.log)")
        if any(view == "chase" for view, *_ in self._recordings):
            self._run(["docker", "cp", f"{self.container}:{WORK}/poses.jsonl",
                       str(self.episode / "gazebo_poses.jsonl")])
        for name in self._spawned:
            self.sh(self._helper("remove", "--world", self.world_name, "--name", name), check=False)
        if copy_ulog and self._t_start is not None:
            root = self.cfg.get("px4_log_root", "/aas/github_apps/PX4-Autopilot/build/px4_sitl_default")
            found = self.sh(["bash", "-c", f"find {shlex.quote(root)} -name '*.ulg' -newermt "
                                           f"@{int(self._t_start) - 5} -printf '%T@ %p\\n' | sort -n | tail -1"],
                            check=False).split()
            if len(found) >= 2:
                self._run(["docker", "cp", f"{self.container}:{found[1]}", str(self.episode / "px4.ulg")])
            else:
                self.notes.append("no PX4 .ulg log found for this flight")

    def summary(self) -> dict:
        return {"world": self.world_name, "anchor": self._anchor, "px4_params": self.applied_params,
                "spawned": self._spawned, "plain_world": self._plain, "notes": self.notes}

    # -- replay
    def replay(self, episode_dir: str | Path, world, speed: float = 1.0) -> dict:
        """Re-fly a recorded episode as a visual-only drone, recording both views.
        Real time (speed 1) by default: the cameras record at their own rate."""
        ep = Path(episode_dir)
        rows = [json.loads(line) for line in open(ep / "signals.jsonl")]
        meta = json.loads((ep / "meta.json").read_text())
        self.prepare(world, None, tuple(meta["person"]) if meta.get("person") else None, set_params=False)
        self.spawn(REPLAY_MODEL, replay_drone_model())
        mover = self._background("replay_move", self._helper("move", "--world", self.world_name),
                                 stdin=subprocess.PIPE)
        self.start(rows[-1]["t"] / speed + 5, follow_model=REPLAY_MODEL)
        kits = [json.loads(line) for line in open(ep / "events.jsonl")]
        kits = [k for k in kits if k["kind"] == "kit_landed"]
        t0, ax, ay = time.time(), *self._anchor
        prev = rows[0]
        for i, r in enumerate(rows):
            while time.time() - t0 < r["t"] / speed:
                time.sleep(0.005)
            de, dn = r["east"] - prev["east"], r["north"] - prev["north"]
            yaw = math.atan2(dn, de) if math.hypot(de, dn) > 0.05 else 0.0
            line = (f"{REPLAY_MODEL} {ax + r['east']:.3f} {ay + r['north']:.3f} {r['alt'] + 0.1:.3f} "
                    f"{math.cos(yaw / 2):.5f} 0 0 {math.sin(yaw / 2):.5f}\n")
            try:
                mover.stdin.write(line.encode())
                mover.stdin.flush()
            except BrokenPipeError:
                self.notes.append("the pose mover stopped early (see sim/replay_move.log)")
                break
            for k in kits:
                if prev["t"] < k["t"] <= r["t"]:
                    import re
                    m = re.findall(r"\(([-\d.]+), ([-\d.]+)\)", k["text"])
                    if m:
                        self.kit(tuple(map(float, m[0])), len(self._spawned))
            prev = r
        mover.stdin.close()
        self.stop(copy_ulog=False)
        return self.summary()


__all__ = ["GazeboControl", "SimError", "person_model", "wind_schedule"]
