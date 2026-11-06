# avians_spec: what the mission must do, and how well a flight did it

- **`signals.py`:** the signal registry. Every signal a spec may name, with
  its unit, its range, its maximum rate of change, whether it is monotone
  (latched events), and a default (so every trace is defined at every step).
  `sar_registry(world)` is the benchmark's set: position, speed, battery,
  clearance, the person estimate (`est_*`, `conf`, `found`), `payload`,
  `released`, `drop_err`, `dist_person`, `help_dispatched`, `help`, `landed`,
  and `unsafe_<zone>` per hazard zone.
- **`stl.py`:** STL over named signals. `in(region)`; comparisons; bare
  booleans; `! & | ->`; the future operators `G F U`; the past operators
  `H` (historically) and `O` (once), with `[a,b]` or unbounded windows.
  Robustness is evaluated on a uniform grid, with the van Herk / Gil-Werman
  running min/max (O(n) per operator). `interval()` gives the ROSI robust
  satisfaction interval of an unfinished flight, with the unobserved future
  bounded by the registry. The test suite checks that it is sound.
- **`spec.py`:** a mission spec is named clauses with plain-language `says`,
  all of which must hold. It produces the pilot-facing description and
  per-clause scores.
- **`monitor.py`:** the online monitor. It runs every few seconds of sim time,
  decides each clause once, and can stop an episode early on a violation.

Example: `G[0,1500] (((O released) & !help) -> dist_person <= 30)` means
"once the kit is down, stay within 30 m until help arrives".

**Not yet ported:** SFO (signal first-order logic, for swarms) from
avians-02. It needs the registry-based signals.
