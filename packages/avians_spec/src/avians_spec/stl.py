"""Signal temporal logic over named signals, with past-time operators.

Syntax (whitespace-insensitive; times in seconds; `inf` allowed in past windows):

    in(r3)                 inside region r3 (signed distance, m)
    battery >= 15          a signal compared with a number
    found                  a boolean signal is 1 (same as found >= 0.5)
    true  false  !phi  phi & psi  phi | psi  phi -> psi  (parentheses)

    future                 G[a,b] phi   always in [t+a, t+b]
                           F[a,b] phi   eventually in [t+a, t+b]
                           phi U[a,b] psi
    past                   H[a,b] phi   historically: always in [t-b, t-a]
                           O[a,b] phi   once: at some time in [t-b, t-a]
                           H phi, O phi the whole past, [0, inf)

    e.g.  G[0,1500] ((O released) & !help -> dist_person <= 25)
          "once the kit is down, stay within 25 m of the person until help arrives"

Robustness (Donze & Maler 2010): atoms are margins (in(R): signed distance;
x >= c: x - c; boolean x: x - 0.5), ! negates, & | are min/max, G F H O are
min/max over their windows. rho > 0 is satisfied with that margin.

Evaluation is on a uniform grid (dt). Windows use the van Herk / Gil-Werman
running min/max, O(n) per operator. A finished flight shorter than the
spec's horizon is padded by holding its last sample.

Online (ROSI, Deshmukh et al. 2017): with the trace known up to now, every
formula's robustness at t=0 lies in an interval [lo, hi] covering every value
the finished flight could still score. The unobserved future of each signal is
bounded by the registry (range, rate, monotone); positions move at most
max_speed. hi <= 0: violated whatever happens next; lo > 0: already satisfied.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .signals import INF, POSITION, Registry

NEUTRAL = {"min": INF, "max": -INF}


# ---------------------------------------------------------------- AST
@dataclass
class Node:
    op: str              # true false in cmp not and or G F U H O
    args: tuple = ()
    lo: float = 0.0
    hi: float = 0.0
    name: str = ""       # region id, or signal name
    cmp: str = ""
    value: float = 0.0

    def __str__(self) -> str:
        if self.op in ("true", "false"):
            return self.op
        if self.op == "in":
            return f"in({self.name})"
        if self.op == "cmp":
            return self.name if self.cmp == "bool" else f"{self.name} {self.cmp} {self.value:g}"
        if self.op == "not":
            return f"!{_paren(self.args[0])}"
        if self.op in ("and", "or"):
            return (" & " if self.op == "and" else " | ").join(_paren(a) for a in self.args)
        if self.op == "U":
            return f"{_paren(self.args[0])} U{_iv(self)} {_paren(self.args[1])}"
        if self.op in ("H", "O") and self.lo == 0 and math.isinf(self.hi):
            return f"{self.op} {_paren(self.args[0])}"
        return f"{self.op}{_iv(self)} {_paren(self.args[0])}"

    def horizon(self) -> float:
        """How far into the future (from t) the formula looks."""
        sub = max((a.horizon() for a in self.args), default=0.0)
        return sub + self.hi if self.op in ("G", "F", "U") else sub

    def regions(self) -> set[str]:
        out = {self.name} if self.op == "in" else set()
        for a in self.args:
            out |= a.regions()
        return out

    def signals(self) -> set[str]:
        out = {self.name} if self.op == "cmp" else (set(POSITION) if self.op == "in" else set())
        for a in self.args:
            out |= a.signals()
        return out


def _iv(n: Node) -> str:
    return f"[{n.lo:g},{'inf' if math.isinf(n.hi) else f'{n.hi:g}'}]"


def _paren(n: Node) -> str:
    s = str(n)
    return f"({s})" if n.op in ("and", "or", "U") else s


# ---------------------------------------------------------------- parser
class STLSyntaxError(ValueError):
    pass


_TOKEN = re.compile(r"\s*(->|<=|>=|[()\[\],!&|<>~]|[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d*)?(?:e-?\d+)?|-?\.\d+)")
TEMPORAL = ("G", "F", "H", "O")


def _tokenize(text: str) -> list[str]:
    toks, pos, text = [], 0, text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise STLSyntaxError(f"unexpected character at {pos}: {text[pos:pos + 12]!r}")
        toks.append(m.group(1))
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return toks


class _Parser:
    """Precedence, loosest first:  ->  |  &  U  unary (! G F H O)."""

    def __init__(self, text: str, registry: Registry | None):
        self.toks, self.i, self.reg = _tokenize(text), 0, registry

    def parse(self) -> Node:
        if not self.toks:
            raise STLSyntaxError("empty spec")
        node = self.implies()
        if self.i != len(self.toks):
            raise STLSyntaxError(f"unexpected '{self.toks[self.i]}'")
        return node

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise STLSyntaxError(f"spec ended early (expected {expected or 'more'})")
        if expected is not None and tok != expected:
            raise STLSyntaxError(f"expected '{expected}', got '{tok}'")
        self.i += 1
        return tok

    def implies(self) -> Node:
        left = self.or_()
        if self.peek() == "->":
            self.take()
            return Node("or", (Node("not", (left,)), self.implies()))
        return left

    def or_(self) -> Node:
        parts = [self.and_()]
        while self.peek() == "|":
            self.take()
            parts.append(self.and_())
        return parts[0] if len(parts) == 1 else Node("or", tuple(parts))

    def and_(self) -> Node:
        parts = [self.until()]
        while self.peek() == "&":
            self.take()
            parts.append(self.until())
        return parts[0] if len(parts) == 1 else Node("and", tuple(parts))

    def until(self) -> Node:
        left = self.unary()
        if self.peek() == "U":
            self.take()
            lo, hi = self.interval(allow_inf=False)
            return Node("U", (left, self.unary()), lo=lo, hi=hi)
        return left

    def unary(self) -> Node:
        tok = self.peek()
        if tok in ("!", "~"):
            self.take()
            return Node("not", (self.unary(),))
        if tok in TEMPORAL:
            self.take()
            past = tok in ("H", "O")
            if self.peek() == "[":
                lo, hi = self.interval(allow_inf=past)
            elif past:
                lo, hi = 0.0, INF
            else:
                raise STLSyntaxError(f"{tok} needs an interval, e.g. {tok}[0,60]")
            return Node(tok, (self.unary(),), lo=lo, hi=hi)
        return self.atom()

    def interval(self, allow_inf: bool) -> tuple[float, float]:
        self.take("[")
        lo = self.number()
        self.take(",")
        if self.peek() == "inf":
            if not allow_inf:
                raise STLSyntaxError("future windows must be bounded")
            self.take()
            hi = INF
        else:
            hi = self.number()
        self.take("]")
        if not 0 <= lo <= hi:
            raise STLSyntaxError(f"interval [{lo:g},{hi:g}] must satisfy 0 <= a <= b")
        return lo, hi

    def number(self) -> float:
        tok = self.take()
        try:
            return float(tok)
        except ValueError:
            raise STLSyntaxError(f"expected a number, got '{tok}'") from None

    def atom(self) -> Node:
        tok = self.take()
        if tok == "(":
            node = self.implies()
            self.take(")")
            return node
        if tok in ("true", "false"):
            return Node(tok)
        if tok == "in":
            self.take("(")
            name = self.take()
            self.take(")")
            return Node("in", name=name)
        if not re.match(r"[A-Za-z_]", tok):
            raise STLSyntaxError(f"unexpected '{tok}'")
        if self.reg is not None and tok not in self.reg:
            raise STLSyntaxError(f"unknown signal '{tok}' (known: {', '.join(self.reg.names())})")
        if self.peek() in ("<", "<=", ">", ">="):
            cmp = self.take()
            return Node("cmp", name=tok, cmp=cmp, value=self.number())
        if self.reg is not None and not self.reg[tok].boolean:
            raise STLSyntaxError(f"'{tok}' is not boolean: compare it with a number")
        return Node("cmp", name=tok, cmp="bool", value=0.5)


def parse(text: str, registry: Registry | None = None) -> Node:
    return _Parser(text, registry).parse()


# ---------------------------------------------------------------- traces
class Trace:
    """Signals on a uniform grid t_k = k * dt, k = 0..n-1."""

    def __init__(self, dt: float, signals: dict[str, np.ndarray]):
        self.dt = dt
        self.signals = {k: np.asarray(v, float) for k, v in signals.items()}
        lens = {len(v) for v in self.signals.values()}
        if len(lens) > 1:
            raise ValueError("all signals need the same length")
        self.n = lens.pop() if lens else 0

    @classmethod
    def from_samples(cls, times, rows: list[dict], dt: float, defaults: dict | None = None) -> "Trace":
        """Zero-order hold of irregular samples onto the grid up to the last time."""
        times = np.asarray(times, float)
        n = int(math.floor(times[-1] / dt + 1e-9)) + 1
        grid = np.arange(n) * dt
        idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
        names = set(defaults or {})
        for r in rows:
            names |= set(r)
        sig = {}
        for name in names:
            d = (defaults or {}).get(name, 0.0)
            col = np.array([r.get(name, d) for r in rows], float)
            sig[name] = col[idx]
        return cls(dt, sig)

    def padded(self, n: int) -> "Trace":
        """Hold the last sample out to n samples (finished flights)."""
        if n <= self.n:
            return self
        return Trace(self.dt, {k: np.concatenate([v, np.full(n - self.n, v[-1])])
                               for k, v in self.signals.items()})

    def __getitem__(self, name: str) -> np.ndarray:
        return self.signals[name]


# ---------------------------------------------------------------- windows
def _running(x: np.ndarray, w: int, how: str) -> np.ndarray:
    """out[i] = how(x[i : i+w]) with x padded by the neutral element:
    van Herk / Gil-Werman, two scans per block."""
    n = len(x)
    if w <= 1 or n == 0:
        return x.copy()
    neutral = NEUTRAL[how]
    acc = np.minimum if how == "min" else np.maximum
    m = -(-(n + w) // w) * w
    xp = np.full(m, neutral)
    xp[:n] = x
    blocks = xp.reshape(-1, w)
    pre = acc.accumulate(blocks, axis=1).ravel()
    suf = acc.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    i = np.arange(n)
    return acc(suf[i], pre[i + w - 1])


def _future(x: np.ndarray, lo: int, hi: int, how: str) -> np.ndarray:
    """out[i] = how(x[i+lo .. i+hi]), clipped to the trace (neutral if empty)."""
    n = len(x)
    r = _running(x, hi - lo + 1, how)
    out = np.full(n, NEUTRAL[how])
    if lo < n:
        out[: n - lo] = r[lo:]
    return out


def _past(x: np.ndarray, lo: int, hi: float, how: str) -> np.ndarray:
    """out[i] = how(x[i-hi .. i-lo]), clipped at 0 (neutral if empty)."""
    n = len(x)
    acc = np.minimum if how == "min" else np.maximum
    shifted = np.full(n, NEUTRAL[how])
    if lo < n:
        shifted[lo:] = x[: n - lo]
    if math.isinf(hi) or hi >= n:
        return acc.accumulate(shifted)
    w = int(hi) - lo + 1
    rev = _running(shifted[::-1], w, how)[::-1]
    return rev


def _until(a: np.ndarray, b: np.ndarray, lo: int, hi: int, need: int) -> np.ndarray:
    """out[i] = max over j in [i+lo, i+hi] of min(b[j], min a[i..j-1]); first `need` entries."""
    n = len(a)
    out = np.full(n, -INF)
    for i in range(min(n, need)):
        j1 = min(n, i + hi + 1)
        if i + lo >= j1:
            continue
        run = np.minimum.accumulate(np.concatenate([[INF], a[i:j1 - 1]]))
        cand = np.minimum(b[i:j1], run)[lo:]
        out[i] = cand.max() if len(cand) else -INF
    return out


# ---------------------------------------------------------------- semantics
def _atom(node: Node, tr: Trace, regions) -> np.ndarray:
    if node.op == "in":
        if regions is None or node.name not in regions:
            raise ValueError(f"spec uses region '{node.name}', which the world does not define")
        return regions[node.name].robustness(tr["east"], tr["north"], tr["alt"])
    x = tr[node.name]
    if node.cmp in ("<", "<="):
        return node.value - x
    return x - node.value          # >=, >, bool


def _rho(node: Node, tr: Trace, regions, need: int) -> np.ndarray:
    op, dt = node.op, tr.dt
    if op == "true":
        return np.full(tr.n, INF)
    if op == "false":
        return np.full(tr.n, -INF)
    if op in ("in", "cmp"):
        return _atom(node, tr, regions)
    if op == "not":
        return -_rho(node.args[0], tr, regions, need)
    if op in ("and", "or"):
        parts = [_rho(a, tr, regions, need) for a in node.args]
        return np.minimum.reduce(parts) if op == "and" else np.maximum.reduce(parts)
    lo = int(round(node.lo / dt))
    hi = INF if math.isinf(node.hi) else int(round(node.hi / dt))
    if op in ("G", "F"):
        return _future(_rho(node.args[0], tr, regions, need + hi), lo, hi, "min" if op == "G" else "max")
    if op in ("H", "O"):
        return _past(_rho(node.args[0], tr, regions, need), lo, hi, "min" if op == "H" else "max")
    if op == "U":
        return _until(_rho(node.args[0], tr, regions, need + hi),
                      _rho(node.args[1], tr, regions, need + hi), lo, hi, need)
    raise ValueError(f"unknown operator {op}")


def robustness(spec: Node, trace: Trace, regions=None) -> float:
    n = int(round(spec.horizon() / trace.dt)) + 1
    return float(_rho(spec, trace.padded(n), regions, 1)[0])


# ---------------------------------------------------------------- online (ROSI)
def _atom_iv(node: Node, tr: Trace, regions, known: int, registry: Registry | None,
             max_speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Bounds of an atom: exact on [0, known), registry bounds after."""
    now = _atom(node, tr, regions)
    lo, hi = now.copy(), now.copy()
    n = tr.n
    if known >= n:
        return lo, hi
    tau = (np.arange(known, n) - (known - 1)) * tr.dt          # seconds past "now"
    if node.op == "in":
        reach = max_speed * tau
        r0 = now[known - 1]
        lo[known:], hi[known:] = r0 - reach, r0 + reach
        return lo, hi
    s = registry[node.name] if registry is not None and node.name in registry else None
    v0 = tr[node.name][known - 1]
    if s is None:
        v_lo, v_hi = np.full(len(tau), -INF), np.full(len(tau), INF)
    else:
        rate = s.rate
        v_lo = np.full(len(tau), v0 if s.monotone == "up" else -INF) if math.isinf(rate) \
            else (np.full(len(tau), v0) if s.monotone == "up" else v0 - rate * tau)
        v_hi = np.full(len(tau), INF) if math.isinf(rate) else v0 + rate * tau
        v_lo, v_hi = np.clip(v_lo, s.lo, s.hi), np.clip(v_hi, s.lo, s.hi)
    if node.cmp in ("<", "<="):
        lo[known:], hi[known:] = node.value - v_hi, node.value - v_lo
    else:
        lo[known:], hi[known:] = v_lo - node.value, v_hi - node.value
    return lo, hi


def _rho_iv(node: Node, tr: Trace, regions, known: int, registry, max_speed: float, need: int):
    op, dt = node.op, tr.dt
    if op in ("true", "false"):
        v = np.full(tr.n, INF if op == "true" else -INF)
        return v, v
    if op in ("in", "cmp"):
        return _atom_iv(node, tr, regions, known, registry, max_speed)
    rec = lambda a, nd=need: _rho_iv(a, tr, regions, known, registry, max_speed, nd)  # noqa: E731
    if op == "not":
        lo, hi = rec(node.args[0])
        return -hi, -lo
    if op in ("and", "or"):
        parts = [rec(a) for a in node.args]
        f = np.minimum.reduce if op == "and" else np.maximum.reduce
        return f([p[0] for p in parts]), f([p[1] for p in parts])
    w_lo = int(round(node.lo / dt))
    w_hi = INF if math.isinf(node.hi) else int(round(node.hi / dt))
    if op in ("G", "F"):
        lo, hi = rec(node.args[0], need + w_hi)
        how = "min" if op == "G" else "max"
        return _future(lo, w_lo, w_hi, how), _future(hi, w_lo, w_hi, how)
    if op in ("H", "O"):
        lo, hi = rec(node.args[0])
        how = "min" if op == "H" else "max"
        return _past(lo, w_lo, w_hi, how), _past(hi, w_lo, w_hi, how)
    if op == "U":
        a_lo, a_hi = rec(node.args[0], need + w_hi)
        b_lo, b_hi = rec(node.args[1], need + w_hi)
        return _until(a_lo, b_lo, w_lo, w_hi, need), _until(a_hi, b_hi, w_lo, w_hi, need)
    raise ValueError(f"unknown operator {op}")


def interval(spec: Node, trace: Trace, regions=None, registry: Registry | None = None,
             max_speed: float = 15.0) -> tuple[float, float]:
    """Robust satisfaction interval at t=0 of a flight still in progress
    (the trace ends at "now")."""
    known = trace.n
    n = max(known, int(round(spec.horizon() / trace.dt)) + 1)
    lo, hi = _rho_iv(spec, trace.padded(n), regions, known, registry, max_speed, 1)
    return float(lo[0]), float(hi[0])


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "satisfied"
    if hi <= 0:
        return "violated"
    return "undecided"


def fmt(v: float):
    """JSON has no infinity."""
    return ("inf" if v > 0 else "-inf") if math.isinf(v) else round(float(v), 3)
