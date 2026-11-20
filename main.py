"""Trophy rarity forecast: earners-and-players decomposition.

Why not persistence: an earn rate is earners / players_tracked. On launch
night the tracked player count multiplied several-fold at a single Exophase
refresh while earners barely moved, so nearly every rate FELL. A forecast
that holds the last rate misses that entirely. This program forecasts the
two sides separately and divides:

  players(t)  - the tracked-player count follows the launch growth curve:
                a line fitted to log(players) against log(1 + hours since
                release), extrapolated to the target and damped, because
                launch growth decays.
  earners_i(t) - each trophy's earner count grows like a progression curve:
                a line fitted to log(earners) against log(1 + hours), its
                slope shrunk toward the pooled slope of similar trophies
                (few reads per trophy early on), then saturated against a
                week-one ceiling for the trophy's kind, so a hot start
                cannot extrapolate past what a first week allows.

The median is earners / players at the target. The spread on the logit
scale combines the random-walk term the house baseline uses with a
dilution term that widens when the player count is forecast to grow a
lot inside the horizon (a refresh then moves every rate at once) and a
fit term from each trophy's own residuals.

Standard library only. Deterministic: no randomness, no wall clock.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside main.py would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RATE_FLOOR = 0.01
RATE_CEIL = 99.99


def clamp_rate(x: float) -> float:
    return min(RATE_CEIL, max(RATE_FLOOR, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(s: str) -> float:
    """ISO-8601 timestamp -> epoch seconds (UTC)."""
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def linfit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, residual_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, n


def week_one_ceiling(trophy) -> float:
    """A generous cap on the share of tracked players who can hold this
    trophy one week after launch, by the kind of trophy it is."""
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or ""
    if tier == "platinum":
        return 0.02
    if "100% completion" in desc:
        return 0.02
    if "complete the final mission" in desc:
        return 0.30
    if desc.startswith("complete chapter"):
        return 0.80
    if desc.startswith("complete gone fishing"):
        return 1.00
    if desc.startswith("complete "):
        return 0.65
    if "250" in desc or "100 locations" in desc or "50 challenges" in desc:
        return 0.10
    if "all " in desc and ("gold medal" in desc or "every" in desc):
        return 0.15
    return 0.50


def forecast_case(case) -> dict:
    trophies = case["trophies"]
    history = [h for h in case.get("history") or [] if h.get("rates") is not None]
    history.sort(key=lambda h: parse_time(h["at"]))
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at_t = parse_time(case["at"])
    target_t = parse_time(case["target"])
    horizon_days = max(0.0, (target_t - at_t) / 86400.0)

    def hours(ts: float) -> float:
        return max(0.0, (ts - release) / 3600.0)

    # ---- players: launch growth curve on log-log, damped extrapolation
    px, py = [], []
    for h in history:
        p = h.get("players_tracked")
        if p:
            px.append(math.log1p(hours(parse_time(h["at"]))))
            py.append(math.log(float(p)))
    p_last = float(history[-1]["players_tracked"]) if history else 1000.0
    x_last = math.log1p(hours(at_t))
    x_target = math.log1p(hours(target_t))
    if len(px) >= 2:
        a, b, _sd, _n = linfit(px, py)
        raw = math.exp(a + b * x_target)
        growth = raw / p_last if p_last > 0 else 1.0
        growth = min(15.0, max(1.0, growth))
        # Damping: launch surges decay; take the extrapolated growth at 70%
        # of its log distance.
        growth = math.exp(0.7 * math.log(growth))
    else:
        growth = 1.0
    players_target = p_last * growth

    # ---- per-trophy earner fits
    n_t = len(trophies)
    last_rate = [None] * n_t
    slopes = [None] * n_t
    fit_sd = [0.0] * n_t
    for i in range(n_t):
        xs, ys = [], []
        for h in history:
            rates = h.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                r = float(rates[i])
                last_rate[i] = r
                p = float(h.get("players_tracked") or 0)
                if p > 0:
                    earners = max(0.35, r / 100.0 * p)
                    xs.append(math.log1p(hours(parse_time(h["at"]))))
                    ys.append(math.log(earners))
        if len(xs) >= 2:
            _a, b, sd, _n = linfit(xs, ys)
            slopes[i] = b
            fit_sd[i] = sd

    # Pool slopes: shrink each trophy's earner-growth slope toward the
    # median slope of trophies whose last rate is in the same band.
    def band(r):
        if r is None:
            return 3
        if r >= 20:
            return 0
        if r >= 2:
            return 1
        if r >= 0.2:
            return 2
        return 3

    band_slopes = {}
    for i in range(n_t):
        if slopes[i] is not None:
            band_slopes.setdefault(band(last_rate[i]), []).append(slopes[i])
    band_median = {}
    for bd, vals in band_slopes.items():
        vals = sorted(vals)
        band_median[bd] = vals[len(vals) // 2]

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rate[i] if last_rate[i] is not None else 0.05
        if r_last >= 99.9:
            rows.append({"id": tr["id"], "q": [99.90, 99.95, 99.97, 99.98, 99.99]})
            continue
        e_last = max(0.35, r_last / 100.0 * p_last)
        slope = slopes[i]
        pooled = band_median.get(band(r_last), 0.6)
        if slope is None:
            slope = pooled
        else:
            slope = 0.55 * slope + 0.45 * pooled
        # Saturation: growth exponents fade over long horizons.
        slope *= math.exp(-0.10 * horizon_days)
        slope = min(2.2, max(0.0, slope))
        e_target = e_last * math.exp(slope * (x_target - x_last))
        ceiling = week_one_ceiling(tr) * players_target
        if e_target > ceiling:
            # Blend toward the ceiling instead of a hard clip.
            e_target = ceiling + (e_target - ceiling) * 0.15
        rate_med = clamp_rate(100.0 * e_target / players_target)

        sigma = 0.30 + 0.50 * math.sqrt(horizon_days)
        if growth > 1.8:
            sigma += 0.22 * math.log(growth)
        sigma += min(0.45, fit_sd[i])
        if r_last <= 0.05:
            sigma += 0.25
        centre = logit(rate_med)
        q = [clamp_rate(inv_logit(centre + z * sigma)) for z in Z]
        q = sorted(q)
        rows.append({"id": tr["id"], "q": [round(v, 4) for v in q]})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        name = path.name[: -len(".in")] + ".out"
        (OUTPUT_DIR / name).write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
