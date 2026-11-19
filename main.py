# SPDX-License-Identifier: MIT
"""Forecast GTA VI trophy earn rates at a target time.

Reads every case in the input directory (schema makefun/gta6-trophies-case/v1)
and writes one forecast per case: five quantiles (5/25/50/75/95) of each
trophy's earn rate, the percent of Exophase-tracked players who will have
earned it at the case target.

Method
------
A trophy's rate is rate(t) = E(t) / P(t), where P is the players-tracked
count and E is that trophy's earner count.  The two move for different
reasons (new players join tracking; players finish content), so the
forecast models them separately and divides at the target:

* P(t): least-squares fit of log P against log(1 + hours since launch)
  over the distinct reads, mildly damped, extrapolated to the target and
  floored at the last observed P.
* E_i(t): per trophy, least-squares fit of log E against log(1 + hours
  since launch).  The slope is shrunk toward the tier mean, so trophies
  with thin history borrow strength from their tier, and the forecast is
  floored at the last observed earner count and capped at half of P_hat.
* The median forecast is 100 * E_hat / P_hat.
* Spread (logit scale) combines a random-walk term growing with the
  square root of the horizon, the median observed refresh jump between
  distinct reads, and the fit residuals.  With fewer than three distinct
  reads no trend can be fit, so the median stays at the last rate (shifted
  down half a logit to allow for the next refresh diluting rates) and the
  spread uses wide priors instead.

Only the Python standard library is used.  No network, no clock, no
randomness: two runs on the same case write byte-identical output.
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

MIN_RATE = 0.01
MAX_RATE = 99.99


def parse_time(value):
    """Parse an ISO-8601 timestamp with a Z suffix or an offset."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp_rate(rate):
    return min(max(float(rate), MIN_RATE), MAX_RATE)


def logit(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value >= 0.0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return 100.0 * exp_value / (1.0 + exp_value)


def round4(value):
    """Round to 4 decimals, half-up, deterministically."""
    return float(Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP))


def least_squares(xs, ys):
    """Return (slope, intercept) of the least-squares line through xs, ys."""
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, mean_y
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    if count % 2 == 1:
        return ordered[count // 2]
    return 0.5 * (ordered[count // 2 - 1] + ordered[count // 2])


def residual_std(xs, ys, slope, intercept):
    if len(xs) < 2:
        return 0.0
    total = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(total / len(xs))


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    count = len(trophies)
    tiers = [str(t.get("tier", "bronze")) for t in trophies]

    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        horizon_minutes = (target - at).total_seconds() / 60.0
    horizon_days = max(float(horizon_minutes), 0.0) / 1440.0

    game = case.get("game") or {}
    release = str(game.get("release_date", "2026-11-19"))
    if len(release) == 10:
        launch = parse_time(release + "T00:00:00Z")
    else:
        launch = parse_time(release)

    def hours_since_launch(moment):
        return max(0.0, (moment - launch).total_seconds() / 3600.0)

    def log_time(moment):
        return math.log1p(hours_since_launch(moment))

    # Last non-null rate per trophy, walking history backwards.
    last_rate = []
    for i in range(count):
        found = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                found = float(rates[i])
                break
        last_rate.append(found)

    # Tier median of the last rates, for trophies never observed.
    tier_rates = defaultdict(list)
    for i, rate in enumerate(last_rate):
        if rate is not None:
            tier_rates[tiers[i]].append(rate)
    tier_median_rate = {}
    for tier, values in tier_rates.items():
        tier_median_rate[tier] = median(values)
    all_last = [r for r in last_rate if r is not None]
    global_median_rate = median(all_last) if all_last else 0.5

    # Distinct reads (Exophase refreshes rarely; long flat stretches of
    # identical reads carry no new information), oldest first.
    distinct = []
    seen = set()
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key in seen:
            continue
        seen.add(key)
        distinct.append(read)

    # Per distinct read: (log-time, players tracked, earners per trophy).
    points = []
    for read in distinct:
        slot = read.get("slot") or read.get("at")
        if slot is None:
            continue
        players = read.get("players_tracked")
        if not players or players <= 0:
            continue
        rates = read.get("rates") or []
        earners = []
        for i in range(count):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                earners.append(None)
            else:
                earners.append(int(round(float(rate) * players / 100.0)))
        points.append((log_time(parse_time(slot)), players, earners))

    x_target = log_time(target)
    players_last = points[-1][1] if points else None

    # ---- Fit P(t): log players against log time. ----
    if len(points) >= 2:
        xs = [p[0] for p in points]
        ys = [math.log(p[1]) for p in points]
        slope_p, intercept_p = least_squares(xs, ys)
        resid_p = residual_std(xs, ys, slope_p, intercept_p)
    else:
        slope_p, resid_p = 0.0, 0.0
    if players_last is not None and len(points) >= 2 and slope_p > 0.0:
        fit_p = math.exp(intercept_p + slope_p * x_target)
        players_hat = players_last * (max(fit_p, players_last) / players_last) ** 0.85
        players_hat = min(players_hat, players_last * 50.0)
    elif players_last is not None:
        players_hat = float(players_last)
    else:
        players_hat = None

    # ---- Fit E_i(t) per trophy, with shrinkage toward the tier mean. ----
    raw_slopes = []
    fit_resid = []
    for i in range(count):
        xs, ys = [], []
        for x, _players, earners in points:
            earner = earners[i]
            if earner is not None and earner > 0:
                xs.append(x)
                ys.append(math.log(earner))
        if len(xs) >= 3:
            slope, intercept = least_squares(xs, ys)
            if slope < 0.0:
                slope = 0.0
            raw_slopes.append(slope)
            fit_resid.append(residual_std(xs, ys, slope, intercept))
        else:
            raw_slopes.append(None)
            fit_resid.append(None)

    tier_slopes = defaultdict(list)
    for i, slope in enumerate(raw_slopes):
        if slope is not None:
            tier_slopes[tiers[i]].append(slope)
    tier_mean_slope = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    valid_slopes = [s for s in raw_slopes if s is not None]
    global_mean_slope = sum(valid_slopes) / len(valid_slopes) if valid_slopes else 0.0

    shrunk_slopes = []
    for i, slope in enumerate(raw_slopes):
        tier_mean = tier_mean_slope.get(tiers[i], global_mean_slope)
        if slope is None:
            shrunk_slopes.append(tier_mean)
        else:
            shrunk_slopes.append(0.6 * slope + 0.4 * tier_mean)

    # Last observed earner count per trophy, with the log-time of the read
    # it came from (a trophy can be missing from the latest read, so the
    # growth must be measured from its own last observation, not the
    # latest read overall).
    earners_last = []
    earners_x = []
    for i in range(count):
        found = None
        found_x = None
        for x, _players, earners in reversed(points):
            if earners[i] is not None:
                found = earners[i]
                found_x = x
                break
        earners_last.append(found)
        earners_x.append(found_x)

    # ---- Refresh jumps: median absolute logit jump between distinct reads.
    jumps_all = []
    jumps_per_trophy = []
    for i in range(count):
        jumps = []
        previous = None
        for _x, players, earners in points:
            earner = earners[i]
            if earner is None:
                continue
            rate = 100.0 * earner / players
            if previous is not None:
                jumps.append(abs(logit(rate) - logit(previous)))
            previous = rate
        jumps_per_trophy.append(jumps)
        jumps_all.extend(jumps)
    global_jump = median(jumps_all)

    sigma_trend = 0.5 * math.sqrt(horizon_days)
    few_reads = len(points) < 3

    rows = []
    for i in range(count):
        rate = last_rate[i]
        if rate is None:
            rate = tier_median_rate.get(tiers[i], global_median_rate)
        if rate >= 99.99:
            # Everybody tracked already holds it; it can only stay there.
            rows.append((trophies[i]["id"], [99.99] * 5))
            continue

        if few_reads or players_hat is None:
            # No trend can be fit: hold the last rate, shifted down half a
            # logit for the next refresh's dilution, with wide priors.
            median_logit = logit(rate) - 0.5
            median_rate = inv_logit(median_logit)
            sigma = math.sqrt(sigma_trend ** 2 + 1.0 ** 2 + 0.5 ** 2)
        else:
            # Median from the E / P model at the target.
            base = earners_last[i]
            if base is None:
                base_rate = tier_median_rate.get(tiers[i], global_median_rate)
                earners_hat = players_hat * base_rate / 100.0
            elif base <= 0:
                earners_hat = 0.5 * math.exp(
                    shrunk_slopes[i] * (x_target - earners_x[i]))
            else:
                earners_hat = base * math.exp(
                    shrunk_slopes[i] * (x_target - earners_x[i]))
                earners_hat = max(earners_hat, float(base))
            earners_hat = min(earners_hat, 0.5 * players_hat)
            median_rate = 100.0 * earners_hat / players_hat
            median_rate = min(max(median_rate, MIN_RATE), MAX_RATE)

            refresh = median(jumps_per_trophy[i])
            if refresh is None:
                refresh = global_jump if global_jump is not None else 0.6
            refresh = min(refresh, 1.5)
            resid = fit_resid[i] if fit_resid[i] is not None else 0.5
            sigma_fit = math.sqrt(min(resid_p, 1.0) ** 2 + min(resid, 1.0) ** 2)
            sigma_fit = min(sigma_fit, 1.0)
            sigma = math.sqrt(sigma_trend ** 2 + refresh ** 2 + sigma_fit ** 2)

        sigma = min(max(sigma, 0.15), 2.5)
        center = logit(median_rate)
        quantiles = []
        for z in Z_VALUES:
            value = inv_logit(center + z * sigma)
            quantiles.append(round4(min(max(value, MIN_RATE), MAX_RATE)))
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append((trophies[i]["id"], quantiles))

    return rows


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        rows = forecast_case(case)
        case_id = str(case.get("id") or name[:-3])
        lines = ["{\"trophies\": ["]
        for index, (trophy_id, quantiles) in enumerate(rows):
            comma = "," if index < len(rows) - 1 else ""
            q_text = ", ".join(f"{q:.4f}" for q in quantiles)
            lines.append(f'{{"id": {trophy_id}, "q": [{q_text}]}}' + comma)
        lines.append("]}")
        out_name = case_id + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
