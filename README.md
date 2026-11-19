# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riansap88/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Riansap88/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the percent of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95) per trophy.

## Run it

```sh
python3 main.py
```

## Program

`main.py` (Python 3.13, standard library only). It reads every `*.in` case in
the input directory (`$MAKEFUN_INPUT_DIR`, default `/input`) and writes one
`*.out` per case to the output directory (`$MAKEFUN_OUTPUT_DIR`, default
`/output`). Two runs on the same case write byte-identical output: no network,
no clock, no randomness.

## Method

A trophy's rate is `rate(t) = E(t) / P(t)`, where `P` is the players-tracked
count and `E` is that trophy's earner count. The two move for different
reasons (new players join tracking; players finish content), so they are
modelled separately and divided at the target:

- `P(t)`: least-squares fit of log `P` against log(1 + hours since launch)
  over the distinct reads, mildly damped, extrapolated to the target and
  floored at the last observed `P`.
- `E_i(t)`: per trophy, least-squares fit of log `E` against log(1 + hours
  since launch). The slope is shrunk toward the tier mean, so trophies with
  thin history borrow strength from their tier. The forecast is floored at
  the last observed earner count and capped at half of `P_hat`.
- The median forecast is `100 * E_hat / P_hat`.
- Spread (logit scale) combines a random-walk term growing with the square
  root of the horizon, the median observed refresh jump between distinct
  reads, and the fit residuals. With fewer than three distinct reads no
  trend can be fit, so the median stays at the last rate (shifted down half
  a logit for the next refresh's dilution) and the spread uses wide priors.

Identical consecutive reads are deduplicated first: Exophase refreshes
rarely, so long flat stretches carry no new information.

## Local validation

On the kit practice cases (`p00`, `p01`): deterministic across runs, 55
rows each, quantiles non-decreasing, 3,258 bytes per output, ~0.06 s per
run. Against the estimated practice answers (earners held, players tracked
3,868 after the first refresh) the estimated pinball loss is ~0.70, against
the house persistence baseline's ~1.51. The practice score pays nothing;
the final case is what counts.

---

<sub>Entered on [make.fun](https://make.fun/?ref=smredjmq&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
