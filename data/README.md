# Data

## `sample_accidents.csv`

A small **synthetic** stand-in for the real dataset, shaped like
per-location accident statistics so `risk_scoring.py` has something
realistic to merge near-miss events against. Columns:

| column           | meaning                                   |
|-------------------|--------------------------------------------|
| `site_id`         | unique location identifier                |
| `name`            | human-readable location name              |
| `lat`, `lon`      | coordinates                               |
| `accidents_count` | recorded accidents in the period          |
| `fatalities`      | recorded fatalities                       |
| `injuries`        | recorded injuries                         |
| `years`           | period the counts cover                   |

## Swapping in the real MoRTH dataset

The problem statement references the **Road Accidents in India (MoRTH)**
dataset on [data.gov.in](https://data.gov.in). To use it instead of the
bundled sample:

1. Download the relevant CSV(s) from data.gov.in.
2. Aggregate/re-shape them to the same columns as `sample_accidents.csv`
   above (one row per location, with `site_id`/`lat`/`lon` — the raw MoRTH
   releases are often state/district-level, so you may need to join against
   a separate lat/lon lookup or geocode intersection names).
3. Point `config.SAMPLE_ACCIDENTS_CSV` (in `backend/app/config.py`) at the
   new file, or overwrite `sample_accidents.csv` in place.
4. Re-run `python scripts/run_pipeline.py`.

No other code changes are required — `risk_scoring.py` only depends on the
column names above.

## `simulated_traffic.mp4` (generated, not committed)

Produced by `scripts/generate_sample_video.py`. Not checked into git (see
`.gitignore`) since it's fully reproducible — regenerate it locally.
