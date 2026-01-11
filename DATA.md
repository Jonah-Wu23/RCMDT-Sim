# Data & Compliance

## What “Real Data” Means Here

In this project, “real” observations primarily refer to **door-to-door (D2D) stop-to-stop link statistics** derived from operational sources (e.g., ETA / stop event feeds). These observations can be contaminated by **non-transport regimes** (schedule adherence / holding / layover) and therefore require an **Observation Operator Audit** before being used as validation evidence.

## Included vs Not Included

- Included:
  - Minimal, synthetic fixtures for CI smoke testing: `tests/fixtures/p14/`
  - Example figures generated from prior runs: `plots/` (as committed in this repo)
- Not guaranteed to be included in all distributions:
  - Full raw operational datasets / API pulls
  - Large SUMO outputs (e.g., full FCD logs)

## Sensitive Information

- Do not commit API keys or raw identifiers.
- If you add new data files, ensure they are either:
  - Publicly redistributable, or
  - Properly anonymized/aggregated so that no PII is present.

## Reproducibility Targets

The reproducibility contract for the open-source repo is:

1. **Without any real data or SUMO installed:** `scripts/smoke/p14_smoke.py` runs using fixtures and produces `P14_ghost_audit.png` + `P14_robustness_cdf.png`.
2. **With SUMO + real processed inputs available locally:** the P14 pipeline can be re-run using the paths in `README.md`.

