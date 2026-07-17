# Data & Compliance

## What “Real Data” Means Here

In this project, “real” observations refer to:
1. **Door-to-door (D2D) segment travel times** derived from operational sources (e.g., GPS/AVL streams), which contain non-transport operational semantics (holding, layover, schedule recovery) and are audited during validation.
2. **Moving-only link speeds** (excluding dwell times), which are supplied to the L2 Inner Loop (IES) for traffic-state reconciliation.

These observations require an **Observation Operator Audit** (Rule C, compared against MAD and Isolation Forest baselines) before being used as validation evidence to prevent "explaining away" regime errors.

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

1. **Without any real data or SUMO installed:** Running the smoke test `scripts/smoke/p14_smoke.py` uses synthetic fixtures and successfully produces mock representations of:
   - **Figure 2 (Audit Geometry & Ghost Jams):** Separating transport and non-transport regimes.
   - **Figure 4 (Robustness Speed CDFs):** Demonstrating distributional alignment.
2. **With SUMO + real processed inputs available locally:** The full RCMDT calibration (L1 BO + L2 IES) and validation pipeline (including MAD and Isolation Forest audit baselines) can be executed using the paths specified in `README.md`.