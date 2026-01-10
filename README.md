# RCMDT: Observation-Operator-Aware Calibration of Mobility Digital Twins

Codebase for an IEEE SMC 2026 paper on robust calibration of bus corridor digital twins.

## What This Project Claims

This repo implements and evaluates **RCMDT**, a hierarchical calibration loop (BO outer loop + IES inner loop) for bus corridor mobility digital twins, with a paper contribution focused on:

- **Observation Operator Audit:** “Real” door-to-door (D2D) measurements can be contaminated by non-transport operational semantics (schedule adherence, holding/layover). Robust conclusions require an auditable observation operator.
- **Regime Separation (Rule C):** We isolate non-transport “ghost jams” with a reproducible rule: flag samples if `(T > T*) ∧ (v_eff < v*)` using `T* = 325s`, `v* = 5 km/h`.
- **Robustness Evidence:** Robustness is validated at the distribution level (K-S test + worst-window stress test). The K-S test is used as validation evidence.
- **Diagnostics (Validation Scope):** Trajectory decomposition and dwell/holding-proxy plots answer where mechanisms differ.

## Paper Alignment

- Data/compliance notes: `DATA.md`
- Main P14 figures (already generated): `plots/P14_ghost_audit.png`, `plots/P14_robustness_cdf.png`

## Repository Layout

- `scripts/`: analysis, calibration, visualization utilities
- `sumo/`: SUMO configs, networks, routes, outputs
- `plots/`: paper figures and diagnostics

## Requirements

- Windows 10/11 (tested), Python 3.11+, SUMO 1.20.0
- Python packages:
  ```bash
  pip install -r requirements.txt
  ```

## Smoke Test (No Real Data / No SUMO)

This repo includes synthetic fixtures so you can run a minimal end-to-end check without any private data:

```bash
python scripts/smoke/p14_smoke.py
```

Or in PowerShell:
```powershell
.\reproduce.ps1
```

## Optional Diagnostics (Use For Mechanism Explanations Only)

Trajectory decomposition (stepped/full-time vs traffic-only) is diagnostic:

```bash
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_68X.png --route 68X --t_critical 325 --speed_kmh 5
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_960.png --route 960 --t_critical 325 --speed_kmh 5
```

Holding proxy vs simulated dwell (diagnostic):
```bash
python scripts/visualization/plot_dwell_distribution.py
```

## Calibration Runs (Advanced / Expensive)

- L1 Bayesian optimization (outer loop): `scripts/calibration/`
- L2 IES assimilation (inner loop): `scripts/calibration/`

## Data Introduction

The data are private. The transit data files are available at folder `sim/data/`. For privacy reason, we have replaced the orignal real world data.

## Troubleshooting

- SUMO outputs empty/small: check `sumo/output/*.xml` sizes and `sumo/output/*.log` logs.
- P14 "raw" K-S is high (~0.5+): this is expected before observation operator audit / regime separation (see `plots/P14_ghost_audit.png`).
