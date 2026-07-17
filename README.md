# RCMDT: Operator-Aware Auditing and Robust Calibration of Bus-Corridor Digital Twins

Research-oriented Chinese overview for faculty outreach: `README_research_zh.md`

Codebase for the IEEE SMC 2026 paper: *"Operator-Aware Auditing and Robust Calibration of Bus-Corridor Digital Twins Using Bayesian Optimization and Iterative Ensemble Smoothing"*.

## What This Project Claims & Implements

This repo implements and evaluates **RCMDT**, a hierarchical calibration loop (BO outer loop + IES inner loop) with a strict scope and freeze contract. Core contributions included in this codebase:

- **Observation Operator Audit & Regime Separation:** “Real” door-to-door (D2D) measurements can be contaminated by non-transport operational semantics. We isolate non-transport “ghost jams” using a predefined **Rule C** (`T > 325s` ∧ `v_eff < 5 km/h`).
- **Data-Driven Audit Baselines:** The codebase includes comparative implementations of **MAD (Median Absolute Deviation)** and **Isolation Forest** to validate the superiority of the predefined Rule C for this specific operational context.
- **A0-A4 Common-Protocol Ablation:** We provide a strict experimental protocol to test mechanisms in isolation:
  - **A0:** Zero-shot (Baseline)
  - **A1:** BO-only (Outer loop only)
  - **A2:** IES-only (Inner loop only)
  - **A3:** Raw-RCMDT (Both loops, un-audited raw data)
  - **A4:** Full-RCMDT (Both loops, audited by Rule C)
- **Cross-Day Robustness:** Calibration parameters and audit thresholds are frozen on development data (Dec 19) and evaluated on a cross-day test split (Dec 30) using distribution-level evidence (K-S tests).

## Repository Layout

- `scripts/`: Analysis, calibration, and filtering utilities (including Rule C, MAD, Isolation Forest).
- `sumo/`: SUMO configs, networks, routes, and outputs.
- `plots/`: Generated paper figures and diagnostics (CDFs, Trajectories, BO vs LHS, etc.).
- `sim/data/`: Synthetic/Anonymized transit data files. *(Note: Original real-world data has been replaced/anonymized for privacy reasons).*

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

## Reproducing Paper Results & Figures

### 1. Operator Audit & Baseline Comparison (Figure 2 & 3)
To run the audit filtering (Rule C vs. MAD vs. Isolation Forest) and generate the robustness and consistency diagnostics:
```bash
# Provide the exact script you use to generate Fig 3 (Audit Robustness)
# e.g., python scripts/analysis/compare_audits.py
```

### 2. A0-A4 Ablation & Cross-Day CDFs (Figure 4)
Scripts to run the isolated configurations and generate the Cumulative Distribution Functions for cross-day validation:
```bash
# e.g., python scripts/visualization/plot_cross_day_cdf.py
```

### 3. Optimization Efficiency (BO vs LHS) (Figure 5)
```bash
# e.g., python scripts/visualization/plot_optimization_efficiency.py
```

## Optional Diagnostics (Mechanism Explanations)

Trajectory decomposition (stepped/full-time vs traffic-only) helps visualize how operational stops distort D2D speed:

```bash
python scripts/visualization/plot_trajectory_stepped.py --real_links data2/processed/link_stats_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/trajectory_stepped_68X.png --route 68X --t_critical 325 --speed_kmh 5
```

Holding proxy vs simulated dwell (diagnostic):
```bash
python scripts/visualization/plot_dwell_distribution.py
```

## Calibration Runs (Advanced / Expensive)

- **L1 Outer Loop (Bayesian Optimization / LHS):** See scripts in `scripts/calibration/` (Uses 40 evaluation budget).
- **L2 Inner Loop (Iterative Ensemble Smoother):** Uses 10 ensemble members and 3 iterations. Scripts located in `scripts/calibration/`.

## Troubleshooting

- **SUMO outputs empty/small:** Check `sumo/output/*.xml` sizes and `sumo/output/*.log` logs.
- **High K-S distance (~0.5+) on raw data:** This is expected before the observation operator audit (Rule C) separates the regimes.
- **IRN Unmatched records:** Sparse IRN matches are expected and limit the IRN check to a diagnostic level rather than a strict validation metric.