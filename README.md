# RCMDT: Robust Calibration of Mobility Digital Twins
**Codebase for IEEE SMC 2026 Submission**

## 📌 Replication Guide

Use this guide to reproduce the key experimental results, specifically the **Baseline (B1)** verification and the **Zero-Shot Robustness (P14)** test.

### 1. Prerequisites

*   **OS**: Windows 10/11 (Validated) or Linux
*   **Python**: 3.11+
*   **SUMO**: 1.20.0 (Must be in system `PATH`)
    *   Verify with: `sumo --version`

### 2. Setup

Install required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scipy shapely geopandas
```

### 3. Data Structure

*   `data/` : **Training Data** (Peak Hour 17:00-18:00) - For Calibration (L1/L2)
*   `data2/` : **Testing Data** (Off-Peak Hour 15:00-16:00) - For Zero-Shot Validation
*   `sumo/` : Simulation Files (Net, Routes, Configs)
    *   `sumo/net/hk_cropped.net.xml`: Cropped Research Area
*   `scripts/` : Python Analysis & Utility Scripts
*   `plots/` :  Generated Figures (Output Directory)

---

### 4. Reproduction Steps

#### Phase 1: Baseline Verification (Experiment B1)
*Goal: Verify the physical baseline under free-flow conditions (No background traffic).*

1.  **Run Simulation** (Freeflow):
    ```bash
    sumo -c sumo/config/baseline_b1.sumocfg
    ```
    *   Expected Runtime: < 10 seconds
    *   Output: `sumo/output/stopinfo_b1.xml` (approx. 44KB)

2.  **Generate Plots**:
    ```bash
    python scripts/evaluate_baseline.py
    ```
    *   Outputs: `plots/spacetime_*_baseline.png` (High-Vis Blue Trajectories)

#### Phase 1.5: Calibration Verification (Experiment B2)
*Goal: Verify the performance of the Optimized L1 Parameters.*

1.  **Run Simulation** (Calibrated + Background):
    ```bash
    sumo -c sumo/config/experiment2_calibrated.sumocfg
    ```
2.  **Verify RMSE**:
    ```bash
    python scripts/metrics_calc.py --real_links "data/processed/link_speeds.csv" --real_dist "data/processed/kmb_route_stop_dist.csv" --sim "sumo/output/stopinfo_exp2.xml" --out "docs/b2_metrics_68X.csv" --route 68X
    ```
    *   *Target*: RMSE < 160s.

#### [Advanced] Phase 1.0: L1 Micro-Inversion (Bayesian Optimization)
*Goal: Reproduce the parameter inversion process (The "Outer Loop").*
*(Warning: Computationally Expensive, ~6-10 hours)*

1.  **Run Optimization**:
    ```bash
    python scripts/calibration/run_calibration_l1_loop.py --rounds 25 --init_points 15
    ```
2.  **Plot Convergence** (Fig 2 in Paper):
    ```bash
    python scripts/calibration/plot_calibration_convergence.py --log data/calibration/B2_log.csv
    ```
    *   Output: `plots/l1_calibration_convergence.png`

#### [Advanced] Phase 1.6: L2 Macro-Assimilation (IES) (Experiment B4)
*Goal: Reproduce the Iterative Ensemble Smoother (The "Inner Loop").*

1.  **Run IES Loop**:
    ```bash
    python scripts/calibration/run_ies_loop.py --ensemble_size 8 --iterations 5
    ```
    *   Output: `data/calibration/ies_results.csv`

#### Phase 2: Zero-Shot Robustness (Experiment P14)
*Goal: Validate generalization to off-peak regime with frozen parameters.*

1.  **Run Simulation** (Off-peak Demand with Frozen L2):
    ```bash
    sumo -c sumo/config/experiment_robustness.sumocfg
    ```
    *   Expected Runtime: ~1-2 minutes
    *   Output: `sumo/output/offpeak_stopinfo.xml`

2.  **Evaluate Metrics** (Statistical Analysis):
    ```bash
    python scripts/evaluate_robustness.py --sim sumo/output/offpeak_stopinfo.xml
    ```
    *   *Output*: Prints KS Distance, RMSE, and P90 Speed metrics to console.

3.  **Generate Visualizations** (SMC Figures):
    
    *   **Ghost Audit & Robustness CDF**:
        ```bash
        python scripts/visualization/plot_p14_robustness.py --sim sumo/output/offpeak_stopinfo.xml
        ```
        *Output*: `plots/P14_ghost_audit.png`, `plots/P14_robustness_cdf.png`

    *   **Spacetime Diagrams** (Off-peak):
        ```bash
        python scripts/plot_spacetime.py --real_links data2/processed/link_times_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/offpeak_spacetime_68X.png --route 68X --hour 15 --ghost --t_critical 325
        python scripts/plot_spacetime.py --real_links data2/processed/link_times_offpeak.csv --real_dist data/processed/kmb_route_stop_dist.csv --sim sumo/output/offpeak_stopinfo.xml --out plots/offpeak_spacetime_960.png --route 960 --hour 15 --ghost --t_critical 325
        ```
        *Output*: `plots/offpeak_spacetime_*.png`

---

### 5. Troubleshooting

*   **Empty Plots / No Data**:
    *   Check `sumo/output/*.xml` file sizes. If < 2KB, the simulation failed or produced no output. Check `logs/` for errors.
*   **KS > 0.5 (Fail) in P14**:
    *   Ensure `evaluate_robustness.py` is successfully filtering "Ghost Jams". The default rule is **Rule C (T* = 325s)**. If the raw data is used directly, KS will naturally be > 0.5 due to the Measurement Model Mismatch.
