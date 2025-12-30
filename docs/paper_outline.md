# Paper Outline: IEEE SMC 2026 (6 Pages)

**Title**
**Robust Calibration of Mobility Digital Twins: A Bayesian-Assimilation Loop for Coupling Stop-Level Dynamics and Corridor Reliability**
*(中文：移动数字孪生的鲁棒校准：耦合站点级动力学与走廊级可靠性的贝叶斯-同化闭环)*

**Keywords**
Mobility Digital Twins; Bus Operations; Schedule Adherence; Observation Operator Audit; Regime Separation; Bayesian Optimization; Data Assimilation; Distributional Robustness.

---

## 0. Abstract (200-250 words)

*   **Context:** Urban bus operations are stochastic Cyber-Physical Systems (CPS). The “reality gap” is driven not only by traffic physics, but also by **operational semantics** (e.g., schedule adherence, holding/layover) that can contaminate door-to-door measurements with non-transport regimes.
*   **Problem:** Calibration based on mean moving speed or un-audited travel-time labels becomes **unidentifiable** (equifinality) and brittle under regime transfer, because it conflates transport dynamics with operator-induced artifacts.
*   **Methodology (RCMDT):** We propose the **Robust Calibration of Mobility Digital Twins (RCMDT)**, a hierarchical framework featuring a cybernetic **Bayesian-Assimilation Loop**:
    1.  **Outer Loop (Micro-Level):** Utilizes Kriging-based **Bayesian Optimization (BO)** with Uncertainty Quantification (UQ) to invert behavioral parameters (stop logic).
    2.  **Inner Loop (Macro-Level):** Integrates a **Constraint-Aware Iterative Ensemble Smoother (IES)** to assimilate corridor reliability states (flow logic).
*   **Operator Audit (Caliber Audit):** We formalize an auditable **observation-operator family** (Op-L2-v0/v1/v1.1) and a physically motivated **regime-separation rule** (Rule C: $T^*=325s$, $v^*=5$ km/h) to decontaminate non-transport “ghost jams”.
*   **Results:** On two Hong Kong bus corridors (68X, 960), raw off-peak transfer fails (K-S $\approx$ 0.54) but passes after auditable decontamination (K-S $\approx$ 0.26), with the worst 15-min window still borderline-pass (K-S $\approx$ 0.33) under the strictest valid operator. Sparse real trajectories are used only as diagnostics; claims rely on distribution-level evidence and kinematic regime separation.

---

## 1. Introduction

### 1.1 Digital Twins as Probabilistic CPS Mirrors
*   Define the Mobility Digital Twin not just as a deterministic simulator, but as a probabilistic mirror of a complex CPS.
*   Highlight the **Inverse Problem**: Estimating system parameters $\theta$ from noisy output $y$ is ill-posed, non-convex, and computationally expensive.

### 1.2 The Micro-Macro Coupling Challenge
*   **Stop-Level Dynamics (Micro):** Passenger processes and operational control at stops (boarding/alighting, holding/layover), inducing long-tail stop/idle times.
*   **Corridor Reliability (Macro):** Emergent corridor properties defined not by average speed, but by the stability of travel time distributions (e.g., 90th percentile lateness).
*   **The Gap:** Existing methods (GA/PSO) decouple these layers, leading to "overfitting to the mean" while failing to capture the **Reliability** (tail distribution).

### 1.3 Contributions (SMC Focus)
1.  **Operator-Aware Observability:** We introduce a rigorous **Caliber Audit** (Op-L2 family) and auditable **regime separation** (Rule C) to decontaminate non-transport artifacts and restore identifiability in human-centric CPS.
2.  **Cybernetic Mechanism:** We propose the **Bayesian-Assimilation Loop**, a closed-loop architecture where a Bayesian surrogate guides micro-parameter search while an IES inner loop stabilizes macro-state assimilation.
3.  **Reliability-based Robust Validation:** We establish a distribution-level validation protocol (K-S + worst-window stress test) and demonstrate regime-transfer robustness under frozen parameters.

---

## 2. Related Work

### 2.1 Simulation-based Optimization in CPS
*   Evolution from Black-box (GA) to **Surrogate-Assisted Methods** (Bayesian Optimization).
*   *Gap:* Most focus on deterministic error minimization, neglecting **Distributional Robustness** and uncertainty quantification.

### 2.2 Data Assimilation for Dynamic Systems
*   Kalman Filters (EnKF) and Ensemble Smoothers (IES).
*   *Gap:* Typically used for online state correction; rarely coupled with **behavioral parameter inversion** in a unified hierarchical loop.

### 2.3 Robustness & Uncertainty in System Identification
*   Distributional metrics (Wasserstein, K-S test).
*   *Gap:* Lack of frameworks addressing identifiability issues caused by **observation operator misalignment** in human-centric systems.

---

## 3. Problem Formulation

### 3.1 System Dynamics & State Space
*   Stochastic Mapping $\mathcal{M}$: $Y_{reliability} = \mathcal{M}(\theta_{stop}, \mathbf{X}_{corridor}, \xi)$.
    *   $\theta_{stop}$: Micro-behavioral parameters (e.g., boarding/alighting coefficients).
    *   $\mathbf{X}_{corridor}$: Macro-state background traffic flows.
    *   $\xi$: Stochastic noise.

### 3.2 Definition: Corridor Reliability
*   Defined over a 1-hour sliding window.
*   **Reliability Vector** includes: Mean Travel Time, 90th Percentile (P90), and bounded worst-case error.
*   **Objective:** Minimize the divergence between Simulated and Real Reliability Distributions:
    $$ \min_{\theta, \mathbf{X}} \mathcal{D}_{KS}(P(Y_{sim}), P(Y_{real})) $$

### 3.3 Observability Analysis (The "Caliber Audit")
*   **Proposition:** Reliance on *Moving Speed* ($\mathcal{O}_{move}$) leads to equifinality.
*   **Layered Operators:**
    *   **Op-L2-v0 (Speed):** Naive moving average (Unidentifiable).
    *   **Op-L2-v1 (D2D):** Door-to-Door reliability including dwell dynamics.
    *   **Op-L2-v1.1 (Audited):** D2D with auditable **regime separation** (Rule C: $T^*=325s$, $v^*=5$ km/h) to exclude non-transport schedule-hold artifacts (“ghost jams”) in robustness verification.

---

## 4. Methodology: The RCMDT Framework

### 4.1 Architecture: The Bayesian-Assimilation Loop (Fig. 1)
*   **A "Control System" View:**
    *   **Outer Loop (Controller L1):** Bayesian Optimizer (BO) using Kriging Surrogates to propose $\theta_{stop}$.
    *   **Inner Loop (Controller L2):** IES Assimilator adjusting $\mathbf{X}_{corridor}$ to match reliability constraints.
    *   **Feedback:** Reliability Divergence (K-S, P90 error).

### 4.2 L1: Outer Loop - Surrogate-Driven Behavioral Inversion
*   **Gaussian Process (Kriging):** Models the objective landscape $f(\theta)$ with Mean (prediction) and Variance (uncertainty).
*   **Acquisition Strategy:** Expected Improvement (EI) guides the search to robust global optima.

### 4.3 L2: Inner Loop - Constraint-Aware Assimilation
*   **Observation Operator:** Maps simulator state to the **Reliability Vector** (not just mean speed).
*   **Constraint-Aware Update:**
    *   Standard IES update: $\mathbf{X}^{a} = \mathbf{X}^{f} + \mathbf{K}(\mathbf{d} - \mathbf{H}\mathbf{X}^{f})$.
    *   **Safety Rails:** explicit bounds (e.g., non-negative flow, max density) injected to ensure physical consistency.

### 4.4 Operator-Aware Observation & Diagnostic Decomposition
*   **(M1) Auditable Rule C definition:** A sample is flagged as non-transport if $(T > T^*) \wedge (v_{eff} < v^*)$, with $T^*=325s$, $v^*=5$ km/h. The goal is not to “make data look like simulation”, but to separate operator artifacts for a reproducible robustness test.
*   **(M2) Holding proxy vs simulated dwell:** The holding proxy is constructed from observable real signals and is not the same physical quantity as simulated dwell. Dwell/holding plots are used to localize mechanism gaps, not as strict calibration targets.
*   **(M3) Traffic-only definition:** Traffic-only segments satisfy $v \ge 5$ km/h and exclude stop/idle phases. When plotting trajectories, Real/Sim axes can be traffic-only accumulated time, so full-time vs traffic-only comparisons are diagnostic (not equivalence claims).

---

## 5. Experimental Setup

### 5.1 Testbed & Protocol
*   **Simulation Platform:** SUMO (Simulation of Urban MObility).
*   **Data Source:** High-frequency traces (Hong Kong), two corridors (68X, 960).
*   **Protocol:**
    *   *Unit:* 1-hour time windows.
    *   *Sets:* Training (Day 1 AM), Testing (Day 1 PM), Robustness Check (Day 2).

### 5.2 Metrics for Reliability Validation
*   **Distribution Matching:** Kolmogorov-Smirnov (K-S) distance on D2D travel-time CDFs (reported for raw vs audited-clean vs simulation).
*   **Tail/Stress Metric:** Worst-window (15-min) K-S (and/or P90 error) under a fixed operator (Rule C) for robustness claims.
*   **Kinematic Plausibility (Diagnostic):** Regime separation in $(T, v_{eff})$ and traffic-only trajectory decomposition (not “trajectory smoothness”).

---

## 6. Results and Discussion

### 6.1 Measurement Audit & Regime Separation (Op-L2-v1.1 / Rule C)
*   **Figure:** `plots/P14_ghost_audit.png`
*   **One-line conclusion:** Rule C systematically isolates long-duration, low-speed samples (the raw long tail) from the transport regime.
*   **Reviewer-guard:** Rule C is an auditable threshold + decision boundary ($T^*=325s$, $v^*=5$ km/h), not manual cherry-picking.

### 6.2 Robustness Verification (Distribution-level, P14 Stress Test)
*   **Figure:** `plots/P14_robustness_cdf.png`
*   **One-line conclusion:** Zero-shot off-peak transfer fails under raw labels (K-S $\approx$ 0.54) but passes after auditable decontamination (K-S $\approx$ 0.26); worst 15-min window remains borderline-pass (K-S $\approx$ 0.33) under the strictest valid operator.
*   **Reviewer-guard:** This is the primary validation evidence; trajectory/spacetime plots are used only as diagnostics.

### 6.3 Kinematic Evidence for Ghost Artifacts (Trip/Segment Level)
*   **Figures:** `plots/ghost_physical_evidence_68X.png`, `plots/ghost_physical_evidence_960.png`
*   **One-line conclusion:** Ghost samples cluster in the $(T, v_{eff})$ region below $(T^*, v^*)$, showing consistent physical separation across corridors.
*   **Reviewer-guard:** Each plot reports `N_clean/N_ghost` to prevent “selected points” criticism.

### 6.4 Diagnostics Under Sparse Real Observations (Not Used for Validation)
*   **Figures:** `plots/trajectory_stepped_68X.png`, `plots/trajectory_stepped_960.png`
*   **One-line conclusion:** Full-time trajectory steps are dominated by stop/holding; traffic-only decomposition reveals the moving-physics “skeleton”.
*   **Scope note:** Real trajectories are sparse (Trips=1, Points=3/4), so these plots explain failure modes and operator choices rather than support validation claims.

### 6.5 Mechanism Gap Localization: Holding Proxy vs Simulated Dwell (Diagnostic)
*   **Figures:** `plots/dwell_distribution_68X.png`, `plots/dwell_distribution_960.png`
*   **One-line conclusion:** The holding proxy exhibits systematic shifts vs simulated dwell, indicating schedule adherence/layover as a major unmodeled mechanism.
*   **Reviewer-guard:** Proxy $\neq$ dwell; used as directional evidence for model extension, not strict calibration.

### 6.6 Calibration Improvement (B2): BO vs LHS
*   **Figure:** `plots/B2_phase_comparison.png`
*   **One-line conclusion:** BO reduces combined loss and improves at least one corridor RMSE relative to LHS.
*   **Reviewer-guard:** Multi-corridor objectives exhibit trade-offs; we avoid claiming uniform improvement across all routes/metrics.

### 6.7 Supplementary: Surrogate Uncertainty Dynamics (Non-monotonic)
*   **Figure:** `plots/B2_gp_variance.png`
*   **Interpretation:** Variance is non-monotonic (exploration jumps / noisy objective); included as a diagnostic, not a convergence proof.

### 6.8 Limitations / Threats to Validity (拆雷口径)
*   Sparse real trajectory observations $\rightarrow$ trajectory/spacetime used only as illustrative diagnostics.
*   Holding proxy vs dwell non-comparability $\rightarrow$ dwell figures are mechanism-gap signals, not strict validation.
*   Rule C threshold choice $\rightarrow$ defended via audit plot + K-S before/after + worst-window stress test with borderline operator.
*   Multi-corridor trade-offs $\rightarrow$ report phase-wise results; future work: multi-objective weighting / Pareto view.
*   Offline calibration $\rightarrow$ online adaptation and explicit holding control remain future work.

---

## 7. Conclusion

*   **Summary:** RCMDT bridges the reality gap by coupling stop-level dynamics and corridor reliability via a Bayesian-Assimilation loop, grounded in an auditable observation-operator (Caliber Audit).
*   **Impact:** A blueprint for calibrating stochastic, human-centric CPS.
*   **Future Work:** Real-time online DT adaptation and explicit modeling of schedule adherence/holding control.

---

## Figure & Table Checklist (SMC Style)

*   **Fig 1. System Block Diagram:** Control-theoretic view. Inputs (Data) $\rightarrow$ Controllers (BO & IES) $\rightarrow$ Plant (DT) $\rightarrow$ Feedback (Reliability Metrics).

### Main paper (recommend ≤6 figures)
*   **Fig 2. Measurement Audit + Rule C Logic:** `plots/P14_ghost_audit.png` (raw vs clean + auditable boundary).
*   **Fig 3. Robustness CDF (P14):** `plots/P14_robustness_cdf.png` (raw fail $\rightarrow$ clean pass; worst-window stress metric).
*   **Fig 4. Kinematic Ghost Evidence (2-panel):** `plots/ghost_physical_evidence_68X.png` + `plots/ghost_physical_evidence_960.png`.
*   **Fig 5. Phase-wise Improvement (B2):** `plots/B2_phase_comparison.png` (BO vs LHS; stability/variance discussion).
*   **Fig 6. Trajectory Decomposition (Diagnostic):** choose one of `plots/trajectory_stepped_68X.png` / `plots/trajectory_stepped_960.png` (explicitly labeled “illustrative; not used for validation”).

### Supplementary / Appendix (remaining plots)
*   `plots/dwell_distribution_68X.png`, `plots/dwell_distribution_960.png`
*   the other `plots/trajectory_stepped_*.png`
*   `plots/B2_gp_variance.png` (uncertainty dynamics; non-monotonic)
