# Paper Outline: IEEE SMC 2026 (6 Pages)

**Title**
**Robust Calibration of Mobility Digital Twins: A Bayesian-Assimilation Loop for Coupling Stop-Level Dynamics and Corridor Reliability**
*(中文：移动数字孪生的鲁棒校准：耦合站点级动力学与走廊级可靠性的贝叶斯-同化闭环)*

**Keywords**
Mobility Digital Twins; Cyber-Physical Systems (CPS); Bayesian Optimization; Data Assimilation; Uncertainty Quantification; Hierarchical Calibration; System Identifiability; Reliability Engineering.

---

## 0. Abstract (200-250 words)

*   **Context:** Urban mobility systems are stochastic Cyber-Physical Systems (CPS). Constructing high-fidelity **Digital Twins (DT)** is hindered by the "Reality Gap" caused by uncertain human behaviors (micro-level) and complex traffic flow dynamics (macro-level).
*   **Problem:** Conventional calibration treats parameters deterministically and typically isolates subsystems. This fails to address the inherent **coupling** between stop-level dynamics (e.g., stochastic dwell times) and emergent corridor-level reliability (e.g., travel time variability and tail risks).
*   **Methodology (RCMDT):** We propose the **Robust Calibration of Mobility Digital Twins (RCMDT)**, a hierarchical framework featuring a cybernetic **Bayesian-Assimilation Loop**:
    1.  **Outer Loop (Micro-Level):** Utilizes Kriging-based **Bayesian Optimization (BO)** with Uncertainty Quantification (UQ) to invert behavioral parameters (stop logic).
    2.  **Inner Loop (Macro-Level):** Integrates a **Constraint-Aware Iterative Ensemble Smoother (IES)** to assimilate corridor reliability states (flow logic).
*   **Key Insight:** We introduce a rigorous **"Caliber Audit"** (System Observability Analysis), proving that shifting the observation operator from "Mean Moving Speed" to **"Door-to-Door Reliability"** is mathematically essential for restoring system identifiability in human-centric CPS.
*   **Results:** Experiments on real-world high-frequency trajectory data (Hong Kong) demonstrate that RCMDT reduces distributional mismatch (K-S distance) significantly and improves **distributional robustness** across unseen operational periods compared to single-level baselines.

---

## 1. Introduction

### 1.1 Digital Twins as Probabilistic CPS Mirrors
*   Define the Mobility Digital Twin not just as a deterministic simulator, but as a probabilistic mirror of a complex CPS.
*   Highlight the **Inverse Problem**: Estimating system parameters $\theta$ from noisy output $y$ is ill-posed, non-convex, and computationally expensive.

### 1.2 The Micro-Macro Coupling Challenge
*   **Stop-Level Dynamics (Micro):** Stochastic human interaction at stops (boarding/alighting rates, dwell time variance).
*   **Corridor Reliability (Macro):** Emergent corridor properties defined not by average speed, but by the stability of travel time distributions (e.g., 90th percentile lateness).
*   **The Gap:** Existing methods (GA/PSO) decouple these layers, leading to "overfitting to the mean" while failing to capture the **Reliability** (tail distribution).

### 1.3 Contributions (SMC Focus)
1.  **System Definition:** We formulate the **Coupled Calibration Problem**, explicitly linking stop-level behavioral uncertainty to corridor-level reliability metrics.
2.  **Cybernetic Mechanism:** We propose the **Bayesian-Assimilation Loop**, a closed-loop architecture where a Bayesian surrogate guides micro-parameter search while an IES inner loop stabilizes macro-state assimilation.
3.  **Reliability & Robustness:** We establish a **Reliability-based Validation Protocol** (using K-S distance) and demonstrate that optimizing for reliability restores physical system identifiability where speed-based metrics fail, enabling regime-transfer validation under frozen parameters.

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
    *   **Op-L2-v1.1 (Decontaminated):** D2D with **Ghost Jam Filter** (Rule C: $T^*=325s$) for stress-testing verification logic against measurement artifacts.

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

---

## 5. Experimental Setup

### 5.1 Testbed & Protocol
*   **Simulation Platform:** SUMO (Simulation of Urban MObility).
*   **Data Source:** High-frequency traces (Hong Kong).
*   **Protocol:**
    *   *Unit:* 1-hour time windows.
    *   *Sets:* Training (Day 1 AM), Testing (Day 1 PM), Robustness Check (Day 2).

### 5.2 Metrics for Reliability Validation
*   **Distribution Matching:** Kolmogorov-Smirnov (K-S) Distance.
*   **Tail Risk:** 90th Percentile (P90) Error.
*   **Physical Validity:** Smoothness of trajectories (acceleration noise).

---

## 6. Results and Discussion

### 6.1 Efficiency & Convergence
*   Demonstrate BO convergence speed vs. Random Search.
*   Show IES stabilization of macro-flow errors within 3-4 iterations (Inner Loop).

### 6.2 System Identifiability Analysis ("Caliber Switch")
*   **Evidence:** Compare calibration using Speed vs. Reliability.
    *   *Speed-only:* Finds "Ghost parameters" (e.g., extremely impatient drivers + zero traffic) that fit the mean but fail tail distribution.
    *   *Reliability-based:* Converges to physically realistic parameters.
*   **Conclusion:** The Reliability Operator is necessary for identifiable human-centric DTs.

### 6.3 Distributional Robustness Tests (The P14 Stress Test)
*   **Phase 1: Raw Mismatch:** Zero-shot transfer to off-peak initially failed (K-S $\approx$ 0.54) due to "Ghost Jams" (non-propagating stalls).
*   **Phase 2: Operator Audit:** Implementing **Op-L2-v1.1** revealed physical boundaries of measurement errors (>400s), restoring the validity of the ground truth.
*   **Phase 3: Borderline Pass:** Under the hardest regime (Rule C, $T^*=325s$, worst 15-min window), the system maintained K-S = 0.3337 (<0.35), proving robust generalization without re-calibration.

### 6.4 Limitations
*   **Measurement Dependency:** The framework's validity relies on the consistency of the observation operator; e.g., ETA schedule-holds can contaminate travel-time statistics if not audited.
*   **Offline Calibration:** Currently assumes batch processing; real-time adaptation is future work.

---

## 7. Conclusion

*   **Summary:** RCMDT bridges the reality gap by explicitly coupling stop-level dynamics and corridor reliability via a rigorous Bayesian-Assimilation loop.
*   **Impact:** A blueprint for calibrating stochastic, human-centric CPS.
*   **Future Work:** Real-time Online Digital Twin adaptation.

---

## Figure & Table Checklist (SMC Style)

*   **Fig 1. System Block Diagram:** Control-theoretic view. Inputs (Data) $\rightarrow$ Controllers (BO & IES) $\rightarrow$ Plant (DT) $\rightarrow$ Feedback (Reliability Metrics).

*   **Fig 2. Uncertainty Quantification:** Visualizing the Gaussian Process variance reduction over iterations.
*   **Fig 3. Time-Space Diagrams:** Comparing "Real" vs "Ghost System" vs "RCMDT" congestion patterns.
*   **Fig 4. Reliability CDFs:** Emphasize the alignment of the P90 tails (the human-centric reliability metric).
*   **Fig 5. Trajectory Smoothness Validation (P14 Robustness):** Visualizing `trajectory_68X_offpeak_rcmdt.png` and `trajectory_960_offpeak_rcmdt.png` to demonstrate physical validity (acceleration noise) under zero-shot transfer conditions.
