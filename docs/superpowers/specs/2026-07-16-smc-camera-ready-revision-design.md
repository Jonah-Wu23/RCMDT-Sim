# SMC 2026 Camera-Ready Revision Design

## 1. Objective

Revise the accepted SMC 2026 paper within the six-page limit by strengthening the evidence for operator-aware auditing and the two-level calibration protocol. The work includes targeted experiment reruns, source-code corrections, reproducible result generation, and tracked editing of the authoritative Word manuscript:

`D:\Documents\Bus Project\SMC\Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.docx`

The camera-ready submission deadline is July 26, 2026. July 26 is reserved for upload and submission-system issues; substantive work must finish by July 25.

## 2. Success Criteria

The revision is complete when all of the following conditions hold:

- The manuscript remains within six IEEE two-column pages, including references.
- Every reviewer comment is addressed through revised text, a corrected definition, a new comparison, or a verified experiment.
- Every reported number is traceable to a generated CSV field and a recorded experiment configuration.
- The full ablation uses common inputs, windows, seeds, budgets, and metrics.
- Rule C is compared with statistical and adaptive alternatives without claiming unsupported global optimality.
- The BO and LHS comparison uses the same simulation budget.
- The transfer experiment is described as cross-day transfer using the actual dates, December 19 and December 30, 2025.
- The final DOCX contains consistent equations, symbols, captions, affiliations, and cross-references and renders as a legible six-page PDF.

## 3. Non-Goals

- Developing a new Bayesian optimization or ensemble-smoother algorithm.
- Expanding the paper beyond six pages.
- Adding DAPPER, ES-MDA, or other baselines to the paper when their results remain indistinguishable or invalid after verification.
- Claiming that Rule C provides ground-truth semantic classification.
- Claiming statistical equivalence when a two-sample test still rejects equal distributions.
- Refactoring unrelated SUMO or data-processing code.

## 4. Current-State Findings

### 4.1 Manuscript

The authoritative DOCX contains five numbered figures, one table, thirteen references, no comments, and no tracked changes. Its text matches `SMC26_0008_MS.pdf`.

Material problems include:

- Related Work announces five streams but contains four.
- Equation (2) does not define the two quantities used by the error term.
- The GP section does not define the candidate vector or the candidate-selection step.
- L1 is described as stop-only although its configuration also contains bus car-following parameters.
- L1 and L2 both use a parameter named `minGap` for different vehicle populations.
- Rule C omits the 1500 m distance condition in the manuscript.
- Rule C alternates between strict and non-strict inequalities.
- The paper calls the December 30 data next-day data although the primary data are dated December 19.
- K-S values and percentage improvements differ between the results and conclusion.
- The conclusion overstates significance and generalization.

### 4.2 Source and Results

The repository already contains threshold sensitivity, semantic-alignment experiments, unified L2 ablations, DAPPER and smoother baselines, plotting scripts, and a smoke pipeline.

Existing outputs require verification:

- `data/calibration_v3/ablation/ablation_results_v3.csv` contains identical results for Base and `+BO`, and for `+Audit` and Full. This indicates that some configuration switches did not affect the evaluated simulations.
- `data/experiments_v4/a1_dapper_baselines/summary.csv` and `a1_smoother_baselines/summary.csv` report identical K-S values across methods.
- `data/experiments_v4/unified_l2/protocol_ablation/full_metrics.csv` has empty transfer fields.
- `data/calibration/B2_jl1_recalculated.csv` has missing simulation outputs for every row.
- `config/calibration/l2_priors.json` specifies 20 ensemble members and five iterations, while the paper protocol specifies 10 and three.

These outputs are evidence candidates, not camera-ready evidence, until their configuration and provenance checks pass.

## 5. Scientific Claim and Evidence Chain

The paper will be organized around this claim:

> Auditing observation semantics before calibration prevents operational artifacts from being absorbed into bus-behavior and background-traffic parameters, thereby improving transfer robustness.

Four evidence layers support the claim:

1. **Contamination exists.** Raw D2D observations contain a slow tail inconsistent with IRN traffic speeds and traffic-only trajectory accumulation.
2. **The fixed audit is stable.** Rule C has interpretable physical conditions and stable behavior across plausible threshold neighborhoods and evaluation windows.
3. **The layered protocol matters.** A common-protocol ablation distinguishes zero-shot, L1-only, L2-only, raw-observation RCMDT, and full RCMDT.
4. **Surrogate search is efficient.** BO and continued LHS are compared under an equal 40-evaluation budget.

The contribution language will emphasize the auditable observation operator, vehicle-scope separation, freeze protocol, and shift-sensitive validation. It will not present the mere combination of BO and IES as an algorithmic invention.

## 6. Terminology and Model Contract

### 6.1 L1 Parameters

Rename the manuscript's `theta_stop` to `theta_bus`. It contains:

- Stop-service parameters: `t_board`, `t_fixed`.
- Bus vehicle behavior parameters: `tau`, `sigma`, `minGap_bus`, `accel`, `decel`.

All of these parameters apply to the bus vehicle or bus stop process and remain frozen during L2 and validation.

### 6.2 L2 Context Vector

Define `x_corr` as a background-traffic context vector:

- `capacityFactor`
- `minGap_background`
- `impatience`

These variables apply to background traffic. The manuscript will avoid treating all three as direct dynamic traffic states.

### 6.3 Calibration Error

Align Equation (2) with `src/calibration/objective.py`. For matched downstream stop index `i`:

`e_i = mean simulated cumulative arrival time at stop i - mean observed cumulative arrival time at stop i`.

The manuscript must define the matching key, reference origin, sample count, route direction, and multi-seed aggregation.

### 6.4 Observation Audit

Use one Rule C definition everywhere:

- `T > 325 s`
- `v_eff < 5 km/h`
- `distance <= 1500 m`

Thresholds are selected using the training data and then frozen for cross-day evaluation. IRN remains an external plausibility check and does not tune the rule on the evaluation data.

## 7. Experiment Design

### 7.1 Common Protocol

Create or consolidate a single paper experiment manifest that records:

- Input file paths and dates.
- Route and direction filters.
- Training and cross-day windows.
- Rule C parameters.
- L1 and L2 parameter bounds.
- L1 budget: 15 shared LHS initial evaluations plus 25 subsequent evaluations.
- L2 settings: `Ne=10`, `K=3`, initial damping `0.3`.
- Seeds: five fixed seeds for the main ablation.
- Output directories and schema versions.

Every run writes a manifest copy and a hash of the effective configuration. A comparison script must reject configurations whose expected differences do not appear in the effective SUMO inputs.

### 7.2 E1: Observation-Rule Comparison

Compare:

- Fixed Rule C.
- IQR/MAD-based statistical filtering.
- Isolation Forest as a data-adaptive baseline.

Use training data for fitting or threshold selection and freeze each method before cross-day evaluation. Report retention rate, full-window K-S, worst-15-minute K-S, and IRN plausibility consistency. Since semantic ground-truth labels are unavailable, the result supports stability and interpretability, not classification accuracy or universal superiority.

Also compute a compact Rule C sensitivity grid around the selected point, such as `T in {275, 325, 375}` and `v in {4, 5, 6}`.

### 7.3 E2: Full Protocol Ablation

Run these configurations with identical seeds and windows:

| ID | Configuration | Audit | L1 BO | L2 IES | Observation semantic |
|---|---|---:|---:|---:|---|
| A0 | Zero-shot | No | No | No | Raw evaluation |
| A1 | BO-only | Fixed by protocol | Yes | No | Audited evaluation |
| A2 | IES-only | Fixed by protocol | No | Yes | Moving-only L2 |
| A3 | Raw-RCMDT | No | Yes | Yes | Raw D2D supplied to L2 |
| A4 | Full-RCMDT | Yes | Yes | Yes | Moving-only L2 |

For each configuration, report mean and standard deviation across five seeds for full-window K-S and worst-window K-S. Report cross-day K-S only when both real and simulated transfer inputs exist. Include sample counts.

The output validator fails when two configurations expected to differ share the same effective configuration hash, when required output files are absent, or when metrics are null.

### 7.4 E3: Equal-Budget BO Versus LHS

For each optimization seed:

- Start BO and LHS from the same 15-point LHS design.
- BO selects 25 additional candidates with expected improvement.
- Continued LHS evaluates 25 additional independently sampled candidates.
- Both methods therefore use 40 simulations.

Report cumulative best objective versus evaluation count, final best objective, and evaluations needed to reach a predeclared target. Use at least three optimization seeds. Retain all 40 evaluations per method; do not compare different budget sizes.

### 7.5 E4: Cross-Day Transfer

Use December 19, 2025 as the primary calibration/evaluation date and December 30, 2025 as the cross-day transfer date. Rename result fields from `next_day_*` to `cross_day_*`. The paper must state the exact dates and avoid next-day wording.

### 7.6 Optional Baselines

DAPPER and ES-MDA remain optional. They enter the paper only if:

- Each method actually applies a distinct update.
- Metrics vary beyond formatting precision where the final states differ.
- The comparison uses the same observations, priors, seeds, and budget.
- The result can be explained within the six-page narrative.

## 8. Code Boundaries and Data Flow

### 8.1 Configuration Layer

Owns parameter definitions, data windows, seeds, and experiment identifiers. It must not compute metrics or mutate simulation outputs.

### 8.2 Execution Layer

Builds effective SUMO inputs from the manifest and runs L1/L2 variants. It writes run-local configuration records and simulation outputs. It must not decide which result supports a paper claim.

### 8.3 Evaluation Layer

Reads immutable outputs and computes common metrics. It validates sample sizes and produces one long-form metrics table. It must not rerun calibration or alter thresholds.

### 8.4 Reporting Layer

Generates the final Table I and figures from validated CSV files. Figure and table scripts must include input paths in metadata or a sidecar manifest.

### 8.5 Manuscript Layer

Consumes only reporting-layer artifacts. Numbers are inserted from the final generated table and checked against the source CSV before submission.

## 9. Six-Page Manuscript Design

The final paper keeps five figures and one table:

- **Fig. 1:** revised RCMDT architecture with `theta_bus`, `x_corr`, the audit path, and freeze protocol.
- **Fig. 2:** merged contamination evidence with raw/clean distribution, Rule C geometry, and trajectory accumulation.
- **Fig. 3:** Rule C sensitivity plus fixed/statistical/adaptive comparison.
- **Fig. 4:** final distributional validation CDF generated from the common protocol.
- **Fig. 5:** equal-budget BO versus LHS cumulative-best curves.
- **Table I:** A0-A4 full ablation with mean, standard deviation, worst-window, cross-day metric, and sample count.

Text allocation:

- Abstract: approximately 180 words.
- Introduction: reduce by approximately 25 percent.
- Related Work: approximately half a page; add a fifth stream on simulation calibration and Bayesian optimization.
- Problem Formulation and Methodology: approximately 1.5 pages.
- Experimental Setup: approximately 0.4 page.
- Results: approximately 1.5 pages.
- Conclusion: approximately 150 words.
- References: approximately 15-17 entries.

The revision removes unsupported expressions such as `significantly outperforms`, `confidently shifted`, and `confirms effective generalization`. It reports residual discrepancy when the K-S test still rejects equality.

## 10. DOCX Editing Workflow

1. Preserve a backup of the authoritative manuscript.
2. Produce a tracked-change review DOCX using minimal, precise OOXML edits.
3. Replace figures and Table I while preserving IEEE styles and numbering.
4. Verify the tracked version through text extraction and PDF rendering.
5. Produce an accepted-changes camera-ready DOCX and a six-page PDF.
6. After final verification, update the user-designated authoritative DOCX while retaining the backup and tracked review version.

The final visual check covers column balancing above Fig. 1, equation alignment, caption legibility, figure resolution, reference overflow, author affiliations, and removal of template placeholders.

## 11. Error Handling and Quality Gates

An experiment result is rejected when:

- A required input or simulation output is missing.
- A metric is null or based on fewer samples than the declared minimum.
- Expected configuration differences produce the same effective hash.
- Training information leaks into cross-day threshold fitting.
- The run uses a seed or parameter setting outside the manifest.
- The reported table cannot be regenerated from the recorded output.

The manuscript is rejected for submission when:

- It exceeds six pages.
- A reported value cannot be found in the final table.
- Abstract, Results, and Conclusion use inconsistent values.
- Any `next-day` wording remains.
- Symbol aliases or inequality definitions conflict.
- The rendered PDF contains overflow, unreadable labels, or unbalanced columns around the full-width figure.

## 12. Verification

- Run the existing P14 smoke workflow with fixtures.
- Add focused checks for configuration hashing, non-null metric schemas, strict Rule C behavior, and transfer-field completeness.
- Verify five valid seeds for the main ablation and at least three for BO-LHS.
- Regenerate all paper artifacts from final outputs.
- Extract the final DOCX text and search for obsolete symbols, wording, numbers, and placeholders.
- Render the final DOCX to PDF and inspect all six pages visually.

## 13. Schedule

- **July 16-17:** consolidate protocol, terminology, manifests, and validation checks.
- **July 18-19:** run threshold comparison and full ablation.
- **July 20:** run equal-budget BO-LHS comparison.
- **July 21:** compute cross-day metrics and freeze final CSV outputs.
- **July 22:** generate five figures and Table I.
- **July 23-24:** revise the Word manuscript with tracked changes.
- **July 25:** complete six-page layout and numerical audit.
- **July 26:** upload only; no planned new experiment.

## 14. Degradation Policy

When time or validity constraints require scope reduction:

1. Remove DAPPER/ES-MDA from the manuscript.
2. Reduce the adaptive threshold comparison from Isolation Forest plus IQR/MAD to IQR/MAD only.
3. Reduce BO-LHS to three seeds while preserving 40 evaluations per method.
4. Retain the full ablation and cross-day transfer; these are mandatory reviewer-response evidence.
5. Remove any failed experiment rather than reframing it as a favorable result.

## 15. Deliverables

- Corrected and validated experiment source code.
- Frozen experiment manifests and final long-form metrics CSV.
- Regenerated paper figures and Table I source.
- Tracked-change manuscript for review.
- Accepted-changes six-page camera-ready DOCX.
- Six-page verification PDF.
- A short change log mapping reviewer comments to manuscript changes.

