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
- Rule C is compared with at least one statistical baseline and one training-fitted adaptive baseline without claiming unsupported global optimality.
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

The executable contract is:

- Filter both sources by the manifest's route and direction.
- Aggregate observed link travel time by `(route, bound, from_seq, to_seq)` and construct cumulative observed time from sequence 1.
- For each simulated vehicle, subtract its arrival time at its first matched stop, then average relative arrival time by downstream `to_seq`.
- Inner-join observed and simulated cumulative times on `(route, bound, to_seq)` and exclude the origin stop.
- Define `i` as one joined downstream stop and `n` as the number of joined downstream stops.
- Reject the evaluation when `n < 3`.

One L1 candidate evaluation uses one deterministic SUMO seed derived from `(optimization_seed, evaluation_index)`. BO and continued LHS use the same seed schedule at matching evaluation indices. The paper reports optimization performance across optimization seeds; it does not pool stop errors from different candidates or seeds into one Equation (2) value.

The scalar candidate score is fixed as follows. Compute the 68X composite loss

`JL1_68X = RMSE(e) + 1.0 * (MAE(e) + 0.5 * std(abs(e))) + 0.3 * Q0.9(abs(e))`.

Compute `RMSE_960` with the same cumulative-time matching contract. A candidate is feasible when both routes have at least three matched downstream stops and `RMSE_960 <= 350 s`. The score supplied to BO or used to rank LHS is `JL1_68X` for a feasible candidate and `2000 + 10 * (RMSE_960 - 350)` for a valid constraint-violating candidate. Missing or malformed simulation output is an infrastructure failure and triggers the retry policy; it is not inserted into the surrogate as a numerical observation. A candidate that remains unevaluable after retries fails that optimization seed.

The 40-evaluation budget counts successful candidate evaluations. Deterministic infrastructure retries do not consume additional budget. The final paper reports `JL1_68X`, its four components, `RMSE_960`, and feasibility for the selected candidate. The earlier RMSE-only B2 score cannot be relabeled as this composite objective.

### 6.4 Observation Audit

Use one Rule C definition everywhere:

- `T > 325 s`
- `v_eff < 5 km/h`
- `distance <= 1500 m`

The thresholds are predeclared protocol constants recorded in the repository before the camera-ready comparison. They express a long-duration, near-stationary, short-link condition and are not selected by minimizing K-S on either split. Development-split sensitivity analysis tests their neighborhood but cannot change the selected values. IRN remains an external plausibility check and does not tune the rule.

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

The data partitions are fixed as follows:

- **Development split:** December 19, 2025, 17:00-18:00 HKT. Rule C sensitivity is described on this split; IQR/MAD and adaptive baselines are fitted here.
- **Cross-day test split:** December 30, 2025, 15:00-16:00 HKT. All audit methods and calibrated configurations are frozen before this split is evaluated.
- There is no separate held-out same-day test in the available repository data. The paper must not describe one.

Metric definitions are fixed:

- `KS-speed` is the two-sample K-S statistic `D`, not its p-value, between real and simulated event-level effective-speed samples on the same frozen link-key set.
- `KS-TT` is defined identically for event-level travel-time samples and is secondary unless Table I has room.
- All A0-A4 rows use the same Rule-C-clean real evaluation population. Audit effects on raw versus clean populations are reported separately in Fig. 2. A3 uses raw D2D observations inside L2 but is evaluated against the same clean target as every other configuration.
- Per-seed metrics are computed first; Table I reports their arithmetic mean and sample standard deviation. Real observations are not duplicated or pooled across seeds.
- `worst-15-min KS` is the maximum valid K-S statistic over 15-minute half-open windows `[start, start+900 s)` advanced every 60 seconds within the one-hour window.
- A full-window K-S value requires at least 20 real and 20 simulated events. A subwindow requires at least five events from each source. Missing minima produce a failed metric, not a zero.
- `retention rate = n_clean_link_keys / n_eligible_raw_link_keys`, where eligibility includes the declared route, direction, time window, and positive distance/travel-time requirements.
- `IRN contradiction rate` is computed separately for each audit method on that method's flagged records: the numerator is the number of IRN-matched flagged D2D link-window records with median effective speed below 5 km/h and matched IRN median speed at least 5 km/h; the denominator is all IRN-matched records flagged by that method. Report numerator, denominator, and unmatched flagged count; do not treat it as a classification-accuracy measure.
- P-values may be reported as diagnostic values but cannot replace `D` or be used to label a model as passed/failed in the main comparison.

Two hashes serve different purposes:

- `provenance_hash` covers the complete canonical manifest, input content hashes, software versions, seed, run ID, and schema version.
- `simulation_effective_hash` covers only canonical sorted JSON for input content hashes, bus parameters, background parameters, observation semantic, simulator settings, and SUMO seed. Timestamps, output paths, and run IDs are excluded.

The validator also records component hashes for `bus_parameters`, `background_parameters`, `observation_semantic`, and `simulator_inputs`, plus explicit mechanism fields `l1_enabled` and `l2_enabled`. It verifies the A0-A4 mechanism matrix in Section 7.3 and the expected equalities listed there. It does not require final numerical parameter hashes to differ, because a calibrated value may legitimately coincide with a baseline value.

### 7.2 E1: Observation-Rule Comparison

Compare at minimum:

- Fixed Rule C.
- MAD-based statistical filtering.
- Isolation Forest as a data-adaptive baseline.

IQR may be included as a second statistical baseline when space and time permit. If Isolation Forest fails because of sample size or implementation instability, replace it with a training-fitted empirical quantile rule using predeclared 95th travel-time and 5th speed percentiles. At least one adaptive baseline must remain.

All E1 methods operate on one aggregated record per `(route, bound, from_seq, to_seq, one-hour window)` after common eligibility filtering: declared routes/directions, positive time and distance, and `distance <= 1500 m`. They are fitted jointly across eligible routes and directions on the development split.

- **MAD:** fit median and `1.4826 * MAD` for `log1p(tt_median)` and `log1p(speed_median)`. When a feature's MAD scale is zero, use `IQR / 1.349` from the same development feature; when that is also zero, assign robust score zero for that feature to every record. Flag when the robust travel-time score is greater than `3.5` and the robust speed score is less than `-3.5`.
- **Isolation Forest:** use features `log1p(tt_median)`, `log1p(speed_median)`, and `log1p(dist_m)`, standardized by development medians and MAD scales. Use `n_estimators=200`, `max_samples='auto'`, `contamination='auto'`, and `random_state=42`. Flag an anomaly only when it is also above the development median travel time and below the development median speed.
- **Quantile fallback:** fit development `Q95(tt_median)` and `Q05(speed_median)` with NumPy's linear interpolation convention (`method='linear'`). Flag records with `tt_median >= Q95` and `speed_median <= Q05`.

Each method converts record decisions to retained link keys by excluding flagged keys. Fitted statistics, model serialization hash, package version, and retained/flagged keys are saved in the audit manifest.

Fit MAD, Isolation Forest, or the quantile fallback on the development split only. Freeze fitted values and model state before evaluating the cross-day split. Rule C thresholds are predeclared physical defaults; the development sensitivity grid is a robustness analysis and does not retune Rule C after inspecting cross-day results. Report retention rate, full-window K-S, worst-15-minute K-S, and IRN contradiction rate. Since semantic ground-truth labels are unavailable, the result supports stability and interpretability, not classification accuracy or universal superiority.

E1 uses the A0 zero-shot simulation outputs as a fixed reference; no audit method triggers calibration or simulation reruns. Every method starts from the same eligible raw real link-key universe on each split. A method's retained real link keys are then applied to the same A0 simulation output before computing its K-S values. Differences in retained keys are therefore an explicit audit-method outcome and must be reported with retention rate; the eligible universe and simulation configuration remain fixed.

IRN consistency is matched at `(route, bound, from_seq, to_seq, one-hour window)` through the frozen link-to-IRN mapping. Unmatched records are excluded from the rate denominator and reported separately.

Also compute a compact Rule C sensitivity grid around the selected point, such as `T in {275, 325, 375}` and `v in {4, 5, 6}`.

### 7.3 E2: Full Protocol Ablation

Run these configurations with identical seeds and windows:

| ID | Configuration | Audit | L1 BO | L2 IES | Observation semantic |
|---|---|---:|---:|---:|---|
| A0 | Zero-shot | No calibration-time audit | No | No | No L2 input |
| A1 | BO-only | Fixed evaluation audit | Yes | No | No L2 input |
| A2 | IES-only | Fixed evaluation audit | No | Yes | Moving-only L2 |
| A3 | Raw-RCMDT | No L2 audit | Yes | Yes | Raw D2D supplied to L2 |
| A4 | Full-RCMDT | Yes | Yes | Yes | Moving-only L2 |

For each configuration, report mean and standard deviation across five seeds for full-window K-S and worst-window K-S on the common clean evaluation population. Cross-day K-S and sample counts are mandatory because the repository contains the required real cross-day inputs. Each configuration must therefore generate the corresponding cross-day simulation output.

Disabled-layer values are fixed:

- Baseline bus parameters are `t_board=2.0 s`, `t_fixed=5.0 s`, `tau=1.0 s`, `sigma=0.5`, `minGap_bus=2.5 m`, `accel=2.6 m/s^2`, and `decel=4.5 m/s^2`.
- Baseline background values are `capacityFactor=1.0`, `minGap_background=2.5 m`, and `impatience=0.5`.
- For a given optimization seed, A1, A3, and A4 use the same frozen L1-selected bus parameters.
- A2, A3, and A4 start L2 from the same background priors and common ensemble perturbation seed schedule.
- A0 uses both baseline sets; A1 freezes baseline background; A2 freezes baseline bus parameters.

The final table reports the fixed real event count once per split. Simulated event counts vary by seed and are reported as mean plus sample standard deviation in the table footnote; the long-form metrics retain each per-seed count.

The output validator fails when two configurations expected to differ share the relevant component hash, when required output files are absent, or when mandatory metrics are null.

The deterministic mechanism and equality checks are:

- A0: `l1_enabled=false`, `l2_enabled=false`; A1: `true,false`; A2: `false,true`; A3 and A4: `true,true`.
- A0 and A2 have equal baseline bus-parameter hashes.
- A1, A3, and A4 have equal frozen L1 bus-parameter hashes for a given seed.
- A0 and A1 have equal baseline background-parameter hashes.
- A2, A3, and A4 have equal L2 prior and ensemble-seed hashes.
- A3 has `observation_semantic=raw_d2d`; A2 and A4 have `observation_semantic=moving_only`.

Final background-parameter hashes may coincide and do not by themselves fail validation. A configuration fails when its declared mechanism fields, frozen-input equalities, or observation semantic violate this matrix.

### 7.4 E3: Equal-Budget BO Versus LHS

For each optimization seed:

- Start BO and LHS from the same 15-point LHS design.
- BO selects 25 additional candidates with expected improvement.
- Continued LHS evaluates 25 additional independently sampled candidates.
- Both methods therefore use 40 simulations.

Report cumulative best objective versus evaluation count, final best objective, and evaluations needed to reach a predeclared target. The normal design uses five optimization seeds. The target is fixed separately for each shared initial design before the additional 25 evaluations as `0.95 * best feasible objective among the 15 shared initial points`; a method that never reaches it is recorded as `not reached`. Retain all 40 evaluations per method; do not compare different budget sizes.

When a shared 15-point initial design contains no feasible candidate, that optimization seed is structurally invalid and stops before the additional evaluations. It is not assigned a post hoc target and is not replaced by an unplanned seed. The final BO-LHS comparison uses the common valid seed set under the five-target/three-minimum rule; fewer than three valid seeds block the BO-LHS evidence.

### 7.5 E4: Cross-Day Transfer

Use December 19, 2025 as the development date and December 30, 2025 as the cross-day transfer date. Rename result fields from `next_day_*` to `cross_day_*`. The paper must state the exact dates and avoid next-day wording. Missing cross-day simulation outputs after the retry policy block the full-ablation deliverable; they do not trigger silent omission.

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

### 8.6 Interface Schemas

The manifest schema, versioned as `paper-manifest/v1`, requires:

- `schema_version`, `experiment_id`, `config_id`, `method_id`, `seed`.
- `datasets[]` with path, SHA-256, observation date, timezone, and time window.
- `routes[]` with route, direction, and link-key selection.
- `l1` with parameter bounds, objective definition, initial design, budget, and seed schedule.
- `l2` with state components, priors, bounds, ensemble size, iterations, damping, and observation semantic.
- `audit` with method, fitted-on split, frozen parameters/model hash, and Rule C conditions when applicable.
- `simulator` with SUMO version, effective input hashes, seed, and timeout.
- `outputs` with run directory and required artifact names.

Each run writes `run-status/v1` containing `status` (`pending`, `running`, `succeeded`, `failed`), attempt number, start/end timestamps, exit code, error summary, manifest hashes, and produced artifact hashes.

The long-form metric schema, `paper-metrics/v1`, requires one row per metric and seed:

- `experiment_id`, `config_id`, `method_id`, `seed`, `split`.
- `metric_name`, `domain` (`speed` or `travel_time`), `value`, `unit`.
- `n_real`, `n_sim`, `n_link_keys`, `window_start`, `window_end`.
- `manifest_hash`, `simulation_output_hash`, `evaluator_version`, `status`.

Reporting scripts accept only validated `paper-metrics/v1` rows with `status=succeeded`. Every figure or table writes an `artifact-sidecar/v1` JSON containing source metric hashes, script version, output hash, and manuscript figure/table identifier.

## 9. Six-Page Manuscript Design

The target layout keeps five figures and one table:

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

Page allocation:

- **Page 1:** title, authors, abstract, Introduction through research gap.
- **Page 2:** contributions, compressed Related Work, Problem Formulation.
- **Page 3:** Methodology and Fig. 1.
- **Page 4:** Experimental Setup, Fig. 2, and the compact Fig. 3.
- **Page 5:** Table I, Fig. 4, Fig. 5, and principal results.
- **Page 6:** residual discussion, limitations, conclusion, acknowledgment, and references.

Page overflow is resolved in this order:

1. Remove repeated background and shorten captions without removing definitions.
2. Merge Fig. 3 into Fig. 2 as a sensitivity/comparison panel, reducing the count to four figures.
3. Remove the trajectory panel from Fig. 2 while retaining the raw/clean distribution and Rule C geometry.
4. Replace Fig. 5 with a compact numerical BO-LHS statement or inset when equal-budget evidence remains traceable.
5. Remove weak or redundant references, especially superseded preprints.

Font size, margins, IEEE template geometry, mandatory ablation, adaptive threshold evidence, and cross-day evidence cannot be reduced to solve overflow.

The revision removes unsupported expressions such as `significantly outperforms`, `confidently shifted`, and `confirms effective generalization`. It reports residual discrepancy when the K-S test still rejects equality.

## 10. DOCX Editing Workflow

Artifacts are stored beside the authoritative manuscript using these exact names:

- Backup: `Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.original-20260716.docx`
- Tracked review: `Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-tracked.docx`
- Accepted final: `Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-final.docx`
- Verification PDF: `Operator-Aware Robust Calibration for Bus-Corridor Digital Twins via Bayesian Optimization and Iterative Ensemble Smoothing.camera-ready-final.pdf`

Workflow:

1. Create the backup once and verify its SHA-256 against the initial authoritative file.
2. Unpack a working copy and create minimal tracked changes with OOXML `w:del`/`w:ins`, preserving unchanged runs and IEEE styles.
3. Replace figures and Table I while preserving captions, anchors, numbering, and section properties.
4. Pack the tracked review artifact and verify it through Pandoc extraction.
5. Accept revisions in a separate OOXML copy by retaining inserted content, removing deleted content, and clearing revision markup; pack the accepted final artifact.
6. Render with Microsoft Word in noninteractive mode when available; Word-rendered page count is authoritative. LibreOffice headless rendering is a preflight fallback and cannot alone approve the final six-page layout.
7. After the accepted final DOCX and Word-rendered PDF pass all checks, copy the accepted final content to the user-designated authoritative DOCX path. Never overwrite the backup or tracked review artifact.

The final visual check covers column balancing above Fig. 1, equation alignment, caption legibility, figure resolution, reference overflow, author affiliations, and removal of template placeholders.

## 11. Error Handling and Quality Gates

An experiment result is rejected when:

- A required input or simulation output is missing.
- A metric is null or based on fewer samples than the declared minimum.
- Expected configuration differences produce the same effective hash.
- Training information leaks into cross-day threshold fitting.
- The run uses a seed or parameter setting outside the manifest.
- The reported table cannot be regenerated from the recorded output.

Failure and retry policy:

- Each `(experiment_id, config_id, seed, split)` receives one initial attempt and at most two deterministic retries using the same manifest and seed. Replacement seeds are forbidden.
- A run timeout is the larger of 30 minutes or three times the median runtime of successful pilot runs; before pilot evidence exists, use 60 minutes.
- Failed partial outputs are moved to an attempt-specific quarantine directory and never reused by aggregation.
- After retries, the A0-A4 comparison uses the intersection of successful seeds across every configuration and both splits. Five seeds are the target; three common seeds are the minimum.
- Fewer than three common seeds, a missing mandatory configuration, or missing cross-day output blocks the full-ablation deliverable and requires a user decision. The manuscript must not substitute unequal seed sets or omit the failed row.
- BO-LHS uses five seeds normally and may fall back to three common seeds. Fewer than three blocks the BO-LHS evidence and activates the manuscript compression fallback that removes Fig. 5 and limits the claim to the already verified single-run observation.

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
- Verify five common valid seeds for the main ablation and BO-LHS under the normal release. A documented degraded release may use exactly three common seeds for either experiment under Section 11; four valid common seeds are reported as four rather than discarded. Fewer than three blocks the corresponding evidence.
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
2. Drop the optional IQR baseline. Retain MAD and either Isolation Forest or the predeclared empirical-quantile adaptive fallback.
3. Reduce BO-LHS from five to three common seeds while preserving 40 evaluations per method.
4. Retain the full ablation and cross-day transfer; these are mandatory reviewer-response evidence.
5. Remove any failed experiment rather than reframing it as a favorable result.

The degradation policy cannot authorize unequal populations, null transfer metrics, fewer than three common seeds, or omission of a mandatory A0-A4 configuration. Those conditions block the relevant claim and require explicit user direction.

## 15. Reviewer Traceability Matrix

The change log uses these fixed identifiers:

| ID | Reviewer requirement | Required evidence or manuscript change |
|---|---|---|
| AE-1 | Resolve reviewer concerns | All rows below completed or explicitly mapped to a verified change |
| R1-1 | Five streams versus four | Add the fifth BO/simulation-calibration stream or change the count; selected design adds the fifth stream |
| R1-2 | Define Equation (2) error | Apply Section 6.3 contract in code and manuscript |
| R1-3 | Align text above Fig. 1 | Word-rendered visual check and balanced columns |
| R1-4 | Explain GP and candidate | Define training pairs, predictive mean/variance, candidate vector, and `argmax EI` selection |
| R1-5 | Differentiate novelty | Reframe contribution around audit, scope separation, and freeze protocol; support with A0-A4 |
| R2-1 | Quantitative and methodological comparison | Common-protocol A0-A4 table; optional external smoother baseline only if valid |
| R2-2 | Threshold basis and alternatives | E1 sensitivity plus MAD and adaptive baseline |
| R2-3 | Introduction and literature distinction | Compressed recent literature comparison and explicit gap paragraph |
| R2-4 | Symbols and formula formatting | Terminology contract, automated obsolete-symbol search, and visual proofread |
| R3-1 | Clarify fundamental novelty | Same contribution reframing as R1-5 with restrained claims |
| R3-2 | Simplify dense writing | Reduce Introduction by about 25 percent and shorten Methodology prose |
| R3-3 | Improve figures and captions | Regenerated figures, self-contained captions, readable labels, and PDF inspection |

Completion requires every identifier to appear once in the final change log with manuscript location, artifact or test reference, and status.

## 16. Deliverables

- Corrected and validated experiment source code.
- Frozen experiment manifests and final long-form metrics CSV.
- Regenerated paper figures and Table I source.
- Tracked-change manuscript for review.
- Accepted-changes six-page camera-ready DOCX.
- Six-page verification PDF.
- A short change log mapping reviewer comments to manuscript changes.
