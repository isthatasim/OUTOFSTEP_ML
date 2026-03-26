# Phase 1 Implementation Plan: Static OOS Q1 Validation Framework

## Goal
Upgrade the existing static OOS framework into a publication-grade validation suite **without breaking legacy behavior**.

## Change Partitioning (Required)

## Safe additions
- Add new scenario runners and reporting wrappers that call existing model/data utilities.
- Add dedicated scenario configs.
- Add scenario-focused tests and docs.
- Add additional output tables/figures without changing existing filenames.

## Moderate-risk refactors
- Unify repeated split logic into shared helpers while preserving current outputs.
- Introduce monotonic-consistency diagnostics into benchmark outputs.
- Normalize scenario naming and artifact registry fields across legacy + benchmark scripts.

## High-risk refactors
- Replacing legacy `main.py` orchestration with modular pipeline.
- Merging/removing legacy `src/*.py` in favor of `src/outofstep_ml/*`.
- Changing existing table schemas consumed by downstream docs/scripts.

---

## Proposed file-by-file plan (with required explanation)

| Proposed file | Why needed | Wrap existing code or new logic | Legacy break risk | Minimal test |
|---|---|---|---|---|
| `src/outofstep_ml/evaluation/scenario_static_validation.py` | Centralize the 10 requested static scenarios into one reproducible API. | Mostly wraps existing loaders/models/metrics; adds new scenario orchestration logic. | Low if called by new script only. | `test_scenario_runner_smoke`: run one tiny scenario and assert output table columns. |
| `src/outofstep_ml/evaluation/scenario_migrations.py` | Implement controlled low-inertia / weak-grid / stress escalation levels. | New logic for deterministic level generation; uses existing predictor. | Low. | `test_migration_levels_monotonic`: verify generated levels follow expected direction (H down, I down, S up). |
| `src/outofstep_ml/evaluation/scenario_heatmaps.py` | Standardize cross-interaction heatmaps over `(H,I,S)` with fixed/stratified covariates. | Wraps existing plotting/grid logic; adds scenario packaging and export schema. | Low-medium (plot dependencies). | `test_heatmap_grid_shape`: assert expected grid dimensions and saved files exist. |
| `src/outofstep_ml/evaluation/monotonic_checks.py` | Add explicit monotonic consistency diagnostics for H/I/S priors. | New logic (finite-difference / local perturbation checks) over existing model inference. | Medium (can expose model inconsistencies). | `test_monotonic_check_schema`: verify metrics fields + bounded violation rates. |
| `src/outofstep_ml/evaluation/imbalance_ablation.py` | Formal no-weight vs class-weight vs current imbalance strategy comparison. | Wraps existing models with controlled option toggles; light new logic. | Low-medium (model param toggling). | `test_imbalance_modes_present`: outputs rows for all 3 modes. |
| `src/outofstep_ml/evaluation/threshold_policy_compare.py` | Explicit scenario outputs for `tau_default`, `tau_F1`, `tau_cost`, `tau_HR`. | Wraps existing threshold utilities and metrics. | Low. | `test_threshold_policy_outputs`: each tau exists and decision metrics computed. |
| `src/outofstep_ml/evaluation/counterfactual_eval.py` | Standardized scenario 10 exports: feasibility, delta magnitudes, recommendation text. | Wraps existing counterfactual utilities; adds summary KPIs. | Low. | `test_counterfactual_fields`: expected columns + non-empty recommendation for risky samples. |
| `scripts/run_static_q1_validation.py` | Single reproducible command to generate all scenario artifacts from current trained/static benchmark setup. | Wrapper script using above modules and existing configs. | Low. | `test_run_static_q1_validation_smoke`: end-to-end smoke with sample data. |
| `scripts/generate_static_q1_tables.py` | Produce publication tables for 10 scenarios and protocol-aware comparisons. | Wraps existing table logic; adds scenario table templates. | Low. | `test_q1_tables_exist`: verify required csv/md files created. |
| `scripts/generate_static_q1_figures.py` | Generate publication figures for migration curves, heatmaps, monotonic checks, threshold trade-offs. | Wrap existing plotting + new scenario figure assembly. | Low-medium (rendering path issues). | `test_q1_figures_exist`: assert key figure paths exist. |
| `configs/static_q1_validation.yaml` | Canonical config for strict protocol + all 10 scenarios. | New config only. | None. | Config load smoke in CLI test. |
| `configs/static_q1_validation_long.yaml` | Longer budget reproducible comparison config (for final paper tables). | New config only. | None. | Parsed by runner in smoke mode. |
| `tests/test_static_q1_scenarios.py` | Guard scenario implementations and output schemas. | New tests. | None to runtime. | Run pytest target. |
| `tests/test_monotonic_checks.py` | Ensure monotonic diagnostics stable and reproducible. | New tests. | None. | Run pytest target. |
| `tests/test_threshold_policies.py` | Ensure threshold policy calculations are consistent and complete. | New tests. | None. | Run pytest target. |
| `docs/CASE_STUDY_VALIDATION_DESIGN.md` | Publication-facing scenario protocol and interpretation mapping. | New doc. | None. | Manual review. |
| `README.md` (update) | Add static Q1 scenario commands and artifact map. | Existing file update (docs only). | Low. | README command/path sanity check. |

---

## Proposed modifications to existing files (Phase 2)

| Existing file | Change type | Why needed | Risk |
|---|---|---|---|
| `src/outofstep_ml/benchmark/runner.py` | Moderate extension | Add optional scenario hooks + monotonic check export while preserving strict protocol. | Medium |
| `src/outofstep_ml/benchmark/tables.py` | Moderate extension | Include scenario-specific tables and monotonic consistency tables. | Medium |
| `scripts/run_full_benchmark.py` | Safe extension | Optional flag to append static-Q1 scenarios after benchmark run. | Low |
| `scripts/generate_figures.py` | Safe extension | Add `--mode static_q1`. | Low-medium |

---

## Implementation sequence recommendation (Phase 2)
1. Safe additions first (new modules/scripts/configs/tests).
2. Integrate scenario outputs into table/figure generation.
3. Add monotonic diagnostics.
4. Only then touch moderate refactors in benchmark runner.
5. Defer high-risk refactors to a separate PR after validation lock.

