# Phase 5: Hyperparameter Tuning - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Automate the search for optimal hyperparameters (learning rate, dropout, dense widths) for the TensorFlow Embedding Model to further improve validation/top-5 performance. This modifies the existing training script to run search trials using Optuna.

</domain>

<decisions>
## Implementation Decisions

### Search Strategy & Infrastructure
- **D-01:** Add a `--tune` flag to the existing `train_action_predictor_embedding.py` script rather than creating a separate standalone tuning script.

### Dataset Scaling
- **D-02:** Use the `100k.parquet` subset exclusively for all tuning trials to keep iteration loops fast while remaining representative.

### Hyperparameter Search Space
- **D-03:** Focus the search on high-impact parameters:
  - Learning Rate (log-scale, ~1e-4 to 1e-2)
  - Dropout Rates (across deep head layers, e.g., 0.1 to 0.4)
  - Dense Head layer widths/capacities (e.g., 128, 256, 512, etc.)

### Trial Pruning & Persistence
- **D-04:** Use Optuna's SQLite storage (e.g., `storage="sqlite:///tuning_history.db"`) to easily persist, pause, and resume tuning studies.
- **D-05:** Integrate Optuna's pruning (e.g., `TFKerasPruningCallback` or equivalent for `model.fit()`) to aggressively kill unpromising trials early and save compute.

### the agent's Discretion
- The exact point of integration for the `objective(trial)` function.
- Choice of standard sampling algorithm (e.g., TPESampler).
- The metric to optimize (e.g., `val_accuracy` or `val_loss`).

</decisions>

<specifics>
## Specific Ideas

- Setup should be as zero-friction as possible while keeping the script intact for standard non-tuning runs.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Tuning Targets
- `.planning/ROADMAP.md` § Phase 5 — Outlines the acceptance criteria: Validation Accuracy > 40%, Top-5 Accuracy > 75%.
- `.planning/experiments.md` — Contains the historical record of runs and generalization gaps leading up to this point.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `train_action_predictor_embedding.py` already has a clean `build_embedding_model` and `train_embedding_model` flow, which can be wrapped to accept trial parameters.

### Established Patterns
- Uses `argparse` for CLI flags (where `--tune` and perhaps `--n_trials` will live).
- Model leverages explicit layer components (e.g., `layers.Dropout(0.25)`) that can be easily replaced with `trial.suggest_float(...)`.

### Integration Points
- `if __name__ == '__main__':` block for CLI routing.

</code_context>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-hyperparameter-tuning*
*Context gathered: 2026-04-14*
