# Phase 5: Hyperparameter Tuning - Research

## Context & Objectives
We need to add hyperparameter tuning via Optuna to `train_action_predictor_embedding.py`. 
Key requirements:
- Use Optuna with `sqlite:///tuning_history.db` for persistence.
- Add `--tune` and `--n_trials` flags.
- Optimize learning rate, dropout rates, and dense layer capacities.
- Crucially, maintain the existing clean execution if `--tune` is NOT provided.

## Codebase Analysis

### 1. `train_action_predictor_embedding.py` Architecture
The script is divided into distinct functions:
- `build_embedding_model(...)`: Defines the Keras Functional API architecture. It currently hardcodes the `Dense` widths (512, 256, 128) and `Dropout` rates (0.25, 0.25, 0.2) in the deep classification head. It also hardcodes the optimizer's learning rate initialization (though it can be overridden later).
- `prepare_inputs(...)`: Creates the input dictionaries for the Keras model.
- `train_embedding_model(...)`: The main pipeline orchestration. It handles data loading, preprocessing, group splitting, encoding, calls `build_embedding_model`, and runs `model.fit`.

### 2. Integration Point for Optuna
To avoid reloading the dataset (which takes a long time and uses a lot of memory) for every tuning trial:
- The data loading, filtering, `build_medium_X`, vocabulary construction, and inputs preparation (`prepare_inputs`) must happen **only once**.
- The tuning should be injected *inside* `train_embedding_model` right before it would normally call `build_embedding_model`.

If the `--tune` flag is present, `train_embedding_model` should:
1. Define a nested `objective(trial)` function.
   - Propose dimensions: e.g., `dense1 = trial.suggest_categorical('dense1', [256, 512, 1024])`
   - Propose learning rate: e.g., `lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)`
   - Propose dropouts: e.g., `drop1 = trial.suggest_float('drop1', 0.1, 0.5)`
   - Call `build_embedding_model(...)` passing in the proposed values.
   - Define callbacks, explicitly including `optuna.integration.TFKerasPruningCallback(trial, "val_loss")`.
   - Run `model.fit(...)` just like normal but with `verbose=0` or `verbose=2` to reduce spam.
   - Return the metric to minimize or maximize (e.g., `min(history.history['val_loss'])`).
2. Create and run the study:
   - `study = optuna.create_study(direction="minimize", storage="sqlite:///tuning_history.db", study_name="tf_action_predictor", load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))`
   - `study.optimize(objective, n_trials=n_trials)`

If `--tune` is absent, `train_embedding_model` acts normally (retaining current behavior).

### 3. Modifications Needed
1. **CLI Parser**: Add `--tune` (action='store_true') and `--n_trials` (type=int, default=50).
2. **`build_embedding_model` Signature**: Expose hyperparameter args: `dense1_units, dense2_units, dense3_units, drop1_rate, drop2_rate, drop3_rate, learning_rate`.
3. **`train_embedding_model` Engine**: 
   - Accept `tune=False` and `n_trials=50`.
   - Add the `if tune:` branch to run the Optuna study.
   - Make sure we `import optuna` (can be at the top level since it's confirmed installed).

## Validation Architecture
- **Correctness Check**: Ensure running the script without `--tune` works identically to the current implementation.
- **Tuning Execution Check**: Ensure running with `--tune` correctly creates the DB, executes trials, and outputs the Best Trial.

## Conclusion
The architecture is well-suited for tuning. By creating an `objective()` closure inside `train_embedding_model(..., tune=True)`, we capture the heavily preprocessed training dictionaries for free without massive refactoring, making it zero-friction to implement.
