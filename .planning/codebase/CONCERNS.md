# Concerns & Technical Debt

## 1. 🔴 CRITICAL: Training/Inference Feature Parity Gap

**Status**: Active — tracked in `fix-overfitting.md`

### Problem
The sanitization function used during training (`sanitize_name()` in `train_action_predictor.py`) differs from the one used during inference (`load_smogon_moves()` in `predict_action.py`):

```python
# Training (canonical)
re.sub(r'[^a-z0-9]', '', name.lower())

# Inference (legacy)
move_key.lower().replace(' ', '').replace('-', '').replace('_', '').replace(':', '').replace('%', 'perc')
```

### Impact
- Smogon valid move sets at inference time don't match training labels
- Moves with `%` (e.g., `Thunder` → training: `thunder`, inference: `thunderperc` if it existed) could be mismatched
- `verify_parity.py` was written specifically to measure this gap
- Results below 90% cause predictions to fall back to unfiltered results

### Mitigation
Port `sanitize_name()` regex approach to `predict_action.py`'s `load_smogon_moves()`.

---

## 2. 🔴 CRITICAL: Model Overfitting

**Status**: Active — tracked in `fix-overfitting.md`

### Problem
Action predictor shows significant train/eval accuracy gap, indicating memorization of training replays rather than learning generalizable strategy:
- Train accuracy: significantly higher than eval
- Top-20 feature importance contains opponent bench species (`p2_slot2_species`, etc.), which are unknown at match start

### Root Causes
1. **Bench species leakage**: The model learns which teams a specific species belongs to, effectively memorizing opponent team compositions
2. **Low regularization in earlier versions**: Current v4 applies heavy regularization (`reg_alpha=15.0`, `reg_lambda=15.0`, `colsample_bytree=0.15`, `min_child_samples=2000`)

### Mitigations Implemented (v4)
- Heavy L1/L2 regularization
- `min_child_samples=2000` (requires massive evidence per split)
- Column subsampling at 15% per tree
- `--blind_opp_bench` flag added to optionally strip `p2_slotX_species` features

### Remaining Work
- Run actual training runs with `--blind_opp_bench` enabled and verify bench species exit top-20 importances

---

## 3. 🟡 HIGH: Hardcoded Credentials in `predict_action.py`

**Status**: Active — not yet resolved

### Problem
```python
SHOWDOWN_USERNAME = os.environ.get("SHOWDOWN_USER", "www31")
SHOWDOWN_PASSWORD = os.environ.get("SHOWDOWN_PASS", "vimvimvim333")
```
Credentials are hardcoded as fallback defaults. If env vars are not set, bot runs with literal plain-text credentials visible in source.

### Risk
- Credential exposure if repo is ever made public
- Cannot rotate credentials without code changes

### Mitigation
Remove hardcoded defaults; enforce env var requirement with a startup check.

---

## 4. 🟡 HIGH: `map_battle_to_dataframe_row()` Normalization Inconsistency

**Status**: Suspected — requires validation

### Problem  
In `predict_action.py` line 727:
```python
species_name = pkmn.species.title().lower().replace(" ","")
```
The `.title()` call uppercases the first letter of each word before `.lower()` reduces it back. This is redundant and potentially differs from how `process_replays.py` normalizes species at parse time (which uses `normalize_species_name()` without `.title()`).

For opponent team (line 785):
```python
species_name = pkmn.species.title()  # Note: NO .lower()!
```
This stores a Title-cased species name into `p2_slot{i}_species`, while training data has lowercase. This is almost certainly a bug.

### Impact
- Categorical feature mismatch for `p2_slotX_species` at inference
- Model may route these to the `Unknown` category silently

---

## 5. 🟡 HIGH: No Formal Test Suite

**Status**: Ongoing gap

### Problem
The codebase has no automated tests. All validation is either manual (live battles) or metric-based (train/eval split). Key failure modes that are untested:
- Parser regression for edge-case log formats
- Feature schema drift between training and inference
- Slot lookup failures (silent `None` returns from `find_slot_id()`)
- Argparse flags that alter feature sets

### Risk
Silent data corruption during training; hard-to-debug inference failures.

---

## 6. 🟡 HIGH: `returnxc()` Syntax Bug

**Status**: Present in code — likely dead path

### Location
`train_pokemon_switch_predictor.py` line 535:
```python
returnxc()  # Bug: should be `return`
```
This will raise a `NameError` at runtime if the exception-handling path is hit. Since it's in an except block triggered by target variable creation failures, it only fires on error—but when it does, it masks the original exception.

---

## 7. 🟠 MEDIUM: Memory Management During Training

**Status**: Partially mitigated

### Problem
Training large datasets (30k+ replays × features) risks OOM during:
- Active move extraction loops (row-by-row iteration)
- Smogon usage stat injection (wide DataFrame join)
- Multi-hot encoding of all revealed moves

### Current Mitigations
- `gc.collect()` called throughout pipeline
- `free_raw_data=True` for LightGBM datasets
- Smogon features downcasted to `float32` immediately
- `del` statements on intermediate DataFrames

### Remaining Concern
The row-by-row `iterrows()` loop in `train_action_predictor.py` for active move extraction is O(N) and slow for large datasets. For N=100k+ rows this can take minutes.

---

## 8. 🟠 MEDIUM: Switch Target Predictor Label Design

**Status**: Architectural concern

### Problem
The switch target predictor (`train_pokemon_switch_predictor.py`) predicts **bench slot index** (0–4) rather than species identity. This means:
- Target label is positional (which bench slot), not semantic (which Pokémon)
- At inference, `predict_action.py` must reverse-lookup the species in that slot
- The `LabelEncoder` for this model encodes slot integers, making the output confusing

### Risk
If the slot mapping changes between training and inference, predictions are silently wrong.

---

## 9. 🟢 LOW: `player_prefix` Undefined Bug in `predict_action.py`

**Status**: Likely latent bug

### Location
`predict_action.py` line 756:
```python
flat_state[f'{player_prefix}_active_usage_{move_name}'] = usage_val
```
`player_prefix` is not defined in the surrounding scope for p1. The variable `prefix` is used in the outer loop. This may cause a `NameError` or silently use a stale variable depending on loop order.

---

## 10. 🟢 LOW: Aggressive Warning Suppression

```python
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
```
These are suppressed globally, which may hide legitimate pandas 3.x migration warnings that could become breaking changes.
