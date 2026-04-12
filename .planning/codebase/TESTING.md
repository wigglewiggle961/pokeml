# Testing & Verification

## Overview

PokéML does not have a formal automated test suite. Validation is done through:
1. **Train/eval split metrics** during model training
2. **Manual parity scripts** (`verify_parity.py`)
3. **Live battle testing** against Pokémon Showdown bots
4. **Log-based debugging** during inference

---

## 1. Model Training Metrics (Primary Validation)

Each training script computes metrics on a held-out test split (default: 20%):

### Action Predictor (`train_action_predictor.py`)
| Metric | Tool |
|--------|------|
| Top-1 Accuracy | `sklearn.accuracy_score` |
| Top-5 Accuracy | `sklearn.top_k_accuracy_score` |
| Multi-logloss | LightGBM `multi_logloss` callback |
| Feature Importance (Top 20) | `lgbm_model.feature_importance(importance_type='gain')` |

**Target performance thresholds** (from `fix-overfitting.md`):
- Train accuracy: ~45% (realistic ceiling given class imbalance)
- Eval accuracy: ~40% (expected generalization target)
- If train >> eval by >20%: overfitting alarm

### Switch Predictor (`train_switch_predictor.py`)
| Metric | Tool |
|--------|------|
| AUC | `sklearn.roc_auc_score` |
| Accuracy | `sklearn.accuracy_score` |
| Classification report | `sklearn.classification_report` |
| Binary logloss | LightGBM `binary_logloss` callback |

### Switch Target Predictor (`train_pokemon_switch_predictor.py`)
| Metric | Tool |
|--------|------|
| Top-1 Accuracy | `sklearn.accuracy_score` |
| Multi-logloss | LightGBM `multi_logloss` callback |
| Feature Importances | `lgbm_model.feature_importance()` → saved to joblib |

---

## 2. Parity Verification (`verify_parity.py`)

A standalone diagnostic script to verify move sanitization parity between training and inference.

### What It Tests
1. Loads `gen9ou-0.json` (Smogon data)
2. Loads `30k.parquet` (training data)
3. Extracts all move names from both sources
4. Applies **both** sanitization routines:
   - **Old**: `replace(' ', '').replace('-', '').replace('_', '').replace(':', '').replace('%', 'perc')`
   - **New**: `re.sub(r'[^a-z0-9]', '', name.lower())`
5. Reports overlap percentage

### Interpretation
| Parity % | Status |
|----------|--------|
| ≥ 90% | ✅ Acceptable |
| 70–89% | ⚠️ Degraded — investigate missing moves |
| < 70% | ❌ Critical — Smogon filtering is broken |

### Running It
```bash
python verify_parity.py
```
Expected output:
```
OLD PARITY: XX.XX% (N/M)
NEW PARITY: YY.YY% (N/M)
Missing N moves. Top 20: [...]
```

---

## 3. Overfitting Detection

The `fix-overfitting.md` file defines the current diagnostic checklist:

### Task 1: Debug Smogon Move Matching
- Print `y_raw` vs `smogon_moves` format comparison
- Goal: Identify which specific moves are missing from intersection

### Task 2: Verify Sanitization Implementation
- Implement `sanitize_name()` regex approach in both `train_action_predictor.py` and `predict_action.py`
- Target: >90% parity in `verify_parity.py`

### Task 3: Bench Blindness
- Add `--blind_opp_bench` flag to strip `p2_slotX_species` from training
- Goal: Top-20 feature importance should not contain bench species
- Verification: `grep` check that argument exists and alters column selection

### Task 4: Dry Test
- Run 50-round test with both blinds active
- Expected: Train/Eval logloss gap ≤ 0.2

---

## 4. Manual Live Battle Testing

### Running the Bot
```bash
python predict_action.py
```

The bot connects to Showdown and logs:
- Turn-by-turn prediction calls
- Feature preparation steps
- Top-K predicted actions with probabilities
- Species lookup results in Smogon data
- Switch probability vs threshold decision

### Key Log Indicators
| Log Message | Meaning |
|-------------|---------|
| `"Mapping Battle state for turn N..."` | Inference cycle started |
| `"DEBUG: Predicting/Filtering for active Pokemon: 'X'"` | Active species identified |
| `"Note: Using base form 'Y' moves for 'X'"` | Forme fallback triggered |
| `"Warning: Active species '...' not found in Smogon data"` | Species missing from usage stats |
| `"FATAL ERROR loading artifacts"` | Model files missing/corrupt |

---

## 5. Checkpoint Recovery

LightGBM training auto-saves checkpoints every 250 rounds via `lgbm_checkpoint_callback()`:
```python
checkpoint_path = f'action_lgbm_checkpoint_v4_{label_suffix}.txt'
```

If training crashes, reload from checkpoint:
```python
model = lgb.Booster(model_file='action_lgbm_checkpoint_v4_medium_move_only.txt')
```

---

## 6. Known Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| No unit tests for `process_replays.py` | Parser bugs silently corrupt training data | Manual spot-checks on sample replays |
| No integration test for feature parity | Training/inference schema drift | `verify_parity.py` + manual review |
| No CI/CD pipeline | Regressions go undetected | Manual pre-training checklist |
| No performance benchmarks for bot speed | Timeout risk on Showdown server | Log timestamps during `choose_move()` |
