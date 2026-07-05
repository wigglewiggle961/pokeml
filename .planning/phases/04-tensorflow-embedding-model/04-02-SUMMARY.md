# Plan 04-02 Summary: train_action_predictor_embedding.py

## What Was Built
- `train_action_predictor_embedding.py` (656 lines) — TF-only embedding model training script

## Architecture Implemented
- **Keras Functional API** with dict-based named `Input` layers
- **Entity embeddings** for: species (dim=48), moves (dim=32), status/hp/type/field (dim=8)
- **Sum-pool projection** for revealed moves via `Dense(32, use_bias=False)`
- **Numerical branch** via `Dense(64, swish)` projection
- **Deep head**: 512→256→128 with `swish` activations, BatchNorm, Dropout(0.25/0.25/0.2)
- **Output**: `Dense(num_classes, softmax)` with `CategoricalCrossentropy(label_smoothing=0.05)`

## Key Files Created
- `train_action_predictor_embedding.py`

## Key Design Decisions
- Imports all helpers from `feature_engineering.py` (no duplication)
- `train_action_predictor.py` untouched (confirmed via git)
- Artifact naming: `action_tf_embedding_v1_*` (no clash with v4 artifacts)
- GroupShuffleSplit by `replay_id` (no data leakage)
- Shared vocabulary encoders per feature group (species, move, status, hp, type)
- `ReduceLROnPlateau` + `EarlyStopping(patience=10)` callbacks
- Saves: `.keras` model, `.json` metadata, `.joblib` artifacts bundle

## Self-Check: PASSED
