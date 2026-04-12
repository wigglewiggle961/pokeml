# Tech Stack

## Runtime Environment
- **Language**: Python 3.x
- **Package Manager**: pip / venv (`.venv` directory present)
- **OS**: Windows (development environment)

## Core ML Frameworks

### Primary: LightGBM
- **Version**: Not pinned (uses `lightgbm` latest compatible)
- **Usage**: All production models — move prediction, binary switch, switch target
- **Training mode**: `gbdt` (Gradient Boosted Decision Trees)
- **Key params (action predictor)**: `num_leaves=15`, `max_depth=5`, `min_child_samples=2000`, `reg_alpha=15.0`, `reg_lambda=15.0`, `colsample_bytree=0.15`
- **GPU support**: Optional via `device: 'gpu'` flag with `max_bin=63` workaround
- **Multi-class objective**: `multiclass` / `multi_logloss`
- **Binary objective**: `binary` / `auc`
- **Serialization**: `.txt` model files via `lgb.Booster.save_model()`
- **Checkpointing**: Custom callback saves every 250 rounds to `*_checkpoint_*.txt`
- **Early stopping**: 50 rounds for move predictor, 100 for switch predictor

### Secondary: TensorFlow / Keras
- **Usage**: Alternative training path (not deployed in production bot)
- **Architecture**: Sequential: `Input → Dense(256) → BN → ReLU → Dropout(0.3) → Dense(128) → BN → ReLU → Dropout(0.3) → Dense(64) → BN → ReLU → Dropout(0.3) → Dense(N_classes, softmax)`
- **Loss (multi-class)**: `CategoricalCrossentropy(label_smoothing=0.1)`
- **Loss (binary)**: `binary_crossentropy`
- **Optimizer**: `Adam`, configurable learning rate
- **Serialization**: `.keras` files

## Hyperparameter Optimization: Optuna
- **Used in**: `train_switch_predictor.py`, `train_pokemon_switch_predictor.py`
- **Binary switch strategy**: `direction='maximize'` optimizing AUC
- **Switch target strategy**: `direction='minimize'` optimizing multi-logloss
- **Default trials**: 50 per study
- **Search space**: `learning_rate`, `n_estimators`, `num_leaves`, `reg_alpha`, `reg_lambda`, `colsample_bytree`, `subsample`, `min_child_samples`
- **Early stopping within trials**: 50 rounds

## Data Stack
| Library | Purpose |
|---------|---------|
| `pandas` | DataFrame manipulation, Parquet I/O |
| `numpy` | Numerical operations, array handling |
| `scikit-learn` | `StandardScaler`, `LabelEncoder`, `train_test_split`, `compute_class_weight`, metrics |
| `joblib` | Serialization of scalers, encoders, feature info dicts |
| `pyarrow` / `fastparquet` | Parquet backend (via pandas) |

## Data Formats
| Format | Description |
|--------|-------------|
| `.log` | Raw Pokémon Showdown replay logs |
| `.parquet` | Processed training datasets (30k–112k+ rows) |
| `.csv` | Legacy training data format (10k.csv ~385MB) |
| `.txt` | LightGBM serialized model files |
| `.keras` | TensorFlow serialized model files |
| `.joblib` | Python object serialization (scalers, encoders, feature info) |
| `.json` | Smogon usage statistics (`gen9ou-0.json`, ~14MB) |

## Utilities
- `gc` — Explicit garbage collection throughout training loops to manage memory
- `re` — Name sanitization, move string cleaning (via `sanitize_name()`)
- `copy` — Deep copy of game state during replay parsing
- `warnings` — Suppresses FutureWarning and PerformanceWarning

## Current Active Model Configuration (predict_action.py)
| Model | Artifact Suffix | Feature Set | Task |
|-------|----------------|-------------|------|
| Move Predictor | `v4_medium_move_only` | medium | Multi-class: which move |
| Binary Switch | `v2_simplified` | simplified | Binary: move or switch |
| Switch Target | `simplified_moves` | simplified | Multi-class: which Pokémon |

## Environment Variables
| Variable | Default | Description |
|---------|---------|-------------|
| `SHOWDOWN_USER` | `www31` | Showdown bot username |
| `SHOWDOWN_PASS` | `vimvimvim333` | Showdown bot password |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | TF log verbosity suppressed |
