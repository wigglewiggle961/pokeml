# PokéML Project Context

> **Use this prompt at the start of new AI conversations to quickly onboard the assistant.**

---

## Project Overview

This is **PokéML**, a **Pokémon Showdown competitive battle AI** project focused on predicting opponent actions in Gen 9 OU (Overused) tier battles. The codebase trains machine learning models on parsed replay data and deploys a real-time battle bot using the `poke-env` library.

---

## Repository Layout (as of 2026-07-05 cleanup)

```
pokeml/
├── *.py                  Live pipeline + data-collection + EDA scripts (run from repo root)
├── data/                 Datasets (*.parquet, *.csv) + gen9ou-0.json   [gitignored]
├── models/               All trained artifacts (.txt/.keras/.joblib) + models_archive/  [gitignored]
├── notebooks/            Exploratory notebooks (catboost.ipynb, tensorflow.ipynb)
├── archive/              Superseded/scratch scripts (bot.py, train_predictor.py, etc.)  [gitignored]
├── replay_logs*/         Raw Showdown .log files (re-downloadable)  [gitignored]
├── metamon_*/            Metamon dataset working dirs  [gitignored]
└── .planning/            Progress, roadmap, experiments log  [tracked]
```

> **Run scripts from the repo root** — paths are root-relative (`data/…`, `models/…`).
> `data/` and `models/` are gitignored (not backed up by git); see `.planning/experiments.md`
> for the run→dataset→artifact mapping that makes results reproducible.

## Core Pipeline

```
Pokemon Showdown Replays (.log files)
         ↓
   process_replays.py (parses replay logs → Parquet dataset)
         ↓
   feature_engineering.py (shared helpers: sanitize, bin_hp, Smogon stats, feature builders)
         ↓
   Training Scripts (train_*.py)
     ├─ train_action_predictor.py         ← LightGBM + plain TF (STABLE — do not modify)
     └─ train_action_predictor_embedding.py ← TF Embedding model (Phase 4, in development)
         ↓
   LightGBM / TensorFlow Models (.txt/.keras + .joblib artifacts)
         ↓
   predict_action.py (PredictionPlayer bot connecting to Showdown)
```

---

## Key Files & Their Roles

| File | Purpose |
|------|---------|
| `process_replays.py` | Parses Showdown `.log` replay files into structured Parquet datasets. Extracts turn-by-turn game state (team slots, HP, status, hazards, boosts, terrain, weather, revealed moves) and labels each state with the action taken (`move:X` or `switch:Y`). Uses normalized species names (lowercase, no hyphens/spaces). |
| `feature_engineering.py` | *(Phase 4 — in development)* **Shared module** containing all helper functions and the medium feature set builder extracted from the training scripts: `sanitize_name`, `bin_hp`, `get_smogon_usages_df`, `find_active_species`, multi-hot encoding logic, and Smogon usage injection. Imported by both `train_action_predictor.py` and `train_action_predictor_embedding.py`. |
| `train_action_predictor.py` | **STABLE — do not modify.** Trains a **multi-class move predictor** for what specific move will be used. Supports `full`, `medium`, and `simplified` feature sets. Uses LightGBM (primary) or a plain TF Dense network. |
| `train_action_predictor_embedding.py` | *(Phase 4 — in development)* **TF-only** move predictor that replaces one-hot encoding with **learned entity embeddings** for species, moves, and types. Multi-input architecture: categorical inputs → Embedding layers; numerical inputs → Dense layers; concatenated into shared classifier head. Targets val_accuracy > 30%. |
| `train_switch_predictor.py` | Trains a **binary classifier** (0=Move, 1=Switch) predicting whether a player will switch. Uses Optuna HPO for LightGBM. |
| `train_pokemon_switch_predictor.py` | Trains a **multi-class classifier** predicting *which Pokémon* will be switched into (slot-based, 5 bench slots). Filters to top-100 Smogon usage Pokémon. |
| `predict_action.py` | The live battle bot. Implements `PredictionPlayer` (extends `poke-env.Player`). Loads trained models + scalers + encoders at runtime. Maps live `Battle` objects to feature DataFrames, runs predictions through: 1) Binary switch model (should I switch?), 2) If switch probability > threshold → switch target model, 3) Else → move model. Uses Smogon usage JSON (`gen9ou-0.json`) to filter predicted moves to valid options per species. |

---

## Feature Engineering Details

### Slot-Based Team Representation
- `p1_slot1` through `p1_slot6` (and `p2_*`) with subfields:
  - `_species` - Pokémon species (normalized: lowercase, no spaces/hyphens)
  - `_hp_perc` - HP percentage (0-100)
  - `_status` - Status condition (none, brn, par, slp, frz, psn, tox, fnt)
  - `_is_active` - Binary (1 if currently on field)
  - `_is_fainted` - Binary (1 if KO'd)
  - `_terastallized` - Binary (1 if Terastallized)
  - `_tera_type` - Tera type if applicable
  - `_boost_{stat}` - Stat boosts for atk, def, spa, spd, spe (-6 to +6)
  - `_revealed_moves` - Comma-separated string of known moves

### Active Pokémon Features
- Extracted dynamically from whichever slot has `is_active=1`
- Prefixed as `p1_active_*` and `p2_active_*`

### Revealed Moves Encoding
- Multi-hot encoded from comma-separated move strings
- Move names are sanitized (underscores replace special chars)
- Binary columns: `{slot}_revealed_moves_{sanitized_move_name}`

### Field State
- `field_weather` - Current weather (none, rain, sun, sand, snow, etc.)
- `field_terrain` - Current terrain (none, electric, grassy, psychic, misty)
- `field_pseudo_weather` - Trick Room, etc.

### Entry Hazards
- `p1_hazard_stealthrock` (0 or 1)
- `p1_hazard_spikes` (0-3 layers)
- `p1_hazard_toxicspikes` (0-2 layers)
- `p1_hazard_stickyweb` (0 or 1)
- Same pattern for `p2_hazard_*`

### Side Conditions
- `p1_side_reflect`, `p1_side_lightscreen`, `p1_side_auroraveil`, `p1_side_tailwind`
- Same pattern for `p2_side_*`

### Context Features
- `last_move_p1`, `last_move_p2` - Previous turn's moves
- `turn_number` - Current battle turn

---

## Model Artifacts Pattern

Models are saved with versioned suffixes:
```
action_lgbm_model_v4_{feature_set}_{predict_mode}.txt
switch_predictor_lgbm_model_v2_{feature_set}.txt
switch_target_predictor_lgbm_model_{feature_set}.txt
```

Each model requires companion files:
- `*_feature_info_*.joblib` - Feature names, category maps, feature order
- `*_scaler_*.joblib` - StandardScaler for numerical features
- `*_label_encoder_*.joblib` - LabelEncoder for target labels

---

## Dependencies

```
poke-env          # Showdown client library
lightgbm          # Primary ML framework
tensorflow/keras  # Alternative ML framework
pandas, numpy     # Data manipulation
scikit-learn      # Preprocessing, metrics
joblib            # Model serialization
optuna            # Hyperparameter optimization
```

---

## Current Configuration (predict_action.py)

| Setting | Value |
|---------|-------|
| Battle Format | `gen9ou` |
| Switch Threshold | `0.85` (probability above this triggers a switch) |
| Smogon Stats File | `gen9ou-0.json` |

---

## Key Assumptions

1. **Data flow**: Replays → Parquet → Models → Live predictions
2. **Species normalization**: Always lowercase, no spaces or hyphens (e.g., `greattusk`, `ironvaliant`)
3. **Primary framework**: LightGBM for production models (`train_action_predictor.py`); TensorFlow Embedding for research/improvement (`train_action_predictor_embedding.py`)
4. **Bot perspective**: Always `p1` (features assume p1 is the bot, p2 is opponent)
5. **Action format**: `move:{movename}` or `switch:{species}`
6. **Stability boundary**: `train_action_predictor.py` is frozen. All new TF experimentation happens in `train_action_predictor_embedding.py` only.

---

## Common Tasks

### Train a new move predictor (LightGBM / plain TF — stable)
```bash
python train_action_predictor.py data.parquet --model_type lightgbm --feature_set medium
```

### Train embedding-based move predictor (TF — Phase 4)
```bash
python train_action_predictor_embedding.py data.parquet --feature_set medium
```

### Train switch predictor (binary)
```bash
python train_switch_predictor.py data.parquet --model_type lightgbm --feature_set simplified
```

### Train switch target predictor
```bash
python train_pokemon_switch_predictor.py data.parquet --model_type lightgbm --feature_set simplified
```

### Process replay logs
```bash
python process_replays.py ./replays_directory output.parquet --format parquet
```

### Run the battle bot
```bash
python predict_action.py
```

---

## File Types in Directory

| Extension | Description |
|-----------|-------------|
| `.log` | Raw Showdown replay logs |
| `.parquet` | Processed training data |
| `.txt` | LightGBM model files |
| `.keras` | TensorFlow model files |
| `.joblib` | Serialized Python objects (scalers, encoders, feature info) |
| `.json` | Smogon usage statistics |
