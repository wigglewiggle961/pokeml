# Architecture

## System Overview

PokéML is a competitive Pokémon Showdown battle AI. The system is composed of three decoupled stages: **data ingestion**, **offline model training**, and **real-time inference**.

```
┌─────────────────────────────────────────────────────────┐
│                   DATA INGESTION STAGE                  │
│                                                         │
│  Pokémon Showdown Servers                               │
│         │                                               │
│         ▼                                               │
│  download_replays.py / bulk_download_replays.py         │
│         │  Raw .log files                               │
│         ▼                                               │
│  process_replays.py  ──────────────────────────────┐   │
│         │  Parses log events (move/switch/hazards)  │   │
│         │  Produces turn-by-turn feature rows       │   │
│  augment_perspectives.py (optional)                 │   │
│         │  Flips p1↔p2 to double training data     │   │
│         ▼                                               │
│  *.parquet  (30k.parquet, 30k_augmented.parquet, etc.) │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 MODEL TRAINING STAGE                    │
│                                                         │
│  train_action_predictor.py                             │
│    → action_lgbm_model_v4_{set}_{mode}.txt + artifacts │
│                                                         │
│  train_switch_predictor.py                             │
│    → switch_predictor_lgbm_model_v2_{set}.txt          │
│                                                         │
│  train_pokemon_switch_predictor.py                     │
│    → switch_target_predictor_lgbm_model_{set}.txt      │
│                                                         │
│  All trainers emit:                                     │
│    *_feature_info_*.joblib  (feature schema)           │
│    *_scaler_*.joblib        (StandardScaler)           │
│    *_label_encoder_*.joblib (LabelEncoder, if used)    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               REAL-TIME INFERENCE STAGE                 │
│                                                         │
│  predict_action.py  (PredictionPlayer)                 │
│    │                                                    │
│    ├─ Battle state → map_battle_to_dataframe_row()     │
│    │      └─ Produces flat dict matching parquet schema│
│    │                                                    │
│    ├─ Binary Switch Model                              │
│    │      └─ "Should I switch?" (prob > 0.85 → yes)   │
│    │                                                    │
│    ├─ Switch Target Model (if switch prob > threshold) │
│    │      └─ "Which Pokémon should I switch to?"       │
│    │                                                    │
│    └─ Move Predictor Model (if no switch)              │
│           └─ "Which move should I use?"                │
│              Filtered by Smogon valid moves per species│
│                                                        │
│  poke-env WebSocket client → Showdown server           │
└─────────────────────────────────────────────────────────┘
```

## Model Ensemble Decision Flow

```
[New Turn in Battle]
        │
        ▼
[Map Battle State → DataFrame Row]
        │
        ▼
[Binary Switch Model]
   P(switch) > 0.85?
   ┌──── YES ─────┐         ┌──── NO ────┐
   ▼               │         │            ▼
[Switch Target]    │         │      [Move Predictor]
  Multi-class      │         │        Multi-class
  5 bench slots    │         │        top-k moves
        │           │         │            │
        ▼           │         │            ▼
[Filter: Top-100   │         │  [Filter: Smogon valid
 Smogon species]   │         │   moves for active mon]
        │           │         │            │
        └───────────┘         └────────────┘
                              │
                              ▼
                    [Select best valid action]
                              │
                              ▼
                    [Send to Showdown via poke-env]
```

## Data Flow: Replay Parsing Detail

`process_replays.py` implements a **two-phase parser**:

1. **Phase 1 (Setup)**: Scan log for `|teamsize|` and `|poke|` lines to initialize slot tracking
2. **Phase 2 (Events)**: Process turn-by-turn events, maintaining mutable game state:
   - `|switch|` / `|drag|` — activates Pokémon, resets boosts
   - `|move|` — records action, updates revealed moves set
   - `|-damage|` / `|-heal|` — updates HP percentage
   - `|-status|` / `|-curestatus|` — tracks status conditions
   - `|-boost|` / `|-unboost|` — tracks stat stages (–6 to +6)
   - `|-sidestart|` / `|-sideend|` — hazards and screens
   - `|-weather|` / `|-terrain|` / `|-fieldstart|` — field conditions
   - `|-terastallize|` — Tera type tracking
   - `|-formechange|` — Species updates (e.g., Mimikyu-Busted)

At each player action, a **deep copy** of the current game state is recorded as a training row.

## Key Design Decisions

### Bot Perspective
Data is always from P1's perspective. The training filter `player_to_move == 'p1'` ensures consistent modeling of the bot's own actions.

### Slot-Based Representation
Teams are represented as 6 fixed slots rather than by species — this allows modeling of positional strategy (who is on the field, who is in reserve).

### Smogon Move Filtering
At inference time, predicted move probabilities are masked to only Smogon-listed moves for the active species. This prevents the model from hallucinating non-viable moves.

### Species Normalization
All species names are normalized to lowercase with no hyphens or spaces (e.g., `Great Tusk` → `greattusk`). This is critical for consistent lookup between the replay parser, training data, and inference code.

### Feature Parity Requirement
The feature engineering in `prepare_input_data_medium()` must exactly mirror what `train_action_predictor.py` produces during training. Any mismatch causes silent prediction degradation.
