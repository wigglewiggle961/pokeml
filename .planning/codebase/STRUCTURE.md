# Repository Structure

## Root Directory Layout

```
pokeml/
├── .agent/                          # GSD workflow agents and skills
│   └── skills/                      # Per-skill SKILL.md files
├── .planning/                       # GSD planning artifacts (created by /gsd-map-codebase)
│   └── codebase/                    # This documentation
├── .venv/                           # Python virtual environment
├── .vscode/                         # VS Code editor settings
├── extension/                       # Browser extension (likely Showdown helper)
├── metamon_parquet_files/           # External Metamon dataset (Parquet format)
├── metamon_processed/               # Processed Metamon data
├── models_archive/                  # Archived/historical model files
├── pokemon-bot-dev/                 # Bot development assets
├── replay-processing/               # Replay processing utilities
├── replay_20k/                      # 20k replay batch directory
├── replay_logs/                     # Raw Showdown replay logs (batch 1)
├── replay_logs_2/                   # Raw Showdown replay logs (batch 2)
├── replay_logs_3/                   # Raw Showdown replay logs (batch 3)
└── __pycache__/                     # Python bytecode cache
```

## Key Source Files

### Data Pipeline
| File | Role | Size |
|------|------|------|
| `process_replays.py` | Core replay parser: `.log` → `.parquet` | 56KB (~998 lines) |
| `augment_perspectives.py` | P1↔P2 perspective flip for data augmentation | 4KB |
| `extract_metamon_logs.py` | Metamon log extraction | 3.5KB |
| `process_metamon_data.py` | Metamon data processing | 10.5KB |
| `download_replays.py` | Single-threaded replay downloader | 10.5KB |
| `bulk_download_replays.py` | Parallel bulk replay downloader | 11KB |

### Model Training
| File | Role | Size |
|------|------|------|
| `train_action_predictor.py` | Move predictor training (LightGBM + TF, `medium`/`full`/`simplified` sets) | 64KB (~1181 lines) |
| `train_switch_predictor.py` | Binary switch classifier training (LightGBM + TF + Optuna HPO) | 60KB (~1083 lines) |
| `train_pokemon_switch_predictor.py` | Switch target multi-class classifier (LightGBM + TF + Optuna HPO) | 42KB (~738 lines) |
| `train_predictor.py` | Legacy training script | 20KB |
| `train_action_predictor_embedding.py` | Embedding-based action predictor experiment | 15KB |
| `train_pokemon_switch_predictor.py` | Switch target predictor (top-100 Smogon filter) | 42KB |

### Inference & Bot
| File | Role | Size |
|------|------|------|
| `predict_action.py` | Live battle bot: `PredictionPlayer` class + all inference logic | 65KB (~1262 lines) |
| `verify_parity.py` | Parity verification script comparing training vs Smogon sanitization | 2KB (~55 lines) |
| `bot.py` | Earlier/legacy bot implementation | 13KB |
| `demonstration.py` | Manual demonstration / ad-hoc testing | 55KB |

### EDA & Notebooks
| File | Role |
|------|------|
| `eda.py`, `eda2.py`, `eda3.py` | Exploratory data analysis |
| `catboost.ipynb` | CatBoost model exploration |
| `tensorflow.ipynb` | TF model experiments |
| `dataset.py` | Dataset utility helpers |
| `test.py` | Ad-hoc test script |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick-start guide |
| `PROJECT_CONTEXT.md` | Detailed context doc for onboarding AI assistants |
| `fix-overfitting.md` | Active task tracking for overfitting/parity fixes |

## Model Artifact Files (Root Level)

### Active Models
| Pattern | Description |
|---------|-------------|
| `action_lgbm_model_v4_medium_move_only.txt` | Primary move predictor (LightGBM) |
| `action_lgbm_feature_info_v4_medium_move_only.joblib` | Feature schema for move predictor |
| `action_lgbm_scaler_v4_medium_move_only.joblib` | StandardScaler for move predictor |
| `action_label_encoder_v4_medium_move_only.joblib` | LabelEncoder for move predictor |
| `action_lgbm_checkpoint_v4_medium_move_only.txt` | Last auto-saved checkpoint |
| `action_tf_model_v4_medium_move_only.keras` | TF move predictor alternative |
| `action_tf_preprocessor_v4_medium_move_only.joblib` | TF preprocessor pipeline |
| `switch_predictor_lgbm_model_v2_simplified.txt` | Binary switch predictor |
| `switch_predictor_lgbm_feature_info_v2_simplified.joblib` | Binary switch feature schema |
| `switch_predictor_lgbm_scaler_v2_simplified.joblib` | Binary switch scaler |
| `switch_target_predictor_lgbm_model_simplified_moves.txt` | Switch target predictor |
| `switch_target_predictor_lgbm_feature_info_simplified_moves.joblib` | Switch target feature schema |
| `switch_target_predictor_lgbm_scaler_simplified_moves.joblib` | Switch target scaler |
| `switch_target_predictor_label_encoder_simplified_moves.joblib` | Switch target encoder |

### Training Data Files
| File | Size | Description |
|------|------|-------------|
| `30k.parquet` | ~22MB | Primary training set |
| `30k_augmented.parquet` | ~113MB | Augmented training set |
| `10k.parquet` | ~7MB | Small training set |
| `10k.csv` | ~385MB | Legacy CSV format |
| `5000.parquet` | ~3.7MB | 5k-replay set |
| `test.parquet` | ~540KB | Hold-out test set |
| `only_p1.parquet` | ~305KB | P1-only filtered dataset |
| `gen9ou-0.json` | ~15MB | Smogon usage statistics |

## Naming Conventions for Model Artifacts
```
{model_type}_{framework}_model_{version}_{feature_set}_{predict_mode}.{ext}

Examples:
  action_lgbm_model_v4_medium_move_only.txt
  switch_predictor_lgbm_model_v2_simplified.txt
  switch_target_predictor_lgbm_model_simplified_moves.txt
```
