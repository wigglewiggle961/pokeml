# PokéML 

**PokéML** is a machine learning-powered competitive Pokémon Showdown AI. It predicts opponent actions in Gen 9 OU (Overused) tier battles by analyzing real-time game state against models trained on millions of turns of competitive replay data.

## Overview

The project follows a complete ML pipeline:
1. **Data Collection**: Bulk downloading Showdown replay logs. (I might just use metamon dataset instead in the future)
2. **Preprocessing**: Parsing `.log` files into structured Parquet datasets with rich feature engineering (HP, status, boosts, hazards, revealed moves, etc.).
3. **Training**: Training multiple specialized models for different aspects of the game.
4. **Deployment**: A real-time battle bot (`PredictionPlayer`) that connects to Showdown and makes decisions based on model ensembles.

## Key Components

### 1. Data Parsing & Augmentation
- `process_replays.py`: Converts raw Showdown logs into structured Parquet format.
- `augment_perspectives.py`: Augments data by flipping perspectives to increase training diversity.

### 2. Model Training
- `train_action_predictor.py`: Trains a multi-class move predictor using **LightGBM** (primary) or a plain TF Dense network. Stable pipeline — do not modify.
- `train_action_predictor_embedding.py`: *(In development — Phase 4)* TF-only trainer that uses **learned entity embeddings** (species, moves, types) instead of one-hot encoding, targeting higher generalisation accuracy.
- `train_switch_predictor.py`: A binary classifier predicting **if** the opponent will switch.
- `train_pokemon_switch_predictor.py`: A multi-class classifier predicting **which Pokémon** the opponent will switch to.

### 2a. Shared Utilities
- `feature_engineering.py`: *(In development — Phase 4)* Shared helpers extracted from the training scripts: `sanitize_name`, `bin_hp`, `get_smogon_usages_df`, `find_active_species`, and the medium feature set builder. Imported by both training scripts to avoid duplication.

### 3. Prediction Bot
- `predict_action.py`: The core bot implementation. It uses an ensemble approach:
    - First, it checks the probability of an opponent switch.
    - If a switch is likely, it uses the switch-target model.
    - Otherwise, it predicts the most likely move.
    - Validates predictions against Smogon usage stats (`gen9ou-0.json`) and current battle constraints.

## Quick Start

### Process Replays
```bash
python process_replays.py ./replays_dir output.parquet --format parquet
```

### Train Move Predictor (LightGBM / plain TF)
```bash
python train_action_predictor.py data.parquet --model_type lightgbm --feature_set medium
```

### Train Move Predictor (Embedding TF — Phase 4)
```bash
python train_action_predictor_embedding.py data.parquet --feature_set medium
```

### Run the Bot
```bash
python predict_action.py
```

## Dependencies
- `poke-env`: Pokémon Showdown client library.
- `lightgbm` / `tensorflow`: Machine learning frameworks.
- `pandas`, `numpy`, `scikit-learn`, `joblib`.
- `optuna`: Hyperparameter optimization.
