"""
train_action_predictor_embedding.py
TensorFlow embedding-based action predictor for PokéML.

Architecture: Keras Functional API with entity embeddings for categorical features.
- Species, moves, status, tera type → learned embedding vectors
- Multi-hot revealed moves → Dense(dim, use_bias=False) sum-pool projection
- Numerical features → direct Dense branch
- All branches concatenated → 512→256→128→softmax(num_classes)

Shared utilities: feature_engineering.py
Artifacts saved with prefix: action_tf_embedding_v1_{feature_set}

Usage:
    python train_action_predictor_embedding.py data.parquet --feature_set medium
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import argparse
import os
import joblib
import json
import gc
import glob
import warnings

from feature_engineering import (
    sanitize_name, bin_hp, get_smogon_usages_df,
    find_active_species, build_medium_X
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# --- Vocabulary Encoders for Embedding Inputs ---
# ---------------------------------------------------------------------------
UNKNOWN_TOKEN = '__UNKNOWN__'

# Categorical columns that get embedding layers (not one-hot encoded)
EMBED_COLS_SPECIES = [
    'p1_active_species', 'p2_active_species',
    'p1_slot1_species', 'p1_slot2_species', 'p1_slot3_species',
    'p1_slot4_species', 'p1_slot5_species', 'p1_slot6_species',
    'p2_slot1_species', 'p2_slot2_species', 'p2_slot3_species',
    'p2_slot4_species', 'p2_slot5_species', 'p2_slot6_species',
]
EMBED_COLS_MOVES = ['last_move_p1', 'last_move_p2']
EMBED_COLS_STATUS = ['p1_active_status', 'p2_active_status',
                     'p1_slot1_status', 'p1_slot2_status', 'p1_slot3_status',
                     'p1_slot4_status', 'p1_slot5_status', 'p1_slot6_status',
                     'p2_slot1_status', 'p2_slot2_status', 'p2_slot3_status',
                     'p2_slot4_status', 'p2_slot5_status', 'p2_slot6_status']
EMBED_COLS_HP = [
    'p1_active_hp_perc', 'p2_active_hp_perc',
    'p1_slot1_hp_perc', 'p1_slot2_hp_perc', 'p1_slot3_hp_perc',
    'p1_slot4_hp_perc', 'p1_slot5_hp_perc', 'p1_slot6_hp_perc',
    'p2_slot1_hp_perc', 'p2_slot2_hp_perc', 'p2_slot3_hp_perc',
    'p2_slot4_hp_perc', 'p2_slot5_hp_perc', 'p2_slot6_hp_perc',
]
EMBED_COLS_TYPE = ['p1_active_tera_type', 'p2_active_tera_type']
EMBED_COLS_FIELD = ['field_weather', 'field_terrain', 'field_pseudo_weather']

# These are treated as numerical (multi-hot / boosts / hazards / usage stats)
# and do NOT get embedding layers.
REVEALED_MOVE_PREFIX_P1 = 'p1_active_revealed_move_'
REVEALED_MOVE_PREFIX_P2 = 'p2_active_revealed_move_'


def build_vocab_encoders(X_train, categorical_embed_cols):
    """
    Fit LabelEncoders per column for embedding lookup.
    Each encoder includes __UNKNOWN__ at index 0 for unseen values.
    Returns:
        vocab_encoders: dict[col -> LabelEncoder]
        vocab_sizes: dict[col -> int]  (includes +1 for unknown)
    """
    print("Building vocabulary encoders for embedding inputs...")

    # Use shared vocabs for logically equivalent columns
    # Species: union of all species columns
    species_cols = [c for c in categorical_embed_cols if c in EMBED_COLS_SPECIES]
    move_cols = [c for c in categorical_embed_cols if c in EMBED_COLS_MOVES]
    status_cols = [c for c in categorical_embed_cols if c in EMBED_COLS_STATUS]
    hp_cols = [c for c in categorical_embed_cols if c in EMBED_COLS_HP]
    type_cols = [c for c in categorical_embed_cols if c in EMBED_COLS_TYPE]
    field_cols = [c for c in categorical_embed_cols if c in EMBED_COLS_FIELD]
    other_cols = [c for c in categorical_embed_cols
                  if c not in species_cols + move_cols + status_cols + hp_cols + type_cols + field_cols]

    def fit_shared_encoder(cols, df):
        """Fit one encoder on the union of values across all cols."""
        all_vals = set([UNKNOWN_TOKEN])
        for c in cols:
            if c in df.columns:
                all_vals.update(df[c].fillna(UNKNOWN_TOKEN).astype(str).unique())
        le = LabelEncoder()
        le.fit(sorted(list(all_vals)))
        return le

    vocab_encoders = {}
    vocab_sizes = {}

    # Fit shared encoders for each group
    groups = [
        ('species', species_cols),
        ('move', move_cols),
        ('status', status_cols),
        ('hp', hp_cols),
        ('type', type_cols),
        ('field', field_cols),
    ]

    for group_name, cols in groups:
        if not cols:
            continue
        le = fit_shared_encoder(cols, X_train)
        vocab_size = len(le.classes_)
        print(f"  Vocab '{group_name}': {vocab_size} entries, shared by {len(cols)} columns")
        for c in cols:
            vocab_encoders[c] = le
            vocab_sizes[c] = vocab_size

    # Other categorical columns: individual encoders
    for c in other_cols:
        if c not in X_train.columns:
            continue
        vals = sorted(list(set([UNKNOWN_TOKEN]) | set(X_train[c].fillna(UNKNOWN_TOKEN).astype(str).unique())))
        le = LabelEncoder()
        le.fit(vals)
        vocab_encoders[c] = le
        vocab_sizes[c] = len(le.classes_)

    print(f"  Total embedding columns: {len(vocab_encoders)}")
    return vocab_encoders, vocab_sizes


# ---------------------------------------------------------------------------
# --- Build Keras Functional API Embedding Model ---
# ---------------------------------------------------------------------------

def build_embedding_model(vocab_sizes, categorical_embed_cols, num_numerical,
                          num_revealed_p1, num_revealed_p2, num_classes,
                          label_smoothing=0.05):
    """
    Build Keras Functional API model with entity embedding branches.
    """
    print(f"Building embedding model: {len(categorical_embed_cols)} embed cols, "
          f"{num_numerical} numerical, {num_revealed_p1}+{num_revealed_p2} revealed move dims, "
          f"{num_classes} output classes")

    # Embedding dimension heuristic
    def embed_dim(vocab_size, group):
        if group == 'species':
            return 48
        elif group == 'move':
            return 32
        elif group == 'hp':
            return 8
        elif group in ('status', 'type', 'field'):
            return 8
        else:
            return max(4, min(32, vocab_size // 4))

    def get_group(col):
        if col in EMBED_COLS_SPECIES: return 'species'
        if col in EMBED_COLS_MOVES: return 'move'
        if col in EMBED_COLS_STATUS: return 'status'
        if col in EMBED_COLS_HP: return 'hp'
        if col in EMBED_COLS_TYPE: return 'type'
        if col in EMBED_COLS_FIELD: return 'field'
        return 'other'

    inputs = {}
    embed_outputs = []

    for col in categorical_embed_cols:
        if col not in vocab_sizes:
            continue
        vs = vocab_sizes[col]
        group = get_group(col)
        dim = embed_dim(vs, group)

        inp = layers.Input(shape=(1,), name=f'input_{col}', dtype='int32')
        inputs[col] = inp
        emb = layers.Embedding(input_dim=vs, output_dim=dim, name=f'emb_{col}')(inp)
        flat = layers.Flatten(name=f'flat_{col}')(emb)
        embed_outputs.append(flat)

    # Numerical branch
    num_inp = layers.Input(shape=(num_numerical,), name='input_numerical', dtype='float32')
    inputs['numerical'] = num_inp
    num_branch = layers.Dense(64, activation='swish', name='num_proj')(num_inp)
    embed_outputs.append(num_branch)

    # Revealed moves branches (sum-pool via Dense with no bias)
    if num_revealed_p1 > 0:
        rev_p1_inp = layers.Input(shape=(num_revealed_p1,), name='input_revealed_p1', dtype='float32')
        inputs['revealed_p1'] = rev_p1_inp
        rev_p1 = layers.Dense(32, use_bias=False, name='move_pool_p1')(rev_p1_inp)
        embed_outputs.append(rev_p1)

    if num_revealed_p2 > 0:
        rev_p2_inp = layers.Input(shape=(num_revealed_p2,), name='input_revealed_p2', dtype='float32')
        inputs['revealed_p2'] = rev_p2_inp
        rev_p2 = layers.Dense(32, use_bias=False, name='move_pool_p2')(rev_p2_inp)
        embed_outputs.append(rev_p2)

    # Concatenate all branches
    if len(embed_outputs) == 1:
        x = embed_outputs[0]
    else:
        x = layers.Concatenate(name='concat_all')(embed_outputs)

    # Deep classification head
    x = layers.Dense(512, use_bias=False, name='dense1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('swish', name='act1')(x)
    x = layers.Dropout(0.25, name='drop1')(x)

    x = layers.Dense(256, use_bias=False, name='dense2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('swish', name='act2')(x)
    x = layers.Dropout(0.25, name='drop2')(x)

    x = layers.Dense(128, use_bias=False, name='dense3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Activation('swish', name='act3')(x)
    x = layers.Dropout(0.2, name='drop3')(x)

    output = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=output, name='action_predictor_embedding')

    # One-hot conversion for label smoothing support
    # Use CategoricalCrossentropy with label_smoothing
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# --- Prepare Dict-Based Inputs ---
# ---------------------------------------------------------------------------

def prepare_inputs(X, vocab_encoders, categorical_embed_cols,
                   numerical_features, revealed_p1_cols, revealed_p2_cols,
                   scaler=None, fit_scaler=False):
    """
    Convert a feature DataFrame into a dict of numpy arrays for model.fit().
    Returns (input_dict, scaler)
    """
    input_dict = {}

    # Categorical embedding inputs
    unk_idx_cache = {}
    for col in categorical_embed_cols:
        if col not in vocab_encoders or col not in X.columns:
            continue
        le = vocab_encoders[col]
        if col not in unk_idx_cache:
            try:
                unk_idx_cache[col] = le.transform([UNKNOWN_TOKEN])[0]
            except Exception:
                unk_idx_cache[col] = 0
        unk_idx = unk_idx_cache[col]

        vals = X[col].fillna(UNKNOWN_TOKEN).astype(str).values
        # Map unseen values to UNKNOWN
        classes_set = set(le.classes_)
        vals_safe = np.where(np.isin(vals, list(classes_set)), vals, UNKNOWN_TOKEN)
        encoded = le.transform(vals_safe).astype(np.int32).reshape(-1, 1)
        input_dict[col] = encoded

    # Numerical branch
    if numerical_features:
        num_data = X[numerical_features].values.astype(np.float32)
        if fit_scaler:
            scaler = StandardScaler()
            num_data = scaler.fit_transform(num_data).astype(np.float32)
        elif scaler is not None:
            num_data = scaler.transform(num_data).astype(np.float32)
        input_dict['numerical'] = num_data

    # Revealed moves branches
    if revealed_p1_cols:
        p1_data = X[revealed_p1_cols].values.astype(np.float32)
        input_dict['revealed_p1'] = p1_data

    if revealed_p2_cols:
        p2_data = X[revealed_p2_cols].values.astype(np.float32)
        input_dict['revealed_p2'] = p2_data

    return input_dict, scaler


# ---------------------------------------------------------------------------
# --- Main Training Function ---
# ---------------------------------------------------------------------------

def train_embedding_model(parquet_path, feature_set='medium',
                          epochs=100000, batch_size=256, learning_rate=0.001,
                          label_smoothing=0.05,
                          min_feature_replay_count=50,
                          disable_smogon_features=False, smogon_top_n=100,
                          test_split_size=0.2, val_split_size=0.15,
                          min_move_count=0):
    """
    Full training pipeline for the TF embedding action predictor.
    """
    print(f"\n--- Starting Embedding Action Predictor Training ---")
    print(f"Feature Set: {feature_set.upper()}")
    print(f"Loading data from: {parquet_path}")

    # --- Load Data ---
    try:
        if os.path.isdir(parquet_path):
            files = glob.glob(os.path.join(parquet_path, "*.parquet"))
            if not files:
                files = glob.glob(os.path.join(parquet_path, "**", "*.parquet"), recursive=True)
            if not files:
                raise FileNotFoundError(f"No parquet files found in: {parquet_path}")
            print(f"  Found {len(files)} parquet files. Loading...")
            dfs = []
            for f in files:
                tmp = pd.read_parquet(f)
                for col in tmp.select_dtypes(include=['float64']).columns:
                    tmp[col] = tmp[col].astype(np.float32)
                for col in tmp.select_dtypes(include=['int64']).columns:
                    tmp[col] = tmp[col].astype(np.int32)
                dfs.append(tmp)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(parquet_path)
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = df[col].astype(np.int32)
        print(f"Data loaded. Shape: {df.shape}")
        gc.collect()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Filter ---
    print("\nFiltering data...")
    df = df.dropna(subset=['action_taken'])

    if feature_set in ['simplified', 'medium']:
        df = df[df['player_to_move'] == 'p1'].copy()
        print(f"Rows after p1 filter: {len(df)}")

    df = df[df['action_taken'].astype(str).str.startswith('move:')].copy()
    if df.empty:
        print("Error: No 'move:' actions found.")
        return
    y_raw = df['action_taken'].str.replace('move:', '', regex=False)
    print(f"Rows after move: filter: {len(df)}")

    if min_move_count > 0:
        action_counts = y_raw.value_counts()
        common_actions = action_counts[action_counts >= min_move_count].index
        mask = y_raw.isin(common_actions)
        df = df[mask].copy()
        y_raw = y_raw[mask].copy()
        print(f"Rows after min_move_count={min_move_count} filter: {len(df)}")

    if df.empty:
        print("Error: No data remaining after filtering.")
        return

    # --- Build Features ---
    if feature_set == 'medium':
        X, numerical_features, categorical_features = build_medium_X(
            df,
            disable_smogon_features=disable_smogon_features,
            smogon_top_n=smogon_top_n,
            min_feature_replay_count=min_feature_replay_count
        )
    else:
        print(f"Error: feature_set '{feature_set}' not supported. Use 'medium'.")
        return

    # --- Encode Target ---
    print("\nEncoding target variable...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw.astype(str))
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} unique moves.")

    suffix = feature_set
    label_encoder_path = f'action_label_encoder_v1_{suffix}.joblib'
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")

    # --- GroupShuffleSplit by replay_id ---
    if 'replay_id' not in df.columns:
        print("Error: 'replay_id' column missing for group splitting.")
        return
    groups = df['replay_id']
    if len(groups) != len(X):
        print(f"Error: Group length ({len(groups)}) != X length ({len(X)}).")
        return

    print("\nSplitting data (GroupShuffleSplit by replay_id)...")
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split_size, random_state=42)
    train_full_idx, test_idx = next(gss_test.split(X, y_encoded, groups=groups))
    X_train_full = X.iloc[train_full_idx]
    y_train_full = y_encoded[train_full_idx]
    groups_train_full = groups.iloc[train_full_idx]
    X_test = X.iloc[test_idx]
    y_test = y_encoded[test_idx]

    train_full_size = 1.0 - test_split_size
    val_rel = val_split_size / train_full_size if train_full_size > 0 else 0.15
    if not (0 < val_rel < 1): val_rel = 0.15

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=42)
    train_idx, val_idx = next(gss_val.split(X_train_full, y_train_full, groups=groups_train_full))
    X_train = X_train_full.iloc[train_idx]
    y_train = y_train_full[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full[val_idx]

    # Verify no leakage
    train_r = set(groups_train_full.iloc[train_idx])
    val_r = set(groups_train_full.iloc[val_idx])
    test_r = set(groups.iloc[test_idx])
    assert not train_r & val_r, "Leakage: train/val overlap!"
    assert not train_r & test_r, "Leakage: train/test overlap!"
    assert not val_r & test_r, "Leakage: val/test overlap!"
    print(f"  Train: {len(X_train)} rows | Val: {len(X_val)} rows | Test: {len(X_test)} rows")
    print("  No replay overlap detected. ✓")

    del X, df, X_train_full, y_train_full, groups, groups_train_full; gc.collect()

    # --- Identify Feature Groups ---
    all_cols = X_train.columns.tolist()

    # Revealed move columns (binary multi-hot)
    revealed_p1_cols = sorted([c for c in all_cols if c.startswith(REVEALED_MOVE_PREFIX_P1)])
    revealed_p2_cols = sorted([c for c in all_cols if c.startswith(REVEALED_MOVE_PREFIX_P2)])

    # Smogon usage columns (numerical)
    usage_cols = [c for c in all_cols if '_active_usage_' in c]

    # Categorical columns that get embeddings
    all_embed_candidates = (EMBED_COLS_SPECIES + EMBED_COLS_MOVES + EMBED_COLS_STATUS +
                           EMBED_COLS_HP + EMBED_COLS_TYPE + EMBED_COLS_FIELD)
    categorical_embed_cols = [c for c in all_embed_candidates if c in all_cols]

    # Pure numerical (boosts, hazards, side conds, turn_number, usage stats, terastallized)
    embed_and_revealed = set(categorical_embed_cols + revealed_p1_cols + revealed_p2_cols + usage_cols)
    pure_numerical = [c for c in numerical_features if c not in embed_and_revealed]
    # Add usage cols to numerical
    numerical_for_model = sorted(list(set(pure_numerical + usage_cols)))

    print(f"\nFeature groups:")
    print(f"  Embedding categorical: {len(categorical_embed_cols)}")
    print(f"  Revealed moves P1: {len(revealed_p1_cols)}")
    print(f"  Revealed moves P2: {len(revealed_p2_cols)}")
    print(f"  Numerical (pure): {len(numerical_for_model)}")

    # --- Vocabulary Encoders ---
    vocab_encoders, vocab_sizes = build_vocab_encoders(X_train, categorical_embed_cols)

    # --- Prepare Inputs ---
    print("\nPreparing model inputs...")
    train_inputs, scaler = prepare_inputs(
        X_train, vocab_encoders, categorical_embed_cols,
        numerical_for_model, revealed_p1_cols, revealed_p2_cols,
        fit_scaler=True
    )
    val_inputs, _ = prepare_inputs(
        X_val, vocab_encoders, categorical_embed_cols,
        numerical_for_model, revealed_p1_cols, revealed_p2_cols,
        scaler=scaler
    )
    test_inputs, _ = prepare_inputs(
        X_test, vocab_encoders, categorical_embed_cols,
        numerical_for_model, revealed_p1_cols, revealed_p2_cols,
        scaler=scaler
    )

    # One-hot encode targets for CategoricalCrossentropy
    print("One-hot encoding targets...")
    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_val_oh = to_categorical(y_val, num_classes=num_classes)
    y_test_oh = to_categorical(y_test, num_classes=num_classes)

    del X_train, X_val, X_test; gc.collect()

    # --- Class Weights ---
    print("\nCalculating class weights...")
    unique_cls, _ = np.unique(y_train, return_counts=True)
    if len(unique_cls) >= 2:
        cw_vals = compute_class_weight('balanced', classes=unique_cls, y=y_train)
        class_weight_dict = dict(zip(unique_cls.tolist(), cw_vals.tolist()))
        for c in range(num_classes):
            if c not in class_weight_dict:
                class_weight_dict[c] = 1.0
    else:
        class_weight_dict = {c: 1.0 for c in range(num_classes)}

    # --- Build Model ---
    model = build_embedding_model(
        vocab_sizes=vocab_sizes,
        categorical_embed_cols=categorical_embed_cols,
        num_numerical=len(numerical_for_model),
        num_revealed_p1=len(revealed_p1_cols),
        num_revealed_p2=len(revealed_p2_cols),
        num_classes=num_classes,
        label_smoothing=label_smoothing
    )

    if learning_rate != 0.001:
        model.optimizer.learning_rate.assign(learning_rate)

    # --- Train ---
    print("\nStarting training...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        train_inputs, y_train_oh,
        validation_data=(val_inputs, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )
    print("Training finished.")

    # --- Evaluate ---
    print("\nEvaluating on test set...")
    results = model.evaluate(test_inputs, y_test_oh, verbose=0)
    test_loss = results[0]
    test_acc = results[1]
    test_top5 = results[2] if len(results) > 2 else float('nan')
    print(f"TF Embedding Test Loss: {test_loss:.4f}")
    print(f"TF Embedding Test Accuracy: {test_acc:.4f}")
    print(f"TF Embedding Test Top-5 Accuracy: {test_top5:.4f}")

    # --- Save Artifacts ---
    model_path = f'action_tf_embedding_v1_{suffix}.keras'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    metadata = {
        'feature_set': feature_set,
        'suffix': suffix,
        'categorical_embed_cols': categorical_embed_cols,
        'numerical_features': numerical_for_model,
        'revealed_p1_cols': revealed_p1_cols,
        'revealed_p2_cols': revealed_p2_cols,
        'vocab_sizes': vocab_sizes,
        'num_classes': num_classes,
        'test_accuracy': float(test_acc),
        'test_top5_accuracy': float(test_top5),
        'test_loss': float(test_loss),
    }
    meta_path = f'action_tf_embedding_metadata_v1_{suffix}.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    artifacts_path = f'action_tf_embedding_artifacts_v1_{suffix}.joblib'
    joblib.dump({
        'vocab_encoders': vocab_encoders,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'categorical_embed_cols': categorical_embed_cols,
        'numerical_features': numerical_for_model,
        'revealed_p1_cols': revealed_p1_cols,
        'revealed_p2_cols': revealed_p2_cols,
    }, artifacts_path)
    print(f"Artifacts saved to {artifacts_path}")

    return model, history


# ---------------------------------------------------------------------------
# --- CLI ---
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train TF Embedding Action Predictor for PokéML (v1)."
    )
    parser.add_argument('parquet_file', type=str,
                        help="Path to input Parquet file or directory.")
    parser.add_argument('--feature_set', choices=['medium'], default='medium',
                        help="Feature set to use (default: medium).")
    parser.add_argument('--epochs', type=int, default=100000,
                        help="Max training epochs (EarlyStopping will stop early).")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Initial learning rate.")
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help="Label smoothing for CategoricalCrossentropy (default 0.05).")
    parser.add_argument('--min_feature_replay_count', type=int, default=50,
                        help="Min unique replays for a revealed move feature to be kept.")
    parser.add_argument('--disable_smogon_features', action='store_true',
                        help="Disable Smogon usage stats injection.")
    parser.add_argument('--smogon_top_n', type=int, default=100,
                        help="Top-N Smogon moves to include (default: 100).")
    parser.add_argument('--test_split', type=float, default=0.2,
                        help="Test set fraction.")
    parser.add_argument('--val_split', type=float, default=0.15,
                        help="Validation set fraction.")
    parser.add_argument('--min_move_count', type=int, default=0,
                        help="Minimum move occurrences to include (0=no filter).")
    args = parser.parse_args()

    if args.test_split + args.val_split >= 1.0:
        print("Error: test_split + val_split must be < 1.0")
        exit(1)

    train_embedding_model(
        parquet_path=args.parquet_file,
        feature_set=args.feature_set,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
        min_feature_replay_count=args.min_feature_replay_count,
        disable_smogon_features=args.disable_smogon_features,
        smogon_top_n=args.smogon_top_n,
        test_split_size=args.test_split,
        val_split_size=args.val_split,
        min_move_count=args.min_move_count,
    )
