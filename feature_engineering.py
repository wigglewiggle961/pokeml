"""
feature_engineering.py
Shared feature engineering utilities for PokéML training scripts.

Used by: train_action_predictor_embedding.py
NOT modifying: train_action_predictor.py (stable pipeline — do not import from it)

Functions mirror the helpers in train_action_predictor.py to ensure 100% parity
between the stable pipeline and the new embedding-based pipeline.
"""

import pandas as pd
import numpy as np
import re
import gc
import json
import os


# ---------------------------------------------------------------------------
# --- Helper: sanitize_name ---
# ---------------------------------------------------------------------------
def sanitize_name(name):
    """
    Standardize a move or species name by removing all non-alphanumeric characters
    and converting to lowercase. Ensures 100% parity with Smogon Usage Stats keys.
    """
    if not isinstance(name, str) or name.lower() == 'unknown':
        return 'unknown'
    return re.sub(r'[^a-z0-9]', '', name.lower())


# ---------------------------------------------------------------------------
# --- Helper: bin_hp ---
# ---------------------------------------------------------------------------
def bin_hp(hp_val):
    """
    Converts a 0-100 HP percentage into categorical bins.
    Focuses on critical thresholds: Fainted (0), Sash/Sturdy (1), Rocks (12-25), etc.
    """
    if pd.isna(hp_val) or hp_val <= 0: return 'Fainted'
    if hp_val <= 1: return 'Sash'
    if hp_val <= 25: return 'Critical'
    if hp_val <= 50: return 'Low'
    if hp_val <= 75: return 'Middle'
    if hp_val <= 99: return 'High'
    return 'Full'


# ---------------------------------------------------------------------------
# --- Helper: get_smogon_usages_df ---
# ---------------------------------------------------------------------------
def get_smogon_usages_df(json_filepath, top_n=None):
    """
    Loads Smogon usage stats and creates a DataFrame.
    If top_n is set, keeps only the N most commonly used moves across all species.
    """
    print(f"Loading Smogon usage stats from: {json_filepath}")
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        smogon_data = raw_data.get('data', {})
        usage_rows = []
        all_sanitized_moves = set()

        for pokemon_name, pokemon_data in smogon_data.items():
            # Standardize Pokemon Name from Smogon data
            species_key = sanitize_name(pokemon_name)
            raw_count = pokemon_data.get('Raw count', 1.0)
            if raw_count == 0:
                raw_count = 1.0

            row = {'species': species_key}
            if isinstance(pokemon_data, dict) and 'Moves' in pokemon_data and isinstance(pokemon_data['Moves'], dict):
                moves = pokemon_data['Moves']
                for move, count in moves.items():
                    # Sanitize to match poke-env IDs
                    sanitized_move = sanitize_name(move)
                    all_sanitized_moves.add(sanitized_move)
                    row[sanitized_move] = float(count) / float(raw_count)

            usage_rows.append(row)

        usage_df = pd.DataFrame(usage_rows).set_index('species').fillna(0.0)

        # Pruning Logic
        if top_n is not None and top_n < len(all_sanitized_moves):
            # Sum usage across all species to find the most common moves
            global_usage = usage_df.sum(axis=0)
            top_moves_list = global_usage.sort_values(ascending=False).head(top_n).index.tolist()
            usage_df = usage_df[top_moves_list]
            final_moves = sorted(top_moves_list)
            print(f"Pruned Smogon moves: Kept Top {top_n} out of {len(all_sanitized_moves)} unique moves.")
        else:
            final_moves = sorted(list(all_sanitized_moves))

        print(f"Loaded usages for {len(usage_df)} species and {len(final_moves)} unique moves.")
        return usage_df, final_moves
    except Exception as e:
        print(f"Error loading Smogon JSON: {e}")
        return pd.DataFrame(), []


# ---------------------------------------------------------------------------
# --- Helper: find_active_species ---
# ---------------------------------------------------------------------------
def find_active_species(row, player_prefix):
    """
    Finds the species of the active Pokemon for a given player prefix ('p1' or 'p2')
    in a DataFrame row containing slot information.
    """
    for i in range(1, 7):  # Check slots 1 to 6
        active_col = f"{player_prefix}_slot{i}_is_active"
        species_col = f"{player_prefix}_slot{i}_species"
        # Check if both columns exist in the row's index (safer than assuming they do)
        if active_col in row.index and species_col in row.index:
            # Check if the active flag is explicitly 1 (not NaN or other values)
            if row[active_col] == 1:
                # Return the species, handle potential None/NaN from species column itself
                return row[species_col] if pd.notna(row[species_col]) else 'Unknown'
    return 'Unknown'  # Return 'Unknown' if no active Pokemon found


# ---------------------------------------------------------------------------
# --- Core: build_medium_X ---
# ---------------------------------------------------------------------------
def build_medium_X(df,
                   disable_smogon_features=False,
                   smogon_top_n=100,
                   min_feature_replay_count=50,
                   blind_own_bench=False,
                   blind_opp_bench=False,
                   smogon_json_path='gen9ou-0.json'):
    """
    Builds the 'medium' feature DataFrame from a pre-filtered replay Parquet DataFrame.

    Assumes the caller has already:
    - Dropped NaN action_taken rows
    - Filtered for player_to_move == 'p1'
    - Filtered for move: actions only
    - The df still has a 'replay_id' column

    Returns:
        (X, numerical_features, categorical_features)
        X: pd.DataFrame of features
        numerical_features: list of numerical column names
        categorical_features: list of categorical column names
    """

    print("\n--- Building MEDIUM feature set (with Active Revealed Moves) ---")
    selected_columns = []
    base_active_features = ['species', 'hp_perc', 'status', 'boost_atk', 'boost_def',
                            'boost_spa', 'boost_spd', 'boost_spe', 'terastallized', 'tera_type']

    print("Identifying column patterns for non-move features...")
    bench_cols = []
    for i in range(1, 7):
        for player in ['p1', 'p2']:
            bench_cols.append(f'{player}_slot{i}_hp_perc')
            bench_cols.append(f'{player}_slot{i}_status')
            if blind_own_bench and player == 'p1':
                continue
            if blind_opp_bench and player == 'p2':
                continue
            bench_cols.append(f'{player}_slot{i}_species')
    selected_columns.extend(bench_cols)

    field_cols = ['field_weather', 'field_terrain', 'field_pseudo_weather']
    selected_columns.extend(field_cols)

    hazard_cols = []
    hazard_types = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
    for player in ['p1', 'p2']:
        for hazard in hazard_types:
            col_name = f'{player}_hazard_{hazard}'
            hazard_cols.append(col_name)
    selected_columns.extend(hazard_cols)

    side_cond_cols = []
    side_cond_types = ['reflect', 'lightscreen', 'auroraveil', 'tailwind']
    for player in ['p1', 'p2']:
        for cond in side_cond_types:
            col_name = f'{player}_side_{cond}'
            side_cond_cols.append(col_name)
    selected_columns.extend(side_cond_cols)

    context_cols = ['last_move_p1', 'last_move_p2', 'turn_number']
    selected_columns.extend(context_cols)

    print(f"Selecting {len(selected_columns)} base columns + preparing for active Pokemon info...")
    valid_selected_columns = [col for col in selected_columns if col in df.columns]
    print(f"  Found {len(valid_selected_columns)} direct columns.")
    missing_base_cols = set(selected_columns) - set(valid_selected_columns)
    if missing_base_cols:
        print(f"  Warning: Missing base columns: {missing_base_cols}")
    X_medium = df[valid_selected_columns].copy()

    print("  Extracting active Pokemon details (vectorized)...")
    p1_active_masks = [(df[f'p1_slot{i}_is_active'] == 1).values for i in range(1, 7)]
    p2_active_masks = [(df[f'p2_slot{i}_is_active'] == 1).values for i in range(1, 7)]

    active_features = {}
    for feat in base_active_features:
        default_val = 'Unknown' if feat in ['species', 'status', 'tera_type'] else np.nan

        # P1
        p1_feat_cols = [df[f'p1_slot{i}_{feat}'].values for i in range(1, 7)]
        active_features[f'p1_active_{feat}'] = np.select(p1_active_masks, p1_feat_cols, default=default_val)
        # P2
        p2_feat_cols = [df[f'p2_slot{i}_{feat}'].values for i in range(1, 7)]
        active_features[f'p2_active_{feat}'] = np.select(p2_active_masks, p2_feat_cols, default=default_val)

    # Extract revealed moves strings
    p1_moves_cols = [df[f'p1_slot{i}_revealed_moves'].values if f'p1_slot{i}_revealed_moves' in df.columns
                     else ['none'] * len(df) for i in range(1, 7)]
    active_features['p1_active_revealed_moves_str'] = np.select(p1_active_masks, p1_moves_cols, default='none')

    p2_moves_cols = [df[f'p2_slot{i}_revealed_moves'].values if f'p2_slot{i}_revealed_moves' in df.columns
                     else ['none'] * len(df) for i in range(1, 7)]
    active_features['p2_active_revealed_moves_str'] = np.select(p2_active_masks, p2_moves_cols, default='none')

    active_features['replay_id'] = df['replay_id'].values if 'replay_id' in df.columns else ['unknown'] * len(df)

    active_df = pd.DataFrame(active_features, index=df.index)
    print("  Active Pokemon details extracted via vectorized operations.")

    print("  Adding active Pokemon details to X DataFrame...")
    active_df['p1_active_revealed_moves_str'] = (
        active_df['p1_active_revealed_moves_str']
        .replace({None: 'none', 'None': 'none'}).fillna('none').astype(str)
    )
    active_df['p2_active_revealed_moves_str'] = (
        active_df['p2_active_revealed_moves_str']
        .replace({None: 'none', 'None': 'none'}).fillna('none').astype(str)
    )

    # Apply HP Binning
    print("  Applying HP Binning to active and bench slots...")
    hp_cols = [col for col in active_df.columns if col.endswith('_hp_perc')]
    hp_cols.extend([col for col in X_medium.columns if col.endswith('_hp_perc')])

    for col in hp_cols:
        if col in active_df.columns:
            active_df[col] = active_df[col].apply(bin_hp)
        if col in X_medium.columns:
            X_medium[col] = X_medium[col].apply(bin_hp)

    X = pd.concat([X_medium, active_df], axis=1)
    print(f"Medium X shape before move encoding: {X.shape}")
    del X_medium, active_df; gc.collect()

    # --- Multi-Hot Encoding of Revealed Moves ---
    print("\nProcessing ACTIVE 'revealed_moves' features (Multi-Hot Encoding)...")
    active_revealed_move_cols = ['p1_active_revealed_moves_str', 'p2_active_revealed_moves_str']
    active_all_revealed_moves = set()
    new_binary_move_cols = []

    print("  Finding unique revealed moves from ACTIVE slots...")
    for col in active_revealed_move_cols:
        if col in X.columns:
            unique_in_col = X[col].fillna('none').astype(str).str.split(',').explode().unique()
            active_all_revealed_moves.update(m for m in unique_in_col if m and m != 'none' and m != 'error_state')
        else:
            print(f"  Warning: Expected active moves column '{col}' not found.")

    if not active_all_revealed_moves:
        print("  Warning: No valid revealed moves found in active slots.")
    else:
        unique_moves_list = sorted(list(active_all_revealed_moves))
        print(f"  Found {len(unique_moves_list)} unique revealed moves across ACTIVE slots.")

        if min_feature_replay_count > 0:
            print(f"  Applying frequency pruning (min_feature_replay_count={min_feature_replay_count})...")
            combined_moves = pd.Series(dtype=str)
            for col in active_revealed_move_cols:
                if col in X.columns:
                    if combined_moves.empty:
                        combined_moves = X[col].fillna('none').astype(str)
                    else:
                        combined_moves = combined_moves + ',' + X[col].fillna('none').astype(str)

            temp_df = pd.DataFrame({'replay_id': X['replay_id'], 'move': combined_moves.str.split(',')})
            temp_exploded = temp_df.explode('move')
            temp_exploded['move'] = temp_exploded['move'].str.strip()
            temp_exploded = temp_exploded[~temp_exploded['move'].isin(['', 'none', 'error_state', 'None'])]
            move_replay_counts = temp_exploded.groupby('move')['replay_id'].nunique().to_dict()

            pruned_moves = [m for m in unique_moves_list if move_replay_counts.get(m, 0) >= min_feature_replay_count]
            dropped = len(unique_moves_list) - len(pruned_moves)
            print(f"    Pruned {dropped} niche moves. Kept {len(pruned_moves)} features.")
            unique_moves_list = sorted(pruned_moves)
            del temp_df, temp_exploded, combined_moves; gc.collect()

        print("  Creating and populating binary revealed move columns for active slots (vectorized)...")
        for base_col in active_revealed_move_cols:
            if base_col not in X.columns:
                continue
            player_prefix = base_col.split('_')[0]
            new_col_prefix = f"{player_prefix}_active_revealed_move"

            dummies = X[base_col].fillna('none').astype(str).str.get_dummies(sep=',')

            for move in unique_moves_list:
                if move in dummies.columns:
                    sanitized_move_name = sanitize_name(move)
                    new_col_name = f"{new_col_prefix}_{sanitized_move_name}"
                    X[new_col_name] = dummies[move].astype(np.int8)
                    new_binary_move_cols.append(new_col_name)
            del dummies; gc.collect()
        print(f"  Created {len(new_binary_move_cols)} new binary active move features.")

    # Drop temporary logic columns
    cols_to_drop = [col for col in active_revealed_move_cols if col in X.columns]
    if 'replay_id' in X.columns:
        cols_to_drop.append('replay_id')
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
    gc.collect()
    print(f"Medium X shape after move encoding: {X.shape}")

    # --- Smogon Usage Stats Injection ---
    new_usage_cols = []
    if not disable_smogon_features:
        if os.path.exists(smogon_json_path):
            print(f"\nInjecting Smogon Usage Stats (Top {smogon_top_n if smogon_top_n else 'All'} moves)...")
            smogon_df, smogon_moves = get_smogon_usages_df(smogon_json_path, top_n=smogon_top_n)
            if not smogon_df.empty:
                for player in ['p1', 'p2']:
                    if f'{player}_active_species' in X.columns:
                        clean_species = X[f'{player}_active_species'].fillna('Unknown').astype(str).apply(sanitize_name)
                        usage_suffix = smogon_df.add_prefix(f'{player}_active_usage_')
                        player_usages = (
                            clean_species.to_frame(name='species')
                            .join(usage_suffix, on='species')
                            .drop(columns=['species'])
                            .fillna(0.0)
                        )
                        player_usages = player_usages.astype(np.float32)
                        X = pd.concat([X, player_usages], axis=1)
                        new_usage_cols.extend(player_usages.columns.tolist())
                        del player_usages; del clean_species; gc.collect()
                print(f"  Added {len(new_usage_cols)} usage stat columns.")
            else:
                print("  Warning: Could not load Smogon usage stats.")
        else:
            print(f"  Warning: Smogon JSON not found at '{smogon_json_path}'. Skipping usage injection.")
    else:
        print("\nSkipping Smogon Usage Stats injection (disable_smogon_features=True).")

    # --- Final Feature Type Classification ---
    print("\nIdentifying final feature types for 'medium' set...")
    numerical_features = []
    categorical_features = []

    for player in ['p1', 'p2']:
        if f'{player}_active_species' in X.columns: categorical_features.append(f'{player}_active_species')
        if f'{player}_active_status' in X.columns: categorical_features.append(f'{player}_active_status')
        if f'{player}_active_tera_type' in X.columns: categorical_features.append(f'{player}_active_tera_type')
        if f'{player}_active_hp_perc' in X.columns: categorical_features.append(f'{player}_active_hp_perc')
        if f'{player}_active_boost_atk' in X.columns: numerical_features.append(f'{player}_active_boost_atk')
        if f'{player}_active_boost_def' in X.columns: numerical_features.append(f'{player}_active_boost_def')
        if f'{player}_active_boost_spa' in X.columns: numerical_features.append(f'{player}_active_boost_spa')
        if f'{player}_active_boost_spd' in X.columns: numerical_features.append(f'{player}_active_boost_spd')
        if f'{player}_active_boost_spe' in X.columns: numerical_features.append(f'{player}_active_boost_spe')
        if f'{player}_active_terastallized' in X.columns: numerical_features.append(f'{player}_active_terastallized')

    numerical_features.extend(new_binary_move_cols)
    numerical_features.extend(new_usage_cols)

    for i in range(1, 7):
        for player in ['p1', 'p2']:
            hp_col = f'{player}_slot{i}_hp_perc'
            status_col = f'{player}_slot{i}_status'
            species_col = f'{player}_slot{i}_species'
            if hp_col in X.columns: categorical_features.append(hp_col)
            if status_col in X.columns: categorical_features.append(status_col)
            if species_col in X.columns: categorical_features.append(species_col)

    categorical_features.extend([f for f in field_cols if f in X.columns])
    numerical_features.extend([f for f in hazard_cols if f in X.columns])
    numerical_features.extend([f for f in side_cond_cols if f in X.columns])
    if 'last_move_p1' in X.columns: categorical_features.append('last_move_p1')
    if 'last_move_p2' in X.columns: categorical_features.append('last_move_p2')
    if 'turn_number' in X.columns: numerical_features.append('turn_number')

    all_medium_cols = list(X.columns)
    numerical_features = sorted(list(set([f for f in numerical_features if f in all_medium_cols])))
    categorical_features = sorted(list(set([f for f in categorical_features if f in all_medium_cols])))

    overlap = set(numerical_features) & set(categorical_features)
    if overlap:
        print(f"Warning: Overlap detected: {overlap}. Removing from numerical.")
        numerical_features = [f for f in numerical_features if f not in overlap]

    print(f"  Numerical features: {len(numerical_features)}")
    print(f"  Categorical features: {len(categorical_features)}")

    return X, numerical_features, categorical_features
