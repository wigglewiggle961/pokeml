import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import lightgbm as lgb
import argparse
import os
import joblib
import warnings
import gc # Garbage collector
from tensorflow.keras.layers import BatchNormalization, Activation
import json
import re
from re import escape as re_escape

# Suppress TensorFlow/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore pandas 3.0 warnings for now

# --- Helper Function (Sanitization) ---
def sanitize_name(name):
    """
    Standardize a move or species name by removing all non-alphanumeric characters 
    and converting to lowercase. Ensures 100% parity with Smogon Usage Stats keys.
    """
    if not isinstance(name, str) or name.lower() == 'unknown':
        return 'unknown'
    return re.sub(r'[^a-z0-9]', '', name.lower())

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

# --- Helper Function (get_smogon_usages_df) ---
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
        
        # --- NEW: Pruning Logic ---
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

# --- Helper Function (find_active_species - unchanged) ---
def find_active_species(row, player_prefix):
    """
    Finds the species of the active Pokemon for a given player prefix ('p1' or 'p2')
    in a DataFrame row containing slot information.
    """
    for i in range(1, 7): # Check slots 1 to 6
        active_col = f"{player_prefix}_slot{i}_is_active"
        species_col = f"{player_prefix}_slot{i}_species"
        # Check if both columns exist in the row's index (safer than assuming they do)
        if active_col in row.index and species_col in row.index:
             # Check if the active flag is explicitly 1 (not NaN or other values)
            if row[active_col] == 1:
                # Return the species, handle potential None/NaN from species column itself
                return row[species_col] if pd.notna(row[species_col]) else 'Unknown'
    return 'Unknown' # Return 'Unknown' if no active Pokemon found

# --- TF Training Function (MODIFIED to accept suffix) ---
def train_tensorflow_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                      y_train_encoded, y_val_encoded, y_test_encoded, # INTEGER encoded y
                                      num_classes, class_weight_dict, label_encoder, # Pass num_classes, weights, encoder
                                      epochs=20, batch_size=128, learning_rate=0.001,
                                      label_suffix=""): # <--- MODIFIED: Added label_suffix
    print(f"\n--- Training TensorFlow Model ---")
    print(f"Input shape: {X_train_processed.shape[1]}")
    print(f"Num classes: {num_classes}")

    # Convert integer labels to one-hot encoding
    print("One-hot encoding target variable for TF...")
    try:
        y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
        y_val_one_hot = to_categorical(y_val_encoded, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)
        print("One-hot encoding complete.")
    except ValueError as e:
         print(f"Error during one-hot encoding: {e}")
         return None, None
    except MemoryError:
        print("MemoryError during one-hot encoding.")
        return None, None

    # Define the Model
    input_dim = X_train_processed.shape[1]
    print(f"Building TF model with input dimension: {input_dim}")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(128, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(64, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    # Use the label_smoothing parameter if provided
    smoothing = 0.1 if num_classes > 50 else 0.0 # Heuristic for large target sets
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
    model.summary()

    # Train the Model
    print("\nStarting TF model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_processed, y_train_one_hot,
        validation_data=(X_val_processed, y_val_one_hot),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=2
    )
    print("TF Training finished.")

    # Evaluate the Model
    print("\nEvaluating TF model on the test set...")
    results = model.evaluate(X_test_processed, y_test_one_hot, verbose=0)
    loss = results[0]
    accuracy = results[1]
    top_5_accuracy = results[2] if len(results) > 2 else np.nan
    print(f"TF Test Loss: {loss:.4f}")
    print(f"TF Test Accuracy: {accuracy:.4f}")
    print(f"TF Test Top-5 Accuracy: {top_5_accuracy:.4f}")

    # Save Model
    # <--- MODIFIED: Use label_suffix in filename --->
    model_save_path = f'models/action_tf_model_v4_{label_suffix}.keras'
    print(f"Saving TF model to {model_save_path}")
    try:
        model.save(model_save_path)
        print("TF Model saved.")
    except Exception as e:
        print(f"Error saving TF model: {e}")

    return history, model


# --- Stability Helper ---
def lgbm_checkpoint_callback(period, model_path):
    """Custom callback to save LGBM model every N iterations."""
    def _callback(env):
        if (env.iteration + 1) % period == 0:
            env.model.save_model(model_path)
            # print(f"  [Checkpoint] Saved to {model_path} (Iteration {env.iteration + 1})")
    return _callback

# --- LGBM Training Function (MODIFIED to accept suffix) ---
def train_lgbm_action_predictor(X_train, X_val, X_test,
                                y_train_encoded, y_val_encoded, y_test_encoded, # Use INTEGER encoded y
                                numerical_features, categorical_features,
                                num_classes, class_weight_dict, label_encoder, # Pass num_classes, weights, encoder
                                label_suffix="",
                                lgbm_estimators=300, lgbm_lr=0.05, label_smoothing=0.0, use_gpu=False): # <--- MODIFIED: Added label_smoothing & use_gpu
    print(f"\n--- Training LightGBM Model ---")
    print(f"Using {len(numerical_features)} numerical and {len(categorical_features)} categorical features for LGBM.")

    # --- Preprocessing Specific to LGBM ---
    print("Converting categorical features to 'category' dtype for LGBM...")
    category_map = {}
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    X_test_lgbm = X_test.copy()
    active_categorical_features = []
    for col in categorical_features:
        if col in X_train_lgbm.columns:
            all_categories = pd.concat([
                X_train_lgbm[col].astype(str),
                X_val_lgbm[col].astype(str),
                X_test_lgbm[col].astype(str)
            ]).unique()
            
            # --- NEW GPU CATEGORICAL FIX ---
            # GPU only supports < 255 categories per feature. 
            if use_gpu and len(all_categories) > 250:
                print(f"  Warning: Feature '{col}' has {len(all_categories)} categories. GPU limit is 255.")
                print(f"           Treating '{col}' as numerical for GPU compatibility.")
                # We do NOT add it to active_categorical_features, 
                # but we still ensure it is encoded as a number.
                X_train_lgbm[col] = X_train_lgbm[col].astype(str).map({val: i for i, val in enumerate(all_categories)}).fillna(-1)
                X_val_lgbm[col] = X_val_lgbm[col].astype(str).map({val: i for i, val in enumerate(all_categories)}).fillna(-1)
                X_test_lgbm[col] = X_test_lgbm[col].astype(str).map({val: i for i, val in enumerate(all_categories)}).fillna(-1)
                continue # Skip marking as 'category' dtype
            
            active_categorical_features.append(col)
            cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)
            X_train_lgbm[col] = X_train_lgbm[col].astype(str).astype(cat_type)
            X_val_lgbm[col] = X_val_lgbm[col].astype(str).astype(cat_type)
            X_test_lgbm[col] = X_test_lgbm[col].astype(str).astype(cat_type)
            category_map[col] = cat_type
        else:
             print(f"Warning: Categorical feature '{col}' not found in training data columns for LGBM.")

    # Scale Numerical Features
    # NOTE: LightGBM is tree-based and does NOT need scaling.
    # We skip scaling to save memory but still save a scaler for compatibility.
    scaler = None
    active_numerical_features = []
    if numerical_features:
        active_numerical_features = [f for f in numerical_features if f in X_train_lgbm.columns]
        print(f"Skipping numerical scaling for LGBM (trees don't need it). {len(active_numerical_features)} numerical features identified.")
        # Fit a scaler on a tiny sample just for saving (in case inference code expects it)
        if active_numerical_features:
            scaler = StandardScaler()
            scaler.fit(X_train_lgbm[active_numerical_features].head(100))
    else:
        print("No numerical features defined.")

    # Prepare LGBM Datasets
    final_feature_names = active_numerical_features + active_categorical_features
    X_train_lgbm = X_train_lgbm[final_feature_names]
    X_val_lgbm = X_val_lgbm[final_feature_names]
    X_test_lgbm = X_test_lgbm[final_feature_names]
    print("Creating LGBM datasets...")
    lgb_train = lgb.Dataset(X_train_lgbm, label=y_train_encoded,
                            categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                            feature_name=final_feature_names,
                            free_raw_data=True, # <--- Task 1: Enable native memory offloading
                            params={'max_bin': 63 if use_gpu else 511})
    lgb_eval = lgb.Dataset(X_val_lgbm, label=y_val_encoded, reference=lgb_train,
                           categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                           feature_name=final_feature_names,
                           free_raw_data=True, # <--- Task 1: Enable native memory offloading
                           params={'max_bin': 63 if use_gpu else 511})

    # Class Weights for LGBM
    sample_weight = None
    if class_weight_dict:
        print("Calculating sample weights for LGBM...")
        try:
             sample_weight = np.array([class_weight_dict.get(cls_idx, 1.0) for cls_idx in y_train_encoded])
             print(f"Sample weights calculated (min: {np.min(sample_weight):.2f}, max: {np.max(sample_weight):.2f}).")
             lgb_train.set_weight(sample_weight)
             print("Applied sample weights to training dataset.")
        except Exception as e:
             print(f"Warning: Could not compute/apply sample weights for LGBM: {e}.")
             sample_weight = None

    # --- Task 3: Free RAM before training begins ---
    print("Pre-training RAM cleanup (freeing original copies)...")
    del X_train_lgbm; del X_val_lgbm; gc.collect()

    # Define LGBM Parameters
    params = {
        'objective': 'multiclass', 
        'metric': ['multi_logloss', 'multi_error'],
        'device': 'gpu' if use_gpu else 'cpu', # <--- NEW: GPU toggle
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_bin': 63 if use_gpu else 511, # <--- NEW: Fix 'bin size cannot run on GPU' error
        'num_class': num_classes, 
        'boosting_type': 'gbdt', 
        'n_estimators': lgbm_estimators,
        'learning_rate': lgbm_lr,
        'label_smoothing': label_smoothing, # <--- NEW: Regularize multiclass targets
        'num_leaves': 24,              # Controls complexity (2^max_depth is 32)
        'max_depth': 5,                # Explicit depth limit
        'min_data_in_leaf': 5000,      # Overfitting prevention (Prevents tiny leaves)
        'reg_alpha': 20.0,             # Regularization L1
        'reg_lambda': 20.0,            # Regularization L2
        'cat_smooth': 500,             # Smoothing for species/moves
        'cat_l2': 50,                  # Regularize categorical features
        'colsample_bytree': 0.15,      # Healthy balance for feature sampling
        'subsample': 0.7,              
        'subsample_freq': 1,           
        'seed': 42, 
        'n_jobs': -1, 
        'verbose': -1,
    }

    # Train LightGBM Model
    print("Starting LGBM model training...")
    evals_result = {}
    
    # <--- NEW: Checkpoint path --->
    checkpoint_path = f'models/action_lgbm_checkpoint_v4_{label_suffix}.txt'
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result),
        lgbm_checkpoint_callback(250, checkpoint_path) # <--- NEW: Auto-save every 250 rounds
    ]
    lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval],
                           valid_names=['train', 'eval'], callbacks=callbacks)
    print("LGBM Training finished.")

    # Evaluate LightGBM Model
    print("\nEvaluating LGBM model on the test set...")
    y_pred_proba = lgbm_model.predict(X_test_lgbm, num_iteration=lgbm_model.best_iteration)
    y_pred_indices = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test_encoded, y_pred_indices)
    try:
        top_5_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=5, labels=np.arange(num_classes))
    except ValueError:
        print("Warning: Cannot calculate top-5 accuracy.")
        top_5_accuracy = np.nan
    print(f"LGBM Test Accuracy: {accuracy:.4f}")
    print(f"LGBM Test Top-5 Accuracy: {top_5_accuracy:.4f}")
    
    # Feature Importance
    print("\n--- Feature Importance (Top 20) ---")
    importance = lgbm_model.feature_importance(importance_type='gain')
    feature_names = np.array(final_feature_names)
    sorted_idx = importance.argsort()[::-1]
    for i in range(min(20, len(feature_names))):
        print(f"{i+1}. {feature_names[sorted_idx[i]]}: {importance[sorted_idx[i]]:.4f}")
    print("-----------------------------------")

    # Save Model and Feature Info
    # <--- MODIFIED: Use label_suffix in filenames --->
    model_save_path = f'models/action_lgbm_model_v4_{label_suffix}.txt'
    print(f"Saving LGBM model to {model_save_path}")
    try:
        lgbm_model.save_model(model_save_path)
        print("LGBM Model saved.")
    except Exception as e:
        print(f"Error saving LGBM model: {e}")

    lgbm_info_path = f'models/action_lgbm_feature_info_v4_{label_suffix}.joblib'
    print(f"Saving LGBM feature info to {lgbm_info_path}")
    try:
        lgbm_info = {
            'numerical_features': active_numerical_features,
            'categorical_features': active_categorical_features,
            'feature_names_in_order': final_feature_names,
            'category_map': category_map
        }
        joblib.dump(lgbm_info, lgbm_info_path)
        print(f"LGBM feature info saved.")
    except Exception as e:
         print(f"Error saving LGBM feature info: {e}")

    if scaler and active_numerical_features:
        scaler_path = f'models/action_lgbm_scaler_v4_{label_suffix}.joblib'
        print(f"Saving LGBM scaler to {scaler_path}")
        try:
             joblib.dump(scaler, scaler_path)
             print(f"LGBM scaler saved.")
        except Exception as e:
             print(f"Error saving LGBM scaler: {e}")

    return lgbm_model


# --- MODIFIED Main execution function ---
def run_action_training(parquet_path, model_type='tensorflow', feature_set='full',
                        min_turn=0, min_move_count=0,
                        min_feature_replay_count=50, # NEW: filter rare input features
                        label_smoothing=0.0,         # NEW: regularize noisy targets
                        test_split_size=0.2, val_split_size=0.15,
                        epochs=30, batch_size=256, learning_rate=0.001,
                        disable_smogon_features=False, smogon_top_n=100, # <--- NEW
                        lgbm_estimators=300, lgbm_lr=0.05,
                        use_gpu=False, blind_own_bench=False, blind_opp_bench=False):
    """Loads data, splits, preprocesses based on feature_set, and trains action predictor."""

    print(f"--- Starting Action Predictor Training (V4) ---")
    print(f"Model type: {model_type.upper()}")
    print(f"Feature Set: {feature_set.upper()}")
    print(f"Loading data from: {parquet_path}")
    try:
        if os.path.isdir(parquet_path):
            import glob
            files = glob.glob(os.path.join(parquet_path, "*.parquet"))
            if not files:
                files = glob.glob(os.path.join(parquet_path, "**", "*.parquet"), recursive=True)
            if not files:
                raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
            print(f"  Found {len(files)} parquet files. Loading and downcasting...")
            dfs = []
            for f in files:
                temp_df = pd.read_parquet(f)
                # Task 2: Immediate Downcasting for RAM efficiency (Crucial for 16GB/3M rows)
                for col in temp_df.select_dtypes(include=['float64']).columns:
                    temp_df[col] = temp_df[col].astype(np.float32)
                for col in temp_df.select_dtypes(include=['int64']).columns:
                    temp_df[col] = temp_df[col].astype(np.int32)
                dfs.append(temp_df)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(parquet_path)
            # Downcast single file
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype(np.float32)
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = df[col].astype(np.int32)
        
        print(f"Data loaded and downcasted. Shape: {df.shape}")
        gc.collect()
    except Exception as e:
        print(f"Error loading Parquet: {e}"); return

    # --- Filter Data ---
    print("\nFiltering data...")
    original_rows = len(df)
    df = df.dropna(subset=['action_taken']) # Essential target
    print(f"Rows after dropping NaN action_taken: {len(df)}")

    # Filter based on player *before* potentially removing player_to_move column
    if feature_set in ['simplified', 'medium']:
        print(f"Filtering for player_to_move == 'p1' (for {feature_set} set)...")
        rows_before_p1_filter = len(df)
        df = df[df['player_to_move'] == 'p1'].copy()
        print(f"Rows after filtering for p1's move: {len(df)} (Removed {rows_before_p1_filter - len(df)})")
        if df.empty: print(f"Error: No data found for player p1's moves (feature_set={feature_set})."); return
    else: # full set uses player_to_move as a feature potentially
        print("Keeping data for both players ('full' feature set).")

    df = df[df['action_taken'].str.startswith('move')].copy()
    # <--- Process Target Variables --->
    rows_before_move_filter = len(df)
    df = df[df['action_taken'].astype(str).str.startswith('move:')].copy()
    if df.empty:
        print("Error: No 'move:' actions found. Cannot train model.")
        return
    print(f"Rows after filtering for move actions: {len(df)} (Removed {rows_before_move_filter - len(df)})")
    
    # Extract only the move name as the target
    y_raw = df['action_taken'].str.replace('move:', '', regex=False)
    print("Target variable 'y_raw' now contains only move names.")

    # <--- NEW: Filter rare moves/actions based on min_move_count --->
    if min_move_count > 0:
        print(f"\nFiltering rare actions with fewer than {min_move_count} occurrences...")
        action_counts = y_raw.value_counts()
        common_actions = action_counts[action_counts >= min_move_count].index
        rare_actions = action_counts[action_counts < min_move_count]
        print(f"  Total unique actions: {len(action_counts)}")
        print(f"  Actions with >= {min_move_count} samples: {len(common_actions)}")
        print(f"  Rare actions removed: {len(rare_actions)}")
        if len(rare_actions) > 0:
            print(f"  Examples of removed rare actions: {list(rare_actions.head(10).index)}")
        
        # Filter to keep only common actions
        rows_before = len(df)
        mask = y_raw.isin(common_actions)
        df = df[mask].copy()
        y_raw = y_raw[mask].copy()
        print(f"  Rows after filtering: {len(df)} (removed {rows_before - len(df)} rows)")
        
        if df.empty:
            print("Error: No data remaining after filtering rare actions.")
            return
    # <--- End of rare action filtering --->


    # Filter based on turn number (applied to potentially filtered df)
    if min_turn > 0:
        initial_rows_turn_filter = len(df)
        df = df[df['turn_number'] >= min_turn].copy()
        print(f"Rows after filtering turns >= {min_turn}: {len(df)} (Removed {initial_rows_turn_filter - len(df)})")
        if df.empty: print("Error: No data remaining after turn filtering."); return

    # --- Conditional Feature Selection and Target Prep ---
    X = None
    # y_raw defined above based on predict_mode
    numerical_features = []
    categorical_features = []
    selected_columns = []
    # <--- MODIFIED: Create a unique suffix for saved files --->
    label_encoder_suffix = f"{feature_set}"

    # --- Feature Set Logic (simplified, medium, full) ---
    # NOTE: The 'medium' block below is the one modified in the previous step
    #       to include active revealed moves. Keep that version.
    #       The 'full' and 'simplified' blocks remain as they were in V4.

    if feature_set == 'simplified':
        # (Keep V4 'simplified' logic - only p1_active_species, p2_active_species)
        print("\n--- Using SIMPLIFIED feature set (active species only) ---")
        print("Extracting active species for P1 and P2...")
        p1_active_series = df.apply(lambda row: find_active_species(row, 'p1'), axis=1)
        p2_active_series = df.apply(lambda row: find_active_species(row, 'p2'), axis=1)
        X = pd.DataFrame({
            'p1_active_species': p1_active_series,
            'p2_active_species': p2_active_series
        }, index=df.index)
        print(f"Simplified X shape: {X.shape}")
        numerical_features = []
        categorical_features = ['p1_active_species', 'p2_active_species']


    elif feature_set == 'medium':
        print("\n--- Using MEDIUM feature set (with Active Revealed Moves) ---")
        selected_columns = []
        base_active_features = ['species', 'hp_perc', 'status', 'boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe', 'terastallized', 'tera_type']
        print("Identifying column patterns for non-move features...")
        bench_cols = []
        for i in range(1, 7):
            for player in ['p1', 'p2']:
                 bench_cols.append(f'{player}_slot{i}_hp_perc')
                 bench_cols.append(f'{player}_slot{i}_status')
                 if blind_own_bench and player == 'p1':
                     continue # Blind the model to its own bench species
                 if blind_opp_bench and player == 'p2':
                     continue # Blind the model to its opponent's bench species
                 bench_cols.append(f'{player}_slot{i}_species')
        selected_columns.extend(bench_cols)
        field_cols = ['field_weather', 'field_terrain', 'field_pseudo_weather']
        selected_columns.extend(field_cols)
        hazard_cols = []
        # NOTE: Column names match process_replays.py output (no underscores in compound words)
        hazard_types = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
        for player in ['p1', 'p2']:
             for hazard in hazard_types:
                  col_name = f'{player}_hazard_{hazard}'
                  hazard_cols.append(col_name)
        selected_columns.extend(hazard_cols)
        side_cond_cols = []
        # NOTE: Column names match process_replays.py output (no underscores in compound words)
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
        if missing_base_cols: print(f"  Warning: Missing base columns: {missing_base_cols}")
        X_medium = df[valid_selected_columns].copy()

        print("  Extracting active Pokemon details (vectorized)...")
        # --- VECTORIZED REPLACEMENT FOR SLOW ITERROWS LOOP ---
        p1_active_masks = [(df[f'p1_slot{i}_is_active'] == 1).values for i in range(1, 7)]
        p2_active_masks = [(df[f'p2_slot{i}_is_active'] == 1).values for i in range(1, 7)]

        active_features = {}
        for feat in base_active_features:
            # Prevent numeric columns from collapsing to 'object' dtype by explicitly using np.nan
            default_val = 'Unknown' if feat in ['species', 'status', 'tera_type'] else np.nan
            
            # P1
            p1_feat_cols = [df[f'p1_slot{i}_{feat}'].values for i in range(1, 7)]
            active_features[f'p1_active_{feat}'] = np.select(p1_active_masks, p1_feat_cols, default=default_val)
            # P2
            p2_feat_cols = [df[f'p2_slot{i}_{feat}'].values for i in range(1, 7)]
            active_features[f'p2_active_{feat}'] = np.select(p2_active_masks, p2_feat_cols, default=default_val)

        # Extract revealed moves strings
        p1_moves_cols = [df[f'p1_slot{i}_revealed_moves'].values if f'p1_slot{i}_revealed_moves' in df.columns else ['none']*len(df) for i in range(1, 7)]
        active_features['p1_active_revealed_moves_str'] = np.select(p1_active_masks, p1_moves_cols, default='none')
        
        p2_moves_cols = [df[f'p2_slot{i}_revealed_moves'].values if f'p2_slot{i}_revealed_moves' in df.columns else ['none']*len(df) for i in range(1, 7)]
        active_features['p2_active_revealed_moves_str'] = np.select(p2_active_masks, p2_moves_cols, default='none')
        
        active_features['replay_id'] = df['replay_id'].values if 'replay_id' in df.columns else ['unknown']*len(df)

        active_df = pd.DataFrame(active_features, index=df.index)
        print("  Active Pokemon details extracted via vectorized operations.")

        print("  Adding active Pokemon details to X DataFrame...")
        # Fill missing string columns (not handled well by np.select with 'default')
        active_df['p1_active_revealed_moves_str'] = active_df['p1_active_revealed_moves_str'].replace({None: 'none', 'None': 'none'}).fillna('none').astype(str)
        active_df['p2_active_revealed_moves_str'] = active_df['p2_active_revealed_moves_str'].replace({None: 'none', 'None': 'none'}).fillna('none').astype(str)
        
        # --- NEW: Apply HP Binning ---
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

        print("\nProcessing ACTIVE 'revealed_moves' features (Multi-Hot Encoding)...")
        active_revealed_move_cols = ['p1_active_revealed_moves_str', 'p2_active_revealed_moves_str']
        active_all_revealed_moves = set()
        new_binary_move_cols = []
        print("  Finding unique revealed moves from ACTIVE slots...")
        for col in active_revealed_move_cols:
            if col in X.columns:
                unique_in_col = X[col].fillna('none').astype(str).str.split(',').explode().unique()
                active_all_revealed_moves.update(m for m in unique_in_col if m and m != 'none' and m != 'error_state')
            else: print(f"  Warning: Expected active moves column '{col}' not found.")
        if not active_all_revealed_moves:
             print("  Warning: No valid revealed moves found in active slots.")
        else:
            unique_moves_list = sorted(list(active_all_revealed_moves))
            print(f"  Found {len(unique_moves_list)} unique revealed moves across ACTIVE slots.")
            
            if min_feature_replay_count > 0:
                print(f"  Applying frequency pruning (min_feature_replay_count={min_feature_replay_count})...")
                # Vectorized frequency pruning
                combined_moves = pd.Series(dtype=str)
                for col in active_revealed_move_cols:
                    if col in X.columns:
                        if combined_moves.empty:
                            combined_moves = X[col].fillna('none').astype(str)
                        else:
                            combined_moves = combined_moves + ',' + X[col].fillna('none').astype(str)
                
                # Explode moves and map to replay_ids
                temp_df = pd.DataFrame({'replay_id': X['replay_id'], 'move': combined_moves.str.split(',')})
                temp_exploded = temp_df.explode('move')
                temp_exploded['move'] = temp_exploded['move'].str.strip()
                
                # Exclude invalid parsed names
                temp_exploded = temp_exploded[~temp_exploded['move'].isin(['', 'none', 'error_state', 'None'])]
                
                # Compute unique replays per move
                move_replay_counts = temp_exploded.groupby('move')['replay_id'].nunique().to_dict()
                
                pruned_moves = [m for m in unique_moves_list if move_replay_counts.get(m, 0) >= min_feature_replay_count]
                dropped = len(unique_moves_list) - len(pruned_moves)
                print(f"    Pruned {dropped} niche moves. Kept {len(pruned_moves)} features.")
                unique_moves_list = sorted(pruned_moves)
                del temp_df, temp_exploded, combined_moves; gc.collect()

            print("  Creating and populating binary revealed move columns for active slots (vectorized)...")
            for base_col in active_revealed_move_cols:
                if base_col not in X.columns: continue
                player_prefix = base_col.split('_')[0]
                new_col_prefix = f"{player_prefix}_active_revealed_move"
                
                # str.get_dummies is optimized for this exact use case
                dummies = X[base_col].fillna('none').astype(str).str.get_dummies(sep=',')
                
                # Filter dummies to only include our pruned unique_moves_list
                # And sanitize column names
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
        if 'replay_id' in X.columns: cols_to_drop.append('replay_id')
        if cols_to_drop: X = X.drop(columns=cols_to_drop)
        gc.collect()
        print(f"Medium X shape after move encoding: {X.shape}")

        new_usage_cols = []
        if not disable_smogon_features:
            print(f"\nInjecting Smogon Usage Stats (Top {smogon_top_n if smogon_top_n else 'All'} moves)...")
            smogon_df, smogon_moves = get_smogon_usages_df('data/gen9ou-0.json', top_n=smogon_top_n)
            if not smogon_df.empty:
                # Standardize dataset labels for the parity check
                dataset_actions_raw = y_raw.unique()
                def std_move(m): return sanitize_name(m)
                dataset_actions_std = {std_move(m) for m in dataset_actions_raw}
                
                overlap = dataset_actions_std.intersection(set(smogon_moves))
                print(f"  Sanitization Parity Check: {len(overlap)}/{len(dataset_actions_std)} dataset moves conceptually matched Smogon entries.")
                
                if len(overlap) < len(dataset_actions_std) * 0.5:
                    missing = list(dataset_actions_std - set(smogon_moves))[:10]
                    print(f"  Warning: Low parity! Check your labels. Missing first 10 std: {missing}")
                
                for player in ['p1', 'p2']:
                    species_col = getattr(X, 'columns', []) 
                    if f'{player}_active_species' in X.columns:
                        clean_species = X[f'{player}_active_species'].fillna('Unknown').astype(str).apply(sanitize_name)
                        usage_suffix = smogon_df.add_prefix(f'{player}_active_usage_')
                        player_usages = clean_species.to_frame(name='species').join(usage_suffix, on='species').drop(columns=['species']).fillna(0.0)
                        
                        # --- Task 2: Downcast Smogon features to Float32 immediately ---
                        player_usages = player_usages.astype(np.float32)
                        
                        X = pd.concat([X, player_usages], axis=1)
                        new_usage_cols.extend(player_usages.columns.tolist())
                        
                         # --- Task 3: Intermediate cleanup ---
                        del player_usages; del clean_species; gc.collect()
                print(f"  Added {len(new_usage_cols)} usage stat columns.")
            else:
                print("  Warning: Could not load Smogon usage stats.")
        else:
            print("\nSkipping Smogon Usage Stats injection due to --disable_smogon_features flag.")
        # --- End NEW CODE ---

        print("\nIdentifying final feature types for 'medium' set...")
        numerical_features = []
        categorical_features = []
        # ( ... Same final feature type identification as provided in previous answer ... )
        # --- Start feature types ---
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
                 hp_col, status_col, species_col = f'{player}_slot{i}_hp_perc', f'{player}_slot{i}_status', f'{player}_slot{i}_species'
                 if hp_col in X.columns: categorical_features.append(hp_col)
                 if status_col in X.columns: categorical_features.append(status_col)
                 if species_col in X.columns: categorical_features.append(species_col)
        categorical_features.extend([f for f in field_cols if f in X.columns])
        numerical_features.extend([f for f in hazard_cols if f in X.columns])
        numerical_features.extend([f for f in side_cond_cols if f in X.columns])
        if 'last_move_p1' in X.columns: categorical_features.append('last_move_p1')
        if 'last_move_p2' in X.columns: categorical_features.append('last_move_p2')
        if 'turn_number' in X.columns: numerical_features.append('turn_number')
        # Dropped predicted_winner and battle_winner to prevent any potential leaks
        all_medium_cols = list(X.columns)
        numerical_features = sorted(list(set([f for f in numerical_features if f in all_medium_cols])))
        categorical_features = sorted(list(set([f for f in categorical_features if f in all_medium_cols])))
        overlap = set(numerical_features) & set(categorical_features)
        if overlap:
            print(f"Warning: Overlap detected: {overlap}. Removing from numerical.")
            numerical_features = [f for f in numerical_features if f not in overlap]
        # --- End feature types ---


    elif feature_set == 'full':
        # (Keep V4 'full' logic - multi-hot encodes all revealed_moves columns)
        print("\n--- Using FULL feature set ---")
        print("\nPreparing features and target...")
        base_exclude = ['replay_id', 'action_taken', 'battle_winner']
        if 'player_to_move' in df.columns: base_exclude.append('player_to_move') # Exclude if present
        cols_to_exclude = base_exclude
        feature_columns = [col for col in df.columns if col not in cols_to_exclude]
        X = df[feature_columns].copy()
        print(f"Initial feature count: {len(feature_columns)}")

        print("\nProcessing 'revealed_moves' features (Multi-Hot Encoding - ALL SLOTS)...")
        revealed_move_cols = sorted([col for col in X.columns if col.endswith('_revealed_moves')])
        all_revealed_moves = set()
        new_binary_move_cols = []

        if not revealed_move_cols:
            print("Warning: No '*_revealed_moves' columns found.")
        else:
            print("  Finding unique revealed moves...")
            for col in revealed_move_cols:
                unique_in_col = X[col].fillna('').astype(str).str.split(',').explode().unique()
                all_revealed_moves.update(m for m in unique_in_col if m and m != 'none' and m != 'error_state')
            unique_moves_list = sorted(list(all_revealed_moves))
            print(f"  Found {len(unique_moves_list)} unique revealed moves across all slots.")
            print("  Creating and populating binary revealed move columns...")
            for base_col in revealed_move_cols:
                X[base_col] = X[base_col].fillna('none')
                try:
                    revealed_sets = X[base_col].str.split(',').apply(set)
                    for move in unique_moves_list:
                        # Sanitize move name
                        sanitized_move_name = move.replace(' ', '_').replace('-', '_').replace(':', '').replace('%', 'perc')
                        new_col_name = f"{base_col}_{sanitized_move_name}"
                        X[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
                        new_binary_move_cols.append(new_col_name)
                    del revealed_sets
                except Exception as e:
                    print(f"  Error processing revealed moves in column {base_col}: {e}")
                gc.collect()
            print(f"  Created {len(new_binary_move_cols)} new binary move features.")
            print("  Dropping original revealed_moves string columns...")
            X = X.drop(columns=revealed_move_cols)
            gc.collect()

        print("\nIdentifying final feature types and handling remaining NaNs for 'full' set...")
        if 'last_move_p1' in X.columns: X['last_move_p1'] = X['last_move_p1'].fillna('none').astype('category')
        if 'last_move_p2' in X.columns: X['last_move_p2'] = X['last_move_p2'].fillna('none').astype('category')
        obj_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols: X[col] = X[col].fillna('Unknown').astype('category') # Use Unknown consistently

        # Add new binary move cols to numerical
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        numerical_features.extend(new_binary_move_cols)
        numerical_features = sorted(list(set(numerical_features))) # Ensure unique and sorted

        categorical_features = sorted(list(set(X.select_dtypes(include=['category']).columns.tolist())))

        overlap = set(numerical_features) & set(categorical_features)
        if overlap: numerical_features = [f for f in numerical_features if f not in overlap]

        if numerical_features:
             nan_counts = X[numerical_features].isnull().sum()
             cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
             if cols_with_nan:
                 print(f"  Numerical columns have NaNs: {cols_with_nan}. Filling with median.")
                 for col in cols_with_nan:
                      median_val = X[col].median()
                      fill_value = median_val if pd.notna(median_val) else 0
                      # Use direct assignment which is less prone to SettingWithCopyWarning
                      X[col] = X[col].fillna(fill_value)

        if X.isnull().sum().sum() > 0:
            print("Warning: NaNs still present after handling. Forcing fill.")
            for col in X.columns:
                 if X[col].isnull().any():
                      if pd.api.types.is_numeric_dtype(X[col]):
                          X[col] = X[col].fillna(0) # Use assignment
                      else:
                          # Ensure categorical NaNs are filled correctly
                          if pd.api.types.is_categorical_dtype(X[col]):
                              if 'Unknown' not in X[col].cat.categories:
                                   X[col] = X[col].cat.add_categories(['Unknown'])
                              X[col] = X[col].fillna('Unknown') # Use assignment
                          else: # If somehow still object/other
                              X[col] = X[col].fillna('Unknown') # Use assignment


    else:
        print(f"Error: Invalid feature_set '{feature_set}'. Choose 'full', 'medium', or 'simplified'.")
        return

    # --- Fill NaNs (General Check - applied AFTER feature set specific handling) ---
    # Note: This section might be slightly redundant now with the improved NaN handling
    # within the feature set blocks, but can serve as a final check.
    print(f"\nFinal NaN Check for '{feature_set}' set...")
    nan_report_before = X.isnull().sum()
    cols_with_nan_before = nan_report_before[nan_report_before > 0]
    if not cols_with_nan_before.empty:
        print(f"  NaNs found BEFORE final handling in columns: {cols_with_nan_before.index.tolist()}")
        # Re-identify types just in case columns were added/changed
        final_numerical = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        final_categorical = [col for col in X.columns if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object']

        for col in final_numerical:
            if X[col].isnull().any():
                 median_val = X[col].median()
                 fill_value = median_val if pd.notna(median_val) else 0
                 X[col] = X[col].fillna(fill_value) # Use assignment

        default_cat_fill = 'Unknown'
        for col in final_categorical:
             if X[col].isnull().any():
                  # Ensure 'Unknown' category exists if categorical
                  if pd.api.types.is_categorical_dtype(X[col]):
                      if default_cat_fill not in X[col].cat.categories:
                           X[col] = X[col].cat.add_categories([default_cat_fill])
                      X[col] = X[col].fillna(default_cat_fill) # Use assignment
                  else: # Object type
                       X[col] = X[col].fillna(default_cat_fill).astype('category') # Fill and convert
        print("  Final NaN handling complete.")
    else:
        print("  No NaNs found before final encoding/scaling.")

    # Final Check
    nan_report_after = X.isnull().sum()
    if nan_report_after.sum() > 0:
        print("Error: NaNs still present AFTER final handling. Columns:")
        print(nan_report_after[nan_report_after > 0])
        return


    # --- Common Steps from here ---

    print(f"\nFinal feature counts for '{feature_set}' set:")
    # Use the feature lists determined by the specific feature_set logic
    print(f"  Numerical: {len(numerical_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Total Features in X: {X.shape[1]}")
    
    # --- Task 2: Global Numeric Downcasting ---
    print("\nDowncasting numerical features to float32 for RAM safety...")
    num_cols = X.select_dtypes(include=[np.float64, np.int64]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].astype(np.float32)
        print(f"  Downcasted {len(num_cols)} columns to float32.")
    gc.collect()


    # --- Encode Target Variable (y) ---
    print(f"\nEncoding target variable (moves)...")
    label_encoder = LabelEncoder()
    try:
        # Ensure y_raw (which depends on predict_mode) is string
        y_encoded = label_encoder.fit_transform(y_raw.astype(str))
    except Exception as e:
        print(f"Error encoding target: {e}. Check target content.")
        return
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} unique actions/moves in the target set.")
    if num_classes < 2: print("Error: Need at least 2 unique actions/moves."); return

    # <--- MODIFIED: Use label_encoder_suffix in filename --->
    label_encoder_path = f'models/action_label_encoder_v4_{label_encoder_suffix}.joblib'
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")

    # <--- NEW: Sample and save 10 lines of preprocessed data --->
    print("\nSaving a 10-row sample of preprocessed data (X and y) for inspection...")
    if len(X) >= 10:
        sample_idx = np.random.choice(len(X), size=10, replace=False)
        X_sample = X.iloc[sample_idx].copy()
        X_sample['TARGET_RAW'] = label_encoder.inverse_transform(y_encoded[sample_idx])
        X_sample['TARGET_ENCODED'] = y_encoded[sample_idx]
        
        sample_csv_path = "preprocessed_sample.csv"
        X_sample.to_csv(sample_csv_path, index=False)
        print(f"Saved {sample_csv_path} with 10 random preprocessed rows.")
        
        print("\n--- 10 Random Samples of Preprocessed Features ---")
        cols_to_print = ['TARGET_RAW', 'TARGET_ENCODED'] + [c for c in X_sample.columns if 'species' in c or 'hp' in c or 'move' in c][:10]
        print(X_sample[cols_to_print].to_string())
    # <--- End NEW --->

    del y_raw; gc.collect()


    # <--- MODIFIED: Use GroupShuffleSplit based on replay_id to prevent leakage --->
    from sklearn.model_selection import GroupShuffleSplit
    
    # Extract groups (replay_id) before it's lost
    # NOTE: We assume df matches X in row count (filtered together)
    # Check if replay_id is available
    if 'replay_id' not in df.columns:
        print("Error: 'replay_id' column missing for group splitting.")
        return

    groups = df['replay_id']
    
    # Verify alignment
    if len(groups) != len(X):
        print(f"Error: Group length ({len(groups)}) != X length ({len(X)}). Row mismatch.")
        return

    print("\nSplitting data into Train, Validation, Test sets (GROUPED by replay_id)...")
    
    try:
        # 1. Split TrainFull / Test
        # We use a large test size because we have augmented data (many rows per replay)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split_size, random_state=42)
        train_full_idx, test_idx = next(gss_test.split(X, y_encoded, groups=groups))
        
        X_train_full = X.iloc[train_full_idx]
        y_train_full_encoded = y_encoded[train_full_idx]
        groups_train_full = groups.iloc[train_full_idx]
        
        X_test = X.iloc[test_idx]
        y_test_encoded = y_encoded[test_idx]
        groups_test = groups.iloc[test_idx]
        
        # 2. Split Train / Val (from TrainFull)
        # Calculate relative validation size
        train_full_size = 1.0 - test_split_size
        val_size_relative = val_split_size / train_full_size if train_full_size > 0 else 0.1
        if not (0 < val_size_relative < 1): val_size_relative = 0.15

        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=42)
        train_idx, val_idx = next(gss_val.split(X_train_full, y_train_full_encoded, groups=groups_train_full))
         
        X_train = X_train_full.iloc[train_idx]
        y_train_encoded = y_train_full_encoded[train_idx]
        groups_train = groups_train_full.iloc[train_idx]
        
        X_val = X_train_full.iloc[val_idx]
        y_val_encoded = y_train_full_encoded[val_idx]
        groups_val = groups_train_full.iloc[val_idx]

        # Verify Splits
        train_replays = set(groups_train)
        val_replays = set(groups_val)
        test_replays = set(groups_test)
        
        print(f"  Train Replays: {len(train_replays)}")
        print(f"  Val Replays:   {len(val_replays)}")
        print(f"  Test Replays:  {len(test_replays)}")
        
        overlap_tv = train_replays.intersection(val_replays)
        overlap_tt = train_replays.intersection(test_replays)
        overlap_vt = val_replays.intersection(test_replays)
        
        if overlap_tv or overlap_tt or overlap_vt:
             print(f"Error: Data Leakage detected! Overlap: TV={len(overlap_tv)}, TT={len(overlap_tt)}, VT={len(overlap_vt)}")
             return
        else:
             print("  Success: No replay overlap between splits.")

    except Exception as e:
        print(f"Error during GROUP splitting: {e}")
        return

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    # Cleanup big objects
    del X, df, X_train_full, y_train_full_encoded, groups, groups_train_full, groups_test, groups_train, groups_val
    gc.collect()


    # --- Calculate Class Weights ---
    print("\nCalculating class weights for handling imbalance...")
    if len(y_train_encoded) == 0: print("Error: y_train_encoded empty."); return
    unique_train_classes, class_counts = np.unique(y_train_encoded, return_counts=True)
    if len(unique_train_classes) < 2:
         print("Warning: Fewer than 2 classes in training data. Using uniform weights.")
         class_weight_dict = {cls_idx: 1.0 for cls_idx in range(num_classes)}
    else:
        try:
             class_weights_values = compute_class_weight('balanced', classes=unique_train_classes, y=y_train_encoded)
             class_weight_dict = dict(zip(unique_train_classes, class_weights_values))
             # Add weight 1.0 for classes not seen in training (essential for TF)
             all_possible_classes = np.arange(num_classes)
             for cls_idx in all_possible_classes:
                  if cls_idx not in class_weight_dict:
                      # print(f"Debug: Assigning weight 1.0 to class {cls_idx} (not in train set)") # Optional Debug
                      class_weight_dict[cls_idx] = 1.0 # Assign default weight
             print(f"Class weights calculated (Example: {list(class_weight_dict.items())[:5]}...)")
        except Exception as e:
             print(f"Error calculating class weights: {e}. Using uniform weights.")
             class_weight_dict = {cls_idx: 1.0 for cls_idx in range(num_classes)}


    # --- Preprocess X data based on model type ---
    X_train_processed, X_val_processed, X_test_processed = None, None, None
    preprocessor = None
    # <--- MODIFIED: Use label_encoder_suffix in filenames --->
    feature_lists_path = f'models/action_feature_lists_v4_{label_encoder_suffix}.joblib'
    preprocessor_path = f'models/action_tf_preprocessor_v4_{label_encoder_suffix}.joblib'

    # Use the actual columns present in X_train after feature engineering
    final_train_cols = X_train.columns.tolist()
    # Ensure numerical/categorical feature lists only contain columns present in X_train
    numerical_features = [f for f in numerical_features if f in final_train_cols]
    categorical_features = [f for f in categorical_features if f in final_train_cols]
    try:
         joblib.dump({
             'feature_columns_final': final_train_cols,
             'numerical_features': numerical_features, # Use filtered lists
             'categorical_features': categorical_features # Use filtered lists
             }, feature_lists_path)
         print(f"Final feature lists saved to {feature_lists_path}")
    except Exception as e: print(f"Error saving feature lists: {e}")


    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline (OneHotEncoder + Scaler)...")
        transformers = []
        # Use the potentially filtered numerical_features list
        if numerical_features: # Check if list is not empty
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            transformers.append(('num', numerical_transformer, numerical_features))
        else: print("Info: No numerical features to scale for TF.")
        # Use the potentially filtered categorical_features list
        if categorical_features: # Check if list is not empty
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
            transformers.append(('cat', categorical_transformer, categorical_features))
        else: print("Info: No categorical features to OneHotEncode for TF.")

        if not transformers: print("Error: No transformers created for TF!"); return

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)

        print("Applying TF preprocessing (fit on train, transform all)...")
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            print(f"  Fit TF preprocessor on training data (Output shape: {X_train_processed.shape})")
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            print(f"TF Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
            joblib.dump(preprocessor, preprocessor_path)
            print(f"TF preprocessor saved to {preprocessor_path}")
        except ValueError as e:
            print(f"ValueError during TF preprocessing: {e}")
            # Debug mismatch
            if categorical_features:
                for col in categorical_features:
                    if col in X_train.columns and col in X_val.columns and col in X_test.columns:
                         train_cats = set(X_train[col].unique())
                         val_cats = set(X_val[col].unique())
                         test_cats = set(X_test[col].unique())
                         if not val_cats.issubset(train_cats) or not test_cats.issubset(train_cats):
                              print(f"  Mismatch detected in column '{col}':")
                              print(f"    Val not in Train: {val_cats - train_cats}")
                              print(f"    Test not in Train: {test_cats - train_cats}")
                    else: print(f"  Column '{col}' not present in all splits.")
            return
        except MemoryError: print("MemoryError during TF preprocessing."); return
        except Exception as e: print(f"Error during TF preprocessing: {e}"); import traceback; traceback.print_exc(); return

        del X_train, X_val, X_test; gc.collect()

    elif model_type == 'lightgbm':
        print("\nPreprocessing for LGBM (dtype conversion, scaling) will occur inside its training function.")
        X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test # Pass original DFs

    else:
        print(f"Error: Unknown model_type '{model_type}'"); return

    # --- Train Selected Model ---
    print(f"\n--- Initiating {model_type.upper()} Model Training ({feature_set.upper()} features) ---")
    if model_type == 'tensorflow':
         if X_train_processed is not None:
             train_tensorflow_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                               y_train_encoded, y_val_encoded, y_test_encoded,
                                               num_classes, class_weight_dict, label_encoder,
                                               epochs, batch_size, learning_rate,
                                               label_suffix=label_encoder_suffix) # <--- MODIFIED pass suffix
         else: print("Skipping TF training due to preprocessing errors.")
    elif model_type == 'lightgbm':
         # Pass potentially filtered feature lists
         train_lgbm_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                     y_train_encoded, y_val_encoded, y_test_encoded,
                                     numerical_features, categorical_features,
                                     num_classes, class_weight_dict, label_encoder,
                                     label_suffix=label_encoder_suffix,
                                     lgbm_estimators=lgbm_estimators, lgbm_lr=lgbm_lr,
                                     label_smoothing=label_smoothing, # <--- NEW: pass smoothing to lgbm
                                     use_gpu=use_gpu) # <--- MODIFIED pass suffix


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action predictor (V4 with predict_mode).") # <--- MODIFIED description
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='lightgbm', help="Type of model to train.")
    parser.add_argument("--feature_set", choices=['full', 'medium', 'simplified'], default='full', help="Feature set to use.")
    # <--- NEW Argument --->
    parser.add_argument("--disable_smogon_features", action="store_true", help="Disable injecting Smogon usage stats features.")
    parser.add_argument("--smogon_top_n", type=int, default=100, help="Number of Top-N Smogon moves to keep (default 100). Use -1 for all.")
    # --------------------
    parser.add_argument("--min_turn", type=int, default=1, help="Minimum turn number to include.")
    parser.add_argument("--min_move_count", type=int, default=0,
                        help="Minimum occurrences for a move to be included (0=no filter, recommended: 50-200).")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction for test set.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction for validation set.")
    # TF specific args
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (TF only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (TF only).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (TF only).")
    parser.add_argument("--lgbm_estimators", type=int, default=1500, help="Number of trees (default 1500).")
    parser.add_argument("--lgbm_lr", type=float, default=0.02, help="Learning rate (default 0.02).")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Target label smoothing (0.0-0.2, recommended: 0.1).")
    parser.add_argument("--min_feature_replay_count", type=int, default=50, help="Min unique replays for a feature to be included.")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU training for LightGBM.")
    parser.add_argument("--blind_own_bench", action="store_true", help="Remove own bench species to prevent overfitting.")
    parser.add_argument("--blind_opp_bench", action="store_true", help="Remove opponent bench species to prevent overfitting.")

    args = parser.parse_args()

    # Validate split sizes
    if args.test_split + args.val_split >= 1.0: print("Error: test_split + val_split must be < 1.0"); exit(1)
    if args.test_split <= 0 or args.val_split <= 0: print("Error: test_split and val_split must be > 0."); exit(1)

    # <--- MODIFIED call to pass new argument --->
    run_action_training(
        parquet_path=args.parquet_file,
        model_type=args.model_type,
        feature_set=args.feature_set,
        min_turn=args.min_turn,
        min_move_count=args.min_move_count,  # NEW: pass min_move_count
        test_split_size=args.test_split,
        val_split_size=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        disable_smogon_features=args.disable_smogon_features,
        smogon_top_n=args.smogon_top_n, # <--- NEW
        lgbm_estimators=args.lgbm_estimators,
        lgbm_lr=args.lgbm_lr,
        label_smoothing=args.label_smoothing,
        min_feature_replay_count=args.min_feature_replay_count,
        use_gpu=args.use_gpu,
        blind_own_bench=args.blind_own_bench,
        blind_opp_bench=args.blind_opp_bench
    )
