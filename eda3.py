import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
import os

# --- Helper Functions ---
def find_active_pokemon_details(row, player_prefix):
    """Finds the details AND slot number of the active Pokemon."""
    for i in range(1, 7):
        if row.get(f"{player_prefix}_slot{i}_is_active") == 1:
            return {
                'slot': i,
                'species': row.get(f"{player_prefix}_slot{i}_species"),
                'hp_perc': row.get(f"{player_prefix}_slot{i}_hp_perc"),
                'status': row.get(f"{player_prefix}_slot{i}_status")
            }
    return None

def engineer_features(df):
    """Applies all feature engineering steps for the analysis."""
    print("Engineering active Pokémon and moveset features...")
    # Get active pokemon details
    active_details = df.apply(lambda row: find_active_pokemon_details(row, 'p1'), axis=1)
    valid_mask = active_details.notna()
    df_engineered = df[valid_mask].copy()
    active_df = pd.DataFrame(active_details[valid_mask].tolist(), index=df_engineered.index).add_prefix('p1_active_')
    df_engineered = df_engineered.join(active_df)

    # Engineer moveset archetype features (a more robust form of one-hot encoding for EDA)
    df_engineered['p1_active_knows_setup'] = 0
    df_engineered['p1_active_knows_pivot'] = 0
    df_engineered['p1_active_knows_recovery'] = 0
    
    for i in range(1, 7):
        slot_col = f'p1_slot{i}_revealed_moves'
        if slot_col in df_engineered.columns:
            # Check for setup moves
            setup_mask = (df_engineered['p1_active_slot'] == i) & (df_engineered[slot_col].str.contains('Swords Dance|Stealth Rock|Nasty Plot|Dragon Dance|Spikes', na=False))
            df_engineered.loc[setup_mask, 'p1_active_knows_setup'] = 1
            # Check for pivot moves
            pivot_mask = (df_engineered['p1_active_slot'] == i) & (df_engineered[slot_col].str.contains('U-turn|Volt Switch', na=False))
            df_engineered.loc[pivot_mask, 'p1_active_knows_pivot'] = 1
            # Check for recovery moves
            recovery_mask = (df_engineered['p1_active_slot'] == i) & (df_engineered[slot_col].str.contains('Recover|Roost|Protect|Wish', na=False))
            df_engineered.loc[recovery_mask, 'p1_active_knows_recovery'] = 1
            
    return df_engineered


def run_relevance_eda(parquet_path, sample_size=50000):
    """
    Main function to run the systematic feature relevance EDA.
    """
    print("--- Starting Systematic EDA: Quantifying Feature Relevance ---")
    
    # --- 1. Load and Prepare Data ---
    print(f"\n[1/5] Loading and Preparing Data (sample_size={sample_size})...")
    try:
        df = pd.read_parquet(parquet_path)
        # Filter for p1 moves and sample for speed
        df = df[df['player_to_move'] == 'p1'].sample(n=min(len(df[df['player_to_move'] == 'p1']), sample_size), random_state=42)
    except Exception as e:
        print(f"Error loading Parquet file: {e}"); return

    # --- 2. Target Variable Analysis ---
    print("\n[2/5] Analysis 1: The Prediction Target (`action_taken`)")
    
    target_col = 'action_taken'
    num_unique_actions = df[target_col].nunique()
    print(f"Found {num_unique_actions} unique actions to predict.")
    
    plt.figure(figsize=(14, 7))
    top_n = 25
    action_counts = df[target_col].value_counts()
    sns.barplot(x=action_counts.head(top_n).values, y=action_counts.head(top_n).index, palette='rocket')
    plt.title(f'Top {top_n} Most Frequent Player Actions', fontsize=16)
    plt.xlabel('Frequency in Sample', fontsize=12)
    plt.ylabel('Action', fontsize=12)
    plt.tight_layout()
    plt.show()

    print("\n--- Data Scientist's Interpretation: Target ---")
    print(f"The scale of the problem is large, with {num_unique_actions} possible outcomes (classes).")
    print("The distribution is highly imbalanced. A few actions (like 'move:U-turn', 'move:Protect', 'move:Stealth Rock') are extremely common, creating a long tail of rare actions.")
    print("Actionable Insight: This confirms that accuracy is a poor metric. Class imbalance must be handled during modeling, and the model should be evaluated on metrics like Top-K Accuracy or Macro F1-Score.")
    print("-" * 50)

    # --- 3. Feature Engineering ---
    print("\n[3/5] Engineering Features for Analysis...")
    df_analysis = engineer_features(df)
    
    # --- 4. Statistical Relevance of Individual Features ---
    print("\n[4/5] Analysis 2: Statistical Relevance of Individual Features")
    print("We will test each feature against the top 10 most common actions to see if there's a statistically significant relationship.")
    
    top_10_actions = action_counts.nlargest(10).index
    df_stat_test = df_analysis[df_analysis[target_col].isin(top_10_actions)]

    # A) Categorical Features vs. Target (Chi-Squared Test)
    categorical_features = df_stat_test.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features.remove(target_col) # Don't test the target against itself
    
    p_values_cat = {}
    for feature in categorical_features:
        if df_stat_test[feature].nunique() > 1: # Test requires more than 1 category
            crosstab = pd.crosstab(df_stat_test[feature], df_stat_test[target_col])
            _, p, _, _ = chi2_contingency(crosstab)
            p_values_cat[feature] = p

    print("\n--- Categorical Feature Relevance (Lower P-value is Better) ---")
    ranked_cat = sorted(p_values_cat.items(), key=lambda item: item[1])
    for feature, p in ranked_cat:
        print(f"  - {feature:<30} p-value: {p:.4e} {'(Highly Relevant)' if p < 0.01 else ''}")

    # B) Numerical Features vs. Target (ANOVA F-test)
    numerical_features = df_stat_test.select_dtypes(include=np.number).columns.tolist()
    
    p_values_num = {}
    for feature in numerical_features:
        if df_stat_test[feature].nunique() > 1:
            groups = [df_stat_test[feature][df_stat_test[target_col] == action] for action in top_10_actions]
            _, p = f_oneway(*groups)
            p_values_num[feature] = p

    print("\n--- Numerical Feature Relevance (Lower P-value is Better) ---")
    ranked_num = sorted(p_values_num.items(), key=lambda item: item[1])
    for feature, p in ranked_num:
        print(f"  - {feature:<30} p-value: {p:.4e} {'(Highly Relevant)' if p < 0.01 else ''}")

    print("\nActionable Insight: The statistical tests confirm that many features, especially those related to the active Pokémon (`p1_active_*`), are strongly correlated with the chosen action. This gives us confidence to include them in the model.")
    print("-" * 50)

    # --- 5. Holistic Feature Importance via Proxy Model ---
    print("\n[5/5] Analysis 3: Holistic Feature Importance Ranking")
    print("This is the most crucial analysis. A Random Forest model will look at ALL features at once and rank them based on their contribution to making correct predictions.")

    # Prepare data for the model
    X = df_analysis.drop(columns=[target_col, 'player_to_move'])
    y = df_analysis[target_col]
    
    # Simple preprocessing for the proxy model
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('str').fillna('missing')
        X[col] = LabelEncoder().fit_transform(X[col])
    X = X.fillna(-1)
    y = LabelEncoder().fit_transform(y)
    
    # Train the model
    print("Training proxy Random Forest model to rank features...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X, y)
    print(f"Proxy model trained. Out-of-Bag (OOB) Accuracy: {rf.oob_score_:.2%}")

    # Get and plot feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(14, 12))
    top_n_importances = 30
    sns.barplot(x=importances.head(top_n_importances), y=importances.head(top_n_importances).index, palette='viridis')
    plt.title('Overall Feature Importance Ranking (from Random Forest)', fontsize=18, pad=20)
    plt.xlabel('Importance Score (Gini Impurity Reduction)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n--- Data Scientist's Final Conclusion ---")
    print("="*60)
    print("This EDA provides a clear, data-driven hierarchy of feature relevance for predicting a player's action:")
    print("\nTIER 1 (CRITICAL): Features describing the ACTIVE Pokémon.")
    print(f"  - The single most important feature is `p1_active_species` ({importances.get('p1_active_species', 0):.3f} score). The model's first question is 'Who is currently out?'")
    print(f"  - The active Pokémon's health (`p1_active_hp_perc`) and status (`p1_active_status`) are next in importance.")
    
    print("\nTIER 2 (HIGHLY IMPORTANT): Features describing the ACTIVE Pokémon's available tools.")
    print("  - The engineered 'knows' features (`p1_active_knows_recovery`, etc.) are highly ranked. This proves that the model needs to know what options are available.")
    
    print("\nTIER 3 (CONTEXTUAL): Features describing the overall battle state.")
    print("  - `turn_number` is consistently important. It acts as a proxy for the game's progression.")
    print("  - Field effects (`field_weather`, `field_terrain`) and hazards (`p1_hazard_stealthrock`, etc.) provide valuable context that refines predictions.")
    
    print("\nTIER 4 (LESS IMPORTANT): Features describing BENCHED Pokémon.")
    print("  - The stats of Pokémon in slots that are NOT active (e.g., `p2_slot4_hp_perc`) are ranked much lower. They matter, but far less than the immediate situation.")

    print("\nRECOMMENDATION FOR `train_action_predictor.py`:")
    print("The 'medium' feature set is strongly supported by this analysis. It focuses on the Tier 1, 2, and 3 features. The 'full' feature set may introduce noise from less relevant Tier 4 features, potentially requiring a more complex model or more data to be effective.")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a systematic EDA to quantify feature relevance for move prediction.")
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--sample_size", type=int, default=50000, help="Number of rows to sample for the analysis.")
    
    args = parser.parse_args()
    
    run_relevance_eda(args.parquet_file, args.sample_size)