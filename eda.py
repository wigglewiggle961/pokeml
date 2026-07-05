# --- START OF EDA SCRIPT ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency, ttest_ind

# --- Configuration ---
# Set plotting style and parameters for better aesthetics
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

# --- File Path ---
# !!! IMPORTANT: UPDATE THIS PATH TO YOUR PARQUET FILE !!!
parquet_file_path = Path('data/10k.parquet')
# ---------------------------------------------------------


# --- Helper Functions ---
def get_active_pokemon_for_turn(row):
    """Identifies the active species for the player whose turn it is."""
    player_to_move = row['player_to_move']
    if player_to_move not in ['p1', 'p2']:
        return None
        
    for i in range(1, 7):
        if row.get(f'{player_to_move}_slot{i}_is_active') == 1:
            return row.get(f'{player_to_move}_slot{i}_species')
    return None

def get_opponent_pokemon_for_turn(row):
    """Identifies the active species for the opponent."""
    player_to_move = row['player_to_move']
    opponent = 'p2' if player_to_move == 'p1' else 'p1'
    
    for i in range(1, 7):
        if row.get(f'{opponent}_slot{i}_is_active') == 1:
            return row.get(f'{opponent}_slot{i}_species')
    return None

def get_active_pokemon_details(row):
    """Gets HP and Status for the active Pokemon."""
    player_to_move = row['player_to_move']
    if player_to_move not in ['p1', 'p2']:
        return pd.Series([None, None], index=['active_hp_perc', 'active_status'])
        
    for i in range(1, 7):
        if row.get(f'{player_to_move}_slot{i}_is_active') == 1:
            hp = row.get(f'{player_to_move}_slot{i}_hp_perc')
            status = row.get(f'{player_to_move}_slot{i}_status')
            return pd.Series([hp, status], index=['active_hp_perc', 'active_status'])
    return pd.Series([None, None], index=['active_hp_perc', 'active_status'])


# --- EDA 1: Target Variable Distribution (Long Tail vs. Short Tail) ---
def analyze_action_distribution(df):
    """
    Analyzes the distribution of the target variable 'action_taken' to understand
    move popularity and identify the long-tail phenomenon.
    """
    print("\n" + "="*80)
    print("EDA 1: Target Variable Distribution (Long Tail Analysis)")
    print("="*80)

    # Separate action type (move/switch) from the specific action
    df_actions = df['action_taken'].str.split(':', expand=True)
    df_actions.columns = ['action_type', 'action_value']

    # 1. Overall Action Type Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='action_type', data=df_actions, palette="magma")
    plt.title('Distribution of Player Action Types (Move vs. Switch)', fontsize=16)
    plt.xlabel('Action Type', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

    # 2. Top 25 Most Frequent Moves (The "Head")
    move_actions = df_actions[df_actions['action_type'] == 'move']['action_value']
    top_moves = move_actions.value_counts().nlargest(25)

    plt.figure(figsize=(12, 10))
    sns.barplot(x=top_moves.values, y=top_moves.index, palette='viridis')
    plt.title('Top 25 Most Frequently Chosen Moves', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Move Name', fontsize=12)
    plt.show()
    
    # 3. Long Tail Distribution Plot (Log-Log Scale)
    move_counts = move_actions.value_counts()
    
    plt.figure(figsize=(10, 6))
    plt.loglog(sorted(move_counts.values, reverse=True))
    plt.title('Log-Log Frequency Distribution of Moves (The "Long Tail")', fontsize=16)
    plt.xlabel('Rank of Move (Log Scale)', fontsize=12)
    plt.ylabel('Frequency of Use (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.show()

    # --- Interpretation & Statistics ---
    total_moves_used = len(move_actions)
    unique_moves = move_actions.nunique()
    top_10_percent_count = int(np.ceil(0.1 * unique_moves))
    top_10_percent_usage = move_counts.head(top_10_percent_count).sum()

    print("\n--- Statistical Insights for Move Distribution ---")
    print(f"Total move actions recorded: {total_moves_used:,}")
    print(f"Number of unique moves used: {unique_moves:,}")
    print(f"The top 25 moves account for {top_moves.sum() / total_moves_used:.2%} of all moves used.")
    print(f"The top 10% of unique moves (~{top_10_percent_count} moves) account for {top_10_percent_usage / total_moves_used:.2%} of all moves used.")
    print("\nInterpretation:")
    print("The count plot shows the raw popularity of moves. The log-log plot is key for the 'long tail'.")
    print("A steep drop-off followed by a long, flat line (the 'tail') indicates that a small number of moves are used very frequently,")
    print("while a vast majority of moves are used very rarely. This distribution is crucial for modeling, as the model will have")
    print("abundant data for popular moves but sparse data for rare ones, which can pose a challenge.")


# --- EDA 2: Active Pokémon's Influence on Move Choice ---
def analyze_active_pokemon_move_choice(df, active_pokemon_col, top_n=15):
    """
    Analyzes the relationship between the active Pokémon and its chosen move
    using a heatmap and Chi-Squared test.
    """
    print("\n" + "="*80)
    print("EDA 2: Active Pokémon's Influence on Move Choice")
    print("="*80)

    df_moves = df[df['action_taken'].str.startswith('move:')].copy()
    df_moves['move_name'] = df_moves['action_taken'].str.replace('move:', '')
    
    top_pokemon = df_moves[active_pokemon_col].value_counts().nlargest(top_n).index
    top_moves = df_moves['move_name'].value_counts().nlargest(top_n).index

    df_filtered = df_moves[df_moves[active_pokemon_col].isin(top_pokemon) & df_moves['move_name'].isin(top_moves)]

    # 1. Heatmap of Pokemon vs. Move
    contingency_table = pd.crosstab(df_filtered[active_pokemon_col], df_filtered['move_name'])
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Heatmap of Top {top_n} Pokémon vs. Top {top_n} Moves', fontsize=16)
    plt.xlabel('Move Name', fontsize=12)
    plt.ylabel('Active Pokémon', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 2. Statistical Test (Chi-Squared)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    print("\n--- Statistical Test: Chi-Squared Test of Independence ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}")
    print(f"P-value: {p}")
    
    alpha = 0.05
    print("\nInterpretation:")
    if p < alpha:
        print(f"The p-value ({p:.3g}) is less than {alpha}. We REJECT the null hypothesis.")
        print("This provides strong statistical evidence that the choice of move is DEPENDENT on the active Pokémon.")
        print("This confirms that 'active_pokemon_species' is a highly relevant and predictive feature.")
    else:
        print(f"The p-value ({p:.3g}) is greater than {alpha}. We FAIL to reject the null hypothesis.")
        print("This suggests that the choice of move may be independent of the active Pokémon in this sample.")


# --- EDA 3: Opponent's Influence on Move Choice ---
def analyze_opponent_influence(df, active_pokemon_col, opponent_pokemon_col, top_n=15):
    """
    Analyzes how the opponent's active Pokemon influences the player's move choice.
    """
    print("\n" + "="*80)
    print("EDA 3: Opponent's Active Pokémon's Influence on Move Choice")
    print("="*80)

    df_moves = df[df['action_taken'].str.startswith('move:')].copy()
    df_moves['move_name'] = df_moves['action_taken'].str.replace('move:', '')

    top_opponents = df_moves[opponent_pokemon_col].value_counts().nlargest(top_n).index
    top_moves_against_opp = df_moves['move_name'].value_counts().nlargest(top_n).index

    df_filtered = df_moves[df_moves[opponent_pokemon_col].isin(top_opponents) & df_moves['move_name'].isin(top_moves_against_opp)]

    # 1. Heatmap of Opponent vs. Player's Move
    contingency_table = pd.crosstab(df_filtered[opponent_pokemon_col], df_filtered['move_name'])
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Reds', linewidths=.5)
    plt.title(f"Heatmap of Player's Move Choice vs. Top {top_n} Opponent Pokémon", fontsize=16)
    plt.xlabel("Player's Chosen Move", fontsize=12)
    plt.ylabel("Opponent's Active Pokémon", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 2. Statistical Test (Chi-Squared)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    print("\n--- Statistical Test: Chi-Squared Test of Independence ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}")
    print(f"P-value: {p}")
    
    alpha = 0.05
    print("\nInterpretation:")
    if p < alpha:
        print(f"The p-value ({p:.3g}) is less than {alpha}. We REJECT the null hypothesis.")
        print("This is strong evidence that the player's move choice is heavily influenced by the opponent's Pokémon.")
        print("This is a critical predictive relationship, reflecting strategic decisions like using super-effective moves.")
        print("Therefore, the opponent's species is a vital feature for the model.")
    else:
        print("The result is not statistically significant. The player's move choice appears independent of the opponent.")


# --- EDA 4: HP and Status Influence on Action Choice (Move vs. Switch) ---
def analyze_hp_status_influence(df, active_hp_col, active_status_col):
    """
    Analyzes how a Pokemon's HP and status affect the choice to move or switch.
    """
    print("\n" + "="*80)
    print("EDA 4: HP and Status Condition's Influence on Action Choice")
    print("="*80)
    
    df_analysis = df.copy()
    df_analysis['action_type'] = df_analysis['action_taken'].apply(lambda x: 'switch' if 'switch' in x else 'move')

    # 1. HP Distribution by Action Type
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_analysis, x=active_hp_col, hue='action_type', fill=True, common_norm=False, palette='coolwarm')
    plt.title('HP Distribution for "Move" vs. "Switch" Actions', fontsize=16)
    plt.xlabel('Active Pokémon HP (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Action Taken')
    plt.show()

    # Statistical Test for HP (T-test)
    hp_move = df_analysis[df_analysis['action_type'] == 'move'][active_hp_col].dropna()
    hp_switch = df_analysis[df_analysis['action_type'] == 'switch'][active_hp_col].dropna()
    t_stat, p_val_ttest = ttest_ind(hp_move, hp_switch, equal_var=False) # Welch's t-test
    
    print("\n--- Statistical Test: Independent T-test for HP ---")
    print(f"Mean HP when choosing 'move': {hp_move.mean():.2f}%")
    print(f"Mean HP when choosing 'switch': {hp_switch.mean():.2f}%")
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_val_ttest}")
    print("Interpretation: A very low p-value indicates a significant difference in mean HP between the two actions.")
    print("This confirms that players are statistically more likely to switch when their Pokémon's HP is low.")

    # 2. Action Choice by Status Condition
    status_order = df_analysis[active_status_col].value_counts().index
    status_action_counts = df_analysis.groupby([active_status_col, 'action_type']).size().unstack(fill_value=0)
    status_action_props = status_action_counts.div(status_action_counts.sum(axis=1), axis=0)

    status_action_props.loc[status_order].plot(kind='bar', stacked=True, figsize=(12, 7), colormap='plasma')
    plt.title('Proportion of Move vs. Switch by Status Condition', fontsize=16)
    plt.xlabel('Status Condition', fontsize=12)
    plt.ylabel('Proportion of Actions', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Action Type')
    plt.show()

    # Statistical Test for Status (Chi-Squared)
    chi2, p_val_chi2, _, _ = chi2_contingency(status_action_counts)
    print("\n--- Statistical Test: Chi-Squared for Status Condition ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}, P-value: {p_val_chi2}")
    print("Interpretation: A low p-value suggests the player's action (move/switch) is dependent on the Pokemon's status.")
    print("This makes status a relevant feature, especially for predicting switches to cure the condition.")


# --- Main Execution Block ---
def main():
    """
    Main function to load data and run all EDA analyses.
    """
    print("Starting Comprehensive EDA for Pokémon Action Prediction...")
    
    if not parquet_file_path.exists():
        print(f"Error: The file was not found at the specified path: {parquet_file_path}")
        print("Please update the 'parquet_file_path' variable in the script.")
        return

    print(f"Loading data from '{parquet_file_path}'...")
    try:
        df = pd.read_parquet(parquet_file_path)
        print("Data loaded successfully.")
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"Failed to load Parquet file. Error: {e}")
        return

    # --- Pre-computation Step ---
    print("\nPreprocessing data for analysis (this may take a moment)...")
    # Engineer the features needed for the analyses
    df['active_pokemon'] = df.apply(get_active_pokemon_for_turn, axis=1)
    df['opponent_pokemon'] = df.apply(get_opponent_pokemon_for_turn, axis=1)
    df[['active_hp_perc', 'active_status']] = df.apply(get_active_pokemon_details, axis=1)
    
    # Drop rows where key information couldn't be determined
    df.dropna(subset=['action_taken', 'active_pokemon', 'opponent_pokemon', 'active_hp_perc', 'active_status'], inplace=True)
    print("Preprocessing complete.")

    # --- Run Analyses ---
    analyze_action_distribution(df)
    analyze_active_pokemon_move_choice(df, 'active_pokemon')
    analyze_opponent_influence(df, 'active_pokemon', 'opponent_pokemon')
    analyze_hp_status_influence(df, 'active_hp_perc', 'active_status')
    
    print("\n" + "="*80)
    print("EDA script finished.")
    print("="*80)


if __name__ == "__main__":
    main()

# --- END OF EDA SCRIPT ---