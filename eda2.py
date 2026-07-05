# --- START OF EXPANDED EDA SCRIPT ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency, ttest_ind
import warnings

# --- Configuration ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None 

# Set plotting style and parameters
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100

# --- File Path ---
# !!! IMPORTANT: UPDATE THIS PATH TO YOUR PARQUET FILE !!!
parquet_file_path = Path('data/10k.parquet')
# ---------------------------------------------------------


# --- Helper Functions (Expanded) ---
def get_contextual_info(row):
    """
    Consolidates the logic for extracting all necessary contextual information for a given turn.
    This is more efficient than applying multiple separate functions.
    """
    player = row['player_to_move']
    if player not in ['p1', 'p2']:
        return pd.Series([None]*10, index=['active_pokemon', 'opponent_pokemon', 'active_hp_perc', 
                                           'active_status', 'is_terastallized', 'tera_type', 
                                           'player_last_move', 'hazards_present', 'toxic_spikes_layers',
                                           'sticky_web_present'])
    
    opponent = 'p2' if player == 'p1' else 'p1'
    
    # Active/Opponent Pokemon and their details
    active_pokemon, opponent_pokemon = None, None
    active_hp, active_status, is_tera, tera_type = None, None, 0, 'none'
    
    for i in range(1, 7):
        if row.get(f'{player}_slot{i}_is_active') == 1:
            active_pokemon = row.get(f'{player}_slot{i}_species')
            active_hp = row.get(f'{player}_slot{i}_hp_perc')
            active_status = row.get(f'{player}_slot{i}_status')
            is_tera = row.get(f'{player}_slot{i}_terastallized', 0)
            tera_type = row.get(f'{player}_slot{i}_tera_type', 'none')
        if row.get(f'{opponent}_slot{i}_is_active') == 1:
            opponent_pokemon = row.get(f'{opponent}_slot{i}_species')

    # Player's own last move
    player_last_move = row.get(f'last_move_{player}')

    # Hazard check (on the player's side of the field)
    sr = row.get(f'{player}_hazard_stealthrock', 0) > 0
    spikes = row.get(f'{player}_hazard_spikes', 0) > 0
    tspikes = row.get(f'{player}_hazard_toxicspikes', 0)
    web = row.get(f'{player}_hazard_stickyweb', 0) > 0
    hazards_present = any([sr, spikes, tspikes > 0, web])
    
    return pd.Series([active_pokemon, opponent_pokemon, active_hp, active_status, is_tera, 
                      tera_type, player_last_move, hazards_present, tspikes, web], 
                     index=['active_pokemon', 'opponent_pokemon', 'active_hp_perc', 
                            'active_status', 'is_terastallized', 'tera_type', 
                            'player_last_move', 'hazards_present', 'toxic_spikes_layers',
                            'sticky_web_present'])

# --- (The first 4 EDA functions from the previous response remain unchanged) ---
# --- They are included here for completeness. You can scroll past them. ---

# --- EDA 1: Target Variable Distribution (Long Tail vs. Short Tail) ---
def analyze_action_distribution(df):
    """
    Analyzes the distribution of the target variable 'action_taken' to understand
    move popularity and identify the long-tail phenomenon.
    """
    print("\n" + "="*80)
    print("EDA 1: Target Variable Distribution (Long Tail Analysis)")
    print("="*80)
    df_actions = df['action_taken'].str.split(':', expand=True)
    df_actions.columns = ['action_type', 'action_value']
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='action_type', data=df_actions, palette="magma", order=['move', 'switch'])
    plt.title('Distribution of Player Action Types (Move vs. Switch)', fontsize=16)
    plt.xlabel('Action Type'); plt.ylabel('Frequency')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.show()
    move_actions = df_actions[df_actions['action_type'] == 'move']['action_value']
    top_moves = move_actions.value_counts().nlargest(25)
    plt.figure(figsize=(12, 10))
    sns.barplot(x=top_moves.values, y=top_moves.index, palette='viridis')
    plt.title('Top 25 Most Frequently Chosen Moves', fontsize=16)
    plt.xlabel('Frequency'); plt.ylabel('Move Name')
    plt.show()
    move_counts = move_actions.value_counts()
    plt.figure(figsize=(10, 6))
    plt.loglog(sorted(move_counts.values, reverse=True))
    plt.title('Log-Log Frequency Distribution of Moves (The "Long Tail")', fontsize=16)
    plt.xlabel('Rank of Move (Log Scale)'); plt.ylabel('Frequency of Use (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.show()
    total_moves_used = len(move_actions)
    unique_moves = move_actions.nunique()
    top_10_percent_count = int(np.ceil(0.1 * unique_moves))
    top_10_percent_usage = move_counts.head(top_10_percent_count).sum()
    print("\n--- Statistical Insights for Move Distribution ---")
    print(f"Total move actions recorded: {total_moves_used:,}")
    print(f"Number of unique moves used: {unique_moves:,}")
    print(f"The top 25 moves account for {top_moves.sum() / total_moves_used:.2%} of all moves used.")
    print(f"The top 10% of unique moves (~{top_10_percent_count} moves) account for {top_10_percent_usage / total_moves_used:.2%} of all moves used.")
    print("\nInterpretation: A steep drop-off followed by a long, flat line (the 'tail') indicates that a small number of moves are used very frequently, while a vast majority are used very rarely. This distribution is crucial for modeling.")

# --- EDA 2: Active Pokémon's Influence on Move Choice ---
def analyze_active_pokemon_move_choice(df, active_pokemon_col, top_n=15):
    print("\n" + "="*80)
    print("EDA 2: Active Pokémon's Influence on Move Choice")
    print("="*80)
    df_moves = df[df['action_taken'].str.startswith('move:')].copy()
    df_moves['move_name'] = df_moves['action_taken'].str.replace('move:', '')
    top_pokemon = df_moves[active_pokemon_col].value_counts().nlargest(top_n).index
    top_moves = df_moves['move_name'].value_counts().nlargest(top_n).index
    df_filtered = df_moves[df_moves[active_pokemon_col].isin(top_pokemon) & df_moves['move_name'].isin(top_moves)]
    contingency_table = pd.crosstab(df_filtered[active_pokemon_col], df_filtered['move_name'])
    plt.figure(figsize=(16, 12))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Heatmap of Top {top_n} Pokémon vs. Top {top_n} Moves', fontsize=16)
    plt.xlabel('Move Name'); plt.ylabel('Active Pokémon')
    plt.xticks(rotation=45, ha='right'); plt.show()
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print("\n--- Statistical Test: Chi-Squared Test of Independence ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}, P-value: {p}")
    print("\nInterpretation: A p-value less than 0.05 provides strong statistical evidence that the choice of move is DEPENDENT on the active Pokémon, making it a highly predictive feature.")

# --- EDA 3: Opponent's Influence on Move Choice ---
def analyze_opponent_influence(df, opponent_pokemon_col, top_n=15):
    print("\n" + "="*80)
    print("EDA 3: Opponent's Active Pokémon's Influence on Move Choice")
    print("="*80)
    df_moves = df[df['action_taken'].str.startswith('move:')].copy()
    df_moves['move_name'] = df_moves['action_taken'].str.replace('move:', '')
    top_opponents = df_moves[opponent_pokemon_col].value_counts().nlargest(top_n).index
    top_moves_against_opp = df_moves['move_name'].value_counts().nlargest(top_n).index
    df_filtered = df_moves[df_moves[opponent_pokemon_col].isin(top_opponents) & df_moves['move_name'].isin(top_moves_against_opp)]
    contingency_table = pd.crosstab(df_filtered[opponent_pokemon_col], df_filtered['move_name'])
    plt.figure(figsize=(16, 12))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Reds', linewidths=.5)
    plt.title(f"Heatmap of Player's Move Choice vs. Top {top_n} Opponent Pokémon", fontsize=16)
    plt.xlabel("Player's Chosen Move"); plt.ylabel("Opponent's Active Pokémon")
    plt.xticks(rotation=45, ha='right'); plt.show()
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print("\n--- Statistical Test: Chi-Squared Test of Independence ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}, P-value: {p}")
    print("\nInterpretation: A low p-value indicates that the player's move choice is heavily influenced by the opponent's Pokémon, reflecting strategic decisions like using super-effective moves.")

# --- EDA 4: HP and Status Influence on Action Choice (Move vs. Switch) ---
def analyze_hp_status_influence(df, active_hp_col, active_status_col):
    print("\n" + "="*80)
    print("EDA 4: HP and Status Condition's Influence on Action Choice")
    print("="*80)
    df_analysis = df.copy()
    df_analysis['action_type'] = df_analysis['action_taken'].apply(lambda x: 'switch' if 'switch' in x else 'move')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_analysis, x=active_hp_col, hue='action_type', fill=True, common_norm=False, palette='coolwarm')
    plt.title('HP Distribution for "Move" vs. "Switch" Actions', fontsize=16)
    plt.xlabel('Active Pokémon HP (%)'); plt.ylabel('Density'); plt.show()
    hp_move = df_analysis[df_analysis['action_type'] == 'move'][active_hp_col].dropna()
    hp_switch = df_analysis[df_analysis['action_type'] == 'switch'][active_hp_col].dropna()
    t_stat, p_val_ttest = ttest_ind(hp_move, hp_switch, equal_var=False)
    print("\n--- Statistical Test: Independent T-test for HP ---")
    print(f"Mean HP when choosing 'move': {hp_move.mean():.2f}%")
    print(f"Mean HP when choosing 'switch': {hp_switch.mean():.2f}%")
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_val_ttest}")
    print("Interpretation: A very low p-value confirms players are statistically more likely to switch when their Pokémon's HP is low.")
    status_order = df_analysis[active_status_col].value_counts().index
    status_action_counts = df_analysis.groupby([active_status_col, 'action_type']).size().unstack(fill_value=0)
    status_action_props = status_action_counts.div(status_action_counts.sum(axis=1), axis=0)
    status_action_props.loc[status_order].plot(kind='bar', stacked=True, figsize=(12, 7), colormap='plasma')
    plt.title('Proportion of Move vs. Switch by Status Condition', fontsize=16)
    plt.xlabel('Status Condition'); plt.ylabel('Proportion of Actions')
    plt.xticks(rotation=45); plt.show()
    chi2, p_val_chi2, _, _ = chi2_contingency(status_action_counts)
    print("\n--- Statistical Test: Chi-Squared for Status Condition ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}, P-value: {p_val_chi2}")
    print("Interpretation: A low p-value suggests the player's action (move/switch) is dependent on the Pokemon's status, making it a relevant feature.")

# --- NEW EDA 5: Battle Momentum - Last Move's Influence on Current Move ---
def analyze_last_move_influence(df, top_n=15):
    """
    Analyzes if the last move a player used influences their current move choice,
    revealing common two-turn strategies (e.g., setup -> attack).
    """
    print("\n" + "="*80)
    print("EDA 5: Battle Momentum - Last Move's Influence on Current Move")
    print("="*80)

    df_moves = df[df['action_taken'].str.startswith('move:')].copy()
    df_moves['move_name'] = df_moves['action_taken'].str.replace('move:', '')
    
    # Filter out turns where the last move was 'none' (e.g., first turn)
    df_filtered = df_moves[df_moves['player_last_move'] != 'none'].copy()

    top_last_moves = df_filtered['player_last_move'].value_counts().nlargest(top_n).index
    top_current_moves = df_filtered['move_name'].value_counts().nlargest(top_n).index

    df_crosstab = df_filtered[df_filtered['player_last_move'].isin(top_last_moves) & 
                              df_filtered['move_name'].isin(top_current_moves)]

    if df_crosstab.empty:
        print("Not enough data to analyze last move influence.")
        return

    contingency_table = pd.crosstab(df_crosstab['player_last_move'], df_crosstab['move_name'])
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='cividis', linewidths=.5)
    plt.title(f'Heatmap of Current Move Choice vs. Player\'s Last Move (Top {top_n})', fontsize=16)
    plt.xlabel('Current Move Chosen', fontsize=12)
    plt.ylabel('Last Move Used by Player', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print("\n--- Statistical Test: Chi-Squared Test of Independence ---")
    print(f"Chi-Squared Statistic: {chi2:.2f}, P-value: {p}")
    print("\nInterpretation:")
    print("A low p-value here is extremely significant. It indicates 'statefulness' in decision-making.")
    print("Players don't choose moves in a vacuum; their previous action heavily influences their next one.")
    print("This reveals common strategies like 'U-turn' into a switch, or a setup move like 'Swords Dance' followed by a powerful attack.")
    print("Therefore, 'last_move' is a critical feature for capturing battle flow and momentum.")

# --- NEW EDA 6: Board Control - Influence of Entry Hazards on Switching ---
def analyze_hazard_influence(df):
    """
    Analyzes if the presence of entry hazards on the player's field
    changes their propensity to switch.
    """
    print("\n" + "="*80)
    print("EDA 6: Board Control - Influence of Entry Hazards on Switching")
    print("="*80)
    
    df_analysis = df.copy()
    df_analysis['action_type'] = df_analysis['action_taken'].apply(lambda x: 'switch' if 'switch' in x else 'move')

    # 1. Bar plot of switch proportion
    props = df_analysis.groupby('hazards_present')['action_type'].value_counts(normalize=True).unstack()
    
    props.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#3a5e8c', '#f9a620'])
    plt.title('Proportion of Actions with vs. without Hazards', fontsize=16)
    plt.xlabel('Are Entry Hazards on Player\'s Field?', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.xticks([0, 1], ['No Hazards', 'Hazards Present'], rotation=0)
    plt.legend(title='Action Type')
    plt.show()

    # 2. Statistical Test (Chi-Squared)
    contingency_table = pd.crosstab(df_analysis['hazards_present'], df_analysis['action_type'])
    
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print("\n--- Statistical Test: Chi-Squared Test of Independence ---")
    print("Contingency Table:")
    print(contingency_table)
    print(f"\nChi-Squared Statistic: {chi2:.2f}, P-value: {p}")
    print("\nInterpretation:")
    print("This analysis reveals a core strategic tension. A low p-value shows that hazards significantly alter player behavior.")
    print("Players must perform a cost-benefit analysis: is the damage or effect from hazards upon switching worth the tactical advantage of a new matchup?")
    print("This feature is vital for the model to understand board state and predict if a player will risk a switch.")

# --- NEW EDA 7: The Game Changer - Terastallization's Effect on Move Choice ---
def analyze_tera_influence(df):
    """
    Compares the most common moves used by Terastallized vs. non-Terastallized Pokemon
    to see how this mechanic warps move selection.
    """
    print("\n" + "="*80)
    print("EDA 7: The Game Changer - Terastallization's Effect on Move Choice")
    print("="*80)

    df_moves = df[df['action_taken'].str.startswith('move:')].copy()
    df_moves['move_name'] = df_moves['action_taken'].str.replace('move:', '')

    tera_active = df_moves[df_moves['is_terastallized'] == 1]
    tera_inactive = df_moves[df_moves['is_terastallized'] == 0]

    if tera_active.empty:
        print("No Terastallization data found to analyze.")
        return

    top_moves_tera = tera_active['move_name'].value_counts().nlargest(15)
    top_moves_normal = tera_inactive['move_name'].value_counts().nlargest(15)

    # 1. Side-by-side bar plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=False)
    
    sns.barplot(ax=axes[0], x=top_moves_normal.values, y=top_moves_normal.index, palette='Blues_r')
    axes[0].set_title('Top 15 Moves (Not Terastallized)', fontsize=16)
    axes[0].set_xlabel('Frequency')
    
    sns.barplot(ax=axes[1], x=top_moves_tera.values, y=top_moves_tera.index, palette='Oranges_r')
    axes[1].set_title('Top 15 Moves (When Terastallized)', fontsize=16)
    axes[1].set_xlabel('Frequency')
    
    plt.suptitle('Comparison of Move Choice: Normal vs. Terastallized State', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 2. Interpretation
    print("\n--- Qualitative Analysis ---")
    print(f"Most common move while Terastallized: '{top_moves_tera.index[0]}'")
    print(f"Most common move normally: '{top_moves_normal.index[0]}'")
    print("\nInterpretation:")
    print("This comparison directly shows how Terastallization warps a player's strategy.")
    print("Often, the move distribution will shift dramatically. Players will prioritize moves that match the Tera Type to gain a massive power boost (STAB).")
    print("The model learning this pattern is crucial for predicting a player's offensive choices in the most pivotal turns of the game.")
    print("The 'is_terastallized' feature is therefore highly predictive of a shift towards powerful, type-matching attacks.")

# --- Main Execution Block ---
def main():
    """
    Main function to load data and run all 7 EDA analyses.
    """
    print("Starting Comprehensive EDA for Pokémon Action Prediction (7 Analyses)...")
    
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
    context_cols = df.apply(get_contextual_info, axis=1)
    df = pd.concat([df, context_cols], axis=1)
    
    # Drop rows where key information couldn't be determined for a clean analysis
    df.dropna(subset=['action_taken', 'active_pokemon', 'opponent_pokemon', 
                      'active_hp_perc', 'active_status', 'player_last_move'], inplace=True)
    print("Preprocessing complete.")

    # --- Run Original Analyses ---
    analyze_action_distribution(df)
    analyze_active_pokemon_move_choice(df, 'active_pokemon')
    analyze_opponent_influence(df, 'opponent_pokemon')
    analyze_hp_status_influence(df, 'active_hp_perc', 'active_status')

    # --- Run NEW Deeper Analyses ---
    analyze_last_move_influence(df)
    analyze_hazard_influence(df)
    analyze_tera_influence(df)
    
    print("\n" + "="*80)
    print("EDA script finished successfully.")
    print("="*80)

if __name__ == "__main__":
    main()

# --- END OF EXPANDED EDA SCRIPT ---```

### Summary of New, Deeper Analyses

