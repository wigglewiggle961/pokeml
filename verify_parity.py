import pandas as pd
import json
import re
import numpy as np

def sanitize_old(name):
    return name.lower().replace(' ', '').replace('-', '').replace('_', '').replace(':', '').replace('%', 'perc')

def sanitize_new(name):
    # Strip everything except alphanumeric
    return re.sub(r'[^a-z0-9]', '', name.lower())

def verify_parity():
    json_path = 'data/gen9ou-0.json'
    parquet_path = '30k.parquet'
    
    print(f"Loading Smogon JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        smogon_data = json.load(f).get('data', {})
    
    smogon_moves = set()
    for pkmn, data in smogon_data.items():
        if 'Moves' in data:
            for move in data['Moves'].keys():
                smogon_moves.add(move)
    
    # Baseline
    smogon_old = {sanitize_old(m) for m in smogon_moves}
    smogon_new = {sanitize_new(m) for m in smogon_moves}
    
    print(f"Loading Parquet sample...")
    # Read a sample to be fast
    df = pd.read_parquet(parquet_path, columns=['action_taken'])
    dataset_moves_raw = {action.replace('move:', '') for action in df['action_taken'].unique() if isinstance(action, str) and action.startswith('move:')}
    
    dataset_old = {sanitize_old(m) for m in dataset_moves_raw}
    overlap_old = dataset_old.intersection(smogon_old)
    parity_old = (len(overlap_old) / len(dataset_old)) * 100 if dataset_old else 0
    
    dataset_new = {sanitize_new(m) for m in dataset_moves_raw}
    overlap_new = dataset_new.intersection(smogon_new)
    parity_new = (len(overlap_new) / len(dataset_new)) * 100 if dataset_new else 0
    
    print(f"\nOLD PARITY: {parity_old:.2f}% ({len(overlap_old)}/{len(dataset_old)})")
    print(f"NEW PARITY: {parity_new:.2f}% ({len(overlap_new)}/{len(dataset_new)})")
    
    if len(dataset_new - smogon_new) > 0:
        missing = sorted(list(dataset_new - smogon_new))
        print(f"\nMissing {len(missing)} moves. Top 20:")
        for m in missing[:20]:
            print(f"  - {m}")

if __name__ == "__main__":
    verify_parity()
