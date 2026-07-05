"""
augment_perspectives.py
Doubles training data by swapping p1/p2 perspectives.

For each p2 row:
  - Swap all p1_* columns with p2_* columns
  - Change player_to_move from 'p2' to 'p1'
  - Keep the action_taken as-is (it was p2's action, now it's "our" action)

This gives the model data from BOTH sides of every battle.

Usage:
    python augment_perspectives.py input.parquet output_augmented.parquet
"""

import pandas as pd
import argparse
import time


def swap_perspectives(df_p2: pd.DataFrame) -> pd.DataFrame:
    """
    Take p2 rows and swap p1/p2 columns so they look like p1 rows.
    """
    # Identify p1 and p2 columns
    p1_cols = [c for c in df_p2.columns if c.startswith('p1_')]
    p2_cols = [c for c in df_p2.columns if c.startswith('p2_')]
    
    # Build rename mapping: p1_* -> p2_*, p2_* -> p1_*
    rename_map = {}
    for col in p1_cols:
        swapped = 'p2_' + col[3:]  # p1_xxx -> p2_xxx
        rename_map[col] = swapped
    for col in p2_cols:
        swapped = 'p1_' + col[3:]  # p2_xxx -> p1_xxx
        rename_map[col] = swapped
    
    # Rename columns (this swaps p1 <-> p2)
    df_swapped = df_p2.rename(columns=rename_map)
    
    # Fix player_to_move
    df_swapped['player_to_move'] = 'p1'
    
    return df_swapped


def augment_parquet(input_path: str, output_path: str):
    """Load parquet, swap p2 perspectives, combine, and save."""
    
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Total rows: {len(df)}")
    
    # Split by player
    df_p1 = df[df['player_to_move'] == 'p1'].copy()
    df_p2 = df[df['player_to_move'] == 'p2'].copy()
    print(f"  p1 rows: {len(df_p1)}")
    print(f"  p2 rows: {len(df_p2)}")
    
    if len(df_p2) == 0:
        print("No p2 rows found! Nothing to augment.")
        df.to_parquet(output_path)
        return
    
    # Swap p2 perspectives
    print("\nSwapping p2 perspectives...")
    start = time.time()
    df_p2_swapped = swap_perspectives(df_p2)
    elapsed = time.time() - start
    print(f"  Swapped {len(df_p2_swapped)} rows in {elapsed:.1f}s")
    
    # Combine: original p1 rows + swapped p2 rows
    print("\nCombining data...")
    df_combined = pd.concat([df_p1, df_p2_swapped], ignore_index=True)
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n  Original p1 rows:  {len(df_p1)}")
    print(f"  Swapped p2 rows:   {len(df_p2_swapped)}")
    print(f"  Combined total:    {len(df_combined)}")
    print(f"  Data increase:     {len(df_combined)/len(df_p1):.1f}x")
    
    # Verify all rows are now p1
    assert (df_combined['player_to_move'] == 'p1').all(), "Not all rows are p1!"
    
    # Save
    print(f"\nSaving to {output_path}...")
    df_combined.to_parquet(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double training data by swapping player perspectives")
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output", help="Output augmented parquet file")
    
    args = parser.parse_args()
    augment_parquet(args.input, args.output)
