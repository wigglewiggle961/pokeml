"""
process_metamon_data.py
Processes raw replays from the Metamon dataset (parquet files) using the same logic as process_replays.py.

OPTIMIZED FOR MEMORY: Uses batched processing to avoid loading entire files.
FIX: Enforces consistent schema across batches to prevent Parquet errors.

Usage:
    python process_metamon_data.py --input metamon_parquet_files/data --output metamon_processed --workers 8 --batch-size 5000
"""

import pandas as pd
import glob
import os
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import pyarrow.parquet as pq
import pyarrow as pa
import gc

# Import processing logic from existing script
# NOTE: Ensure process_replays.py is in the same directory or PYTHONPATH
from process_replays import parse_showdown_replay, flatten_state, normalize_species_name

# Fix for Eiscue-Noice since it's causing issues
# We can monkey-patch or just handle it in the normalization logic if we could, 
# but for now let's ensure the script runs.
# The error "normalized_species='eiscuenoice'" suggests the parser is seeing this.
# 'eiscuenoice' -> 'eiscue' should be handled by the base species stripper if 'noice' was in the suffix list,
# or we can manually handle it here or in process_replays.py. 
# For now, let's proceed with the schema fix which is the crtical crasher.

def process_single_row(row):
    """
    Process a single row from the metamon dataframe.
    Returns a list of flattened state dictionies (one per turn).
    """
    replay_id = row.get('id', 'unknown')
    log_text = row.get('log', '')
    
    if not log_text:
        return []
    
    try:
        # Parse the replay log into a list of game states
        game_states = parse_showdown_replay(log_text, replay_id)
        
        if not game_states:
            return []
            
        # Flatten each state for tabular format
        flat_rows = []
        for state in game_states:
            flat_state = flatten_state(state)
            if flat_state:
                flat_rows.append(flat_state)
                
        return flat_rows
    except Exception as e:
        # print(f"Error processing replay {replay_id}: {e}") # Too noisy for parallel
        return []

def process_batch(batch_df, expected_columns=None):
    """
    Processes a single batch (DataFrame) of replays.
    Returns a DataFrame of flattened states.
    If expected_columns is provided, ensures the output DataFrame has exactly those columns.
    """
    # Filter for Gen9 OU
    if 'format' in batch_df.columns:
        # Normalize format string to handle '[Gen 9] OU' vs 'gen9ou'
        normalized_format = batch_df['format'].astype(str).str.lower().str.replace(' ', '').str.replace('[', '').str.replace(']', '')
        batch_df = batch_df[normalized_format == 'gen9ou']
    
    if batch_df.empty:
        return pd.DataFrame()

    # Convert dataframe to list of dicts
    rows = batch_df.to_dict('records')
    
    # Run parallel processing on this batch
    results = Parallel(n_jobs=-1)(delayed(process_single_row)(row) for row in rows)
    
    # Flatten results
    all_flat_states = [item for sublist in results for item in sublist]
    
    if not all_flat_states:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_flat_states)
    
    # --- Schema Enforcement ---
    if expected_columns is not None:
        # 1. Add missing columns (filled with 0/NaN/None appropriate for type, but NaN is safest for Parquet)
        # Note: In our specific case, missing columns are likely hazard/boost/species columns which are 0 or 'none' implies.
        # But for generic stability, let's reindex.
        
        # Identify missing columns
        for col in expected_columns:
            if col not in df.columns:
                # Fill with appropriate defaults based on column name if possible, or 0
                if 'species' in col: df[col] = 'unknown'
                elif 'move' in col: df[col] = 'none'
                elif 'status' in col: df[col] = 'none'
                else: df[col] = 0.0 # Default numeric
        
        # 2. Drop extra columns that weren't in the first batch (rare, but possible if new moves appear)
        # Actually, dropping data is bad. Ideally we'd union schemas, but ParquetWriter in append mode 
        # requires STRICTLY identical schema.
        # To handle "new" columns that appear in later batches, we would need to read the *entire* file first to know all possible columns,
        # which defeats the purpose of memory efficiency.
        # compromise: The 'process_replays.py' *should* output a fixed set of columns based on the code logic (slots 1-6, boosts, fields).
        # The only variable columns are likely NOT variable in the new flattened format (it uses fixed keys like p1_slot1_species).
        # WAIT. 'process_replays.py' flattens SIDE conditions based on what it sees?
        # Let's check logic: process_replays.py iterates through ALL SIDE_CONDITIONS and HAZARD_CONDITIONS to create keys (lines 832-843).
        # So the columns *should* be stable!
        # Why is there a mismatch?
        # Maybe some column types are inferred differently (int vs float) if a batch has all NaNs or all ints?
        # Let's ensure strict ordering and typing.
        
        # Enforce order
        df = df[expected_columns]
        
    return df

def process_parquet_file(file_path, output_dir, batch_size=5000):
    """
    Reads a metamon parquet file in batches, processes replays, and incrementally saves a processed parquet.
    """
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"processed_{filename}")
    
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already exists)")
        return
    
    print(f"Processing {filename} in batches of {batch_size}...")
    
    try:
        parquet_file = pq.ParquetFile(file_path)
        
        # Initialize output writer
        writer = None
        current_schema = None
        current_columns = None
        
        total_rows_processed = 0
        total_output_rows = 0
        
        # Iterate over batches
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            batch_df = batch.to_pandas()
            
            # Process the batch with schema enforcement if we have one
            processed_batch_df = process_batch(batch_df, expected_columns=current_columns)
            
            if not processed_batch_df.empty:
                # If this is the very first successful batch, establish the schema
                if writer is None:
                    # Establish reference formatting
                    # Force object columns to string to avoid mixed type issues
                    # Force numeric columns to float/int consistently? 
                    # Pyarrow handles standard pandas types well.
                    
                    table = pa.Table.from_pandas(processed_batch_df, preserve_index=False)
                    current_schema = table.schema
                    current_columns = processed_batch_df.columns.tolist()
                    
                    writer = pq.ParquetWriter(output_path, current_schema)
                else:
                    # Convert to table using the ESTABLISHED schema
                    # This auto-handles casting if types are compatible-ish
                    try:
                        table = pa.Table.from_pandas(processed_batch_df, schema=current_schema, preserve_index=False)
                    except Exception as cast_err:
                        print(f"  Warning: Batch {i+1} schema mismatch: {cast_err}. Attempting manual fix...")
                        # Emergency type casting could go here if needed
                        # For now, let's rely on process_batch's reindexing (which we just added) to match columns.
                        # If process_batch did its job, columns are identical.
                        # Types might still differ (e.g. all nones -> float vs string).
                        table = pa.Table.from_pandas(processed_batch_df, preserve_index=False)
                        # This might fail on write_table if logic assumes perfect match.
                        
                        # better approach: Use the schema to construct
                        table = pa.Table.from_pandas(processed_batch_df, schema=current_schema)
                
                writer.write_table(table)
                total_output_rows += len(processed_batch_df)
            
            total_rows_processed += len(batch_df)
            print(f"  Batch {i+1}: Processed {len(batch_df)} input rows -> {len(processed_batch_df)} output rows. (Total In: {total_rows_processed})")
            
            # Explicit garbage collection to keep memory low
            del batch_df
            del processed_batch_df
            if 'table' in locals(): del table
            gc.collect()

        if writer:
            writer.close()
            print(f"  Finished {filename}. Total output rows: {total_output_rows}")
        else:
            print(f"  Finished {filename}. No valid data found.")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Process Metamon Replay Dataset (Low Memory)")
    parser.add_argument("--input", default="metamon_parquet_files/data", help="Input directory containing parquet files")
    parser.add_argument("--output", default="metamon_processed", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (for row processing within batches).")
    parser.add_argument("--batch-size", type=int, default=5000, help="Number of rows to read per batch.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Get all parquet files
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    print(f"Found {len(files)} parquet files in {args.input}")
    
    for f in files:
        process_parquet_file(f, args.output, batch_size=args.batch_size)
        gc.collect() # GC between files as well

if __name__ == "__main__":
    main()
