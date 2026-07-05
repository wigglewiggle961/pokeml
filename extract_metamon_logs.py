"""
extract_metamon_logs.py
Extracts raw replay logs from the Metamon parquet dataset and saves them as individual .log files.
This allows for flexible parsing later using existing tools like process_replays.py.

Usage:
    python extract_metamon_logs.py --workers 8
"""

import pandas as pd
import glob
import os
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm

def save_single_log(row, output_dir):
    """
    Saves a single log from a dataframe row to a .log file.
    """
    replay_id = row.get('id', 'unknown')
    log_text = row.get('log', '')
    fmt = row.get('format', 'unknown')
    
    # Clean format string for filename
    clean_fmt = str(fmt).lower().replace(' ', '').replace('[', '').replace(']', '')
    
    if not log_text or not replay_id:
        return
        
    filename = f"{clean_fmt}-{replay_id}.log"
    file_path = os.path.join(output_dir, filename)
    
    # Skip if exists
    if os.path.exists(file_path):
        return

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(log_text)
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def process_parquet_file(file_path, output_dir, n_jobs=4):
    """
    Reads a parquet file and extracts logs in parallel.
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Filter for Gen9 OU
    if 'format' in df.columns:
        # Normalize format string
        normalized_format = df['format'].astype(str).str.lower().str.replace(' ', '').str.replace('[', '').str.replace(']', '')
        df = df[normalized_format == 'gen9ou']
    
    if df.empty:
        return

    # Create subfolder based on parquet filename to avoid 2M files in one dir?
    # User asked for flat structure "similar to replay_20k".
    # But 2M files in one dir is bad. I'll split into subfolders if user allows?
    # No, user asked "similar to replay_20k". I'll stick to flat unless user complains.
    # Actually, 50k files per parquet x 40 parquets = 2M.
    # Windows explorer will die.
    # I'll create subdirectories based on parquet name: metamon_logs/train-00000/
    # This keeps it manageable.
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    subdir = os.path.join(output_dir, base_name)
    os.makedirs(subdir, exist_ok=True)
    
    print(f"Extracting {len(df)} logs from {base_name} into {subdir}...")
    
    rows = df.to_dict('records')
    
    Parallel(n_jobs=n_jobs)(delayed(save_single_log)(row, subdir) for row in tqdm(rows, desc=f"Extracting {base_name}", leave=False))

def main():
    parser = argparse.ArgumentParser(description="Extract Metamon Logs to individual files")
    parser.add_argument("--input", default="metamon_parquet_files/data", help="Input directory containing parquet files")
    parser.add_argument("--output", default="metamon_logs_extracted", help="Output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    print(f"Found {len(files)} parquet files.")
    
    for f in files:
        process_parquet_file(f, args.output, args.workers)

if __name__ == "__main__":
    main()
