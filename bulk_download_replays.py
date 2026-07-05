"""
bulk_download_replays.py
Downloads Pokemon Showdown replays by iterating through sequential IDs.

Usage:
    python bulk_download_replays.py --format gen9ou --start_id 2531300000 --count 50000 --output replays/
"""

import requests
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://replay.pokemonshowdown.com"

def download_replay(replay_id: str, output_dir: str) -> tuple[str, bool]:
    """Download a single replay log."""
    url = f"{BASE_URL}/{replay_id}.log"
    output_path = os.path.join(output_dir, f"{replay_id}.log")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        return replay_id, True
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and len(response.text) > 100:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            return replay_id, True
        else:
            return replay_id, False
    except Exception as e:
        return replay_id, False


def get_recent_replays(format_name: str, page: int = 1) -> list[dict]:
    """Get recent replays from the search API."""
    url = f"{BASE_URL}/search.json?format={format_name}&page={page}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def find_id_range(format_name: str) -> tuple[int, int]:
    """Find the approximate ID range for recent replays."""
    replays = get_recent_replays(format_name, page=1)
    if not replays:
        return 0, 0
    
    # Extract numeric IDs
    ids = []
    for r in replays:
        rid = r.get('id', '')
        if '-' in rid:
            try:
                num_id = int(rid.split('-')[1])
                ids.append(num_id)
            except:
                pass
    
    if ids:
        return min(ids), max(ids)
    return 0, 0


def bulk_download(format_name: str, start_id: int, count: int, output_dir: str, workers: int = 10):
    """Download replays by iterating through IDs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate replay IDs to try
    replay_ids = [f"{format_name}-{start_id + i}" for i in range(count)]
    
    print(f"Attempting to download {count} replays...")
    print(f"ID range: {format_name}-{start_id} to {format_name}-{start_id + count - 1}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {workers}")
    print()
    
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_replay, rid, output_dir): rid for rid in replay_ids}
        
        with tqdm(total=len(replay_ids), desc="Downloading") as pbar:
            for future in as_completed(futures):
                replay_id, success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix({"success": successful, "failed": failed})
    
    print(f"\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed/Not found: {failed}")
    print(f"Success rate: {successful/count*100:.1f}%")


def download_from_api(format_name: str, pages: int, output_dir: str, min_rating: int = 0):
    """Download replays discovered via the search API."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_replays = []
    print(f"Fetching replay list from API ({pages} pages)...")
    
    for page in tqdm(range(1, pages + 1), desc="Fetching pages"):
        replays = get_recent_replays(format_name, page)
        for r in replays:
            if r.get('rating', 0) >= min_rating:
                all_replays.append(r['id'])
        time.sleep(0.5)  # Be nice to the server
    
    print(f"Found {len(all_replays)} replays (rating >= {min_rating})")
    
    # Download them
    successful = 0
    for replay_id in tqdm(all_replays, desc="Downloading"):
        _, success = download_replay(replay_id, output_dir)
        if success:
            successful += 1
        time.sleep(0.1)  # Rate limiting
    
    print(f"\nDownloaded {successful} / {len(all_replays)} replays")


def get_replays_before(format_name: str, before_timestamp: int) -> list[dict]:
    """Get replays from before a specific timestamp."""
    url = f"{BASE_URL}/search.json?format={format_name}&before={before_timestamp}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []


def download_time_based(format_name: str, target_count: int, output_dir: str, 
                        min_rating: int = 0, start_before: int = None):
    """
    Download replays using time-based pagination (unlimited!).
    Uses 'before=timestamp' to keep fetching older replays.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Start from now if no timestamp provided
    if start_before is None:
        start_before = int(time.time())
    
    print(f"=" * 60)
    print(f"TIME-BASED REPLAY DOWNLOAD")
    print(f"=" * 60)
    print(f"Format: {format_name}")
    print(f"Target: {target_count} replays")
    print(f"Min rating: {min_rating}")
    print(f"Starting before timestamp: {start_before}")
    print(f"Output: {output_dir}")
    print()
    
    collected_ids = []
    current_before = start_before
    batch_num = 0
    
    # Keep fetching until we have enough
    with tqdm(total=target_count, desc="Collecting replay IDs") as pbar:
        while len(collected_ids) < target_count:
            batch_num += 1
            replays = get_replays_before(format_name, current_before)
            
            if not replays:
                print(f"\nNo more replays found (reached end of history)")
                break
            
            # Filter by rating and collect IDs
            for r in replays:
                rating = r.get('rating') or 0  # Handle None ratings
                if rating >= min_rating:
                    collected_ids.append(r['id'])
                    pbar.update(1)
                    if len(collected_ids) >= target_count:
                        break
            
            # Get the oldest timestamp from this batch for next iteration
            oldest_time = min(r.get('uploadtime', current_before) for r in replays)
            
            # If we're not making progress, break
            if oldest_time >= current_before:
                print(f"\nStuck at timestamp {oldest_time}, breaking")
                break
            
            current_before = oldest_time
            time.sleep(0.3)  # Rate limiting
    
    print(f"\nCollected {len(collected_ids)} replay IDs")
    print(f"Oldest timestamp reached: {current_before}")
    
    # Now download them
    print(f"\nDownloading replays...")
    successful = 0
    skipped = 0
    
    for replay_id in tqdm(collected_ids, desc="Downloading"):
        output_path = os.path.join(output_dir, f"{replay_id}.log")
        if os.path.exists(output_path):
            skipped += 1
            continue
        _, success = download_replay(replay_id, output_dir)
        if success:
            successful += 1
        time.sleep(0.05)  # Faster since we know these exist
    
    print(f"\nDownload complete!")
    print(f"New downloads: {successful}")
    print(f"Already existed (skipped): {skipped}")
    print(f"Total in folder: {successful + skipped}")
    
    # Save the last timestamp for resuming later
    resume_file = os.path.join(output_dir, "_resume_timestamp.txt")
    with open(resume_file, 'w') as f:
        f.write(str(current_before))
    print(f"Resume timestamp saved to: {resume_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk download Pokemon Showdown replays")
    parser.add_argument("--format", type=str, default="gen9ou", help="Battle format")
    parser.add_argument("--output", type=str, default="replays", help="Output directory")
    parser.add_argument("--mode", choices=['sequential', 'api', 'timetravel'], default='timetravel',
                        help="Download mode: 'sequential' (iterate IDs), 'api' (page-based), 'timetravel' (time-based, unlimited)")
    
    # Sequential mode options
    parser.add_argument("--start_id", type=int, default=None, help="Starting ID for sequential mode")
    parser.add_argument("--count", type=int, default=10000, help="Number of replays to download")
    parser.add_argument("--workers", type=int, default=10, help="Parallel download workers (sequential mode)")
    
    # API/timetravel mode options
    parser.add_argument("--pages", type=int, default=100, help="Number of pages to fetch (API mode)")
    parser.add_argument("--min_rating", type=int, default=1200, help="Minimum ELO rating filter")
    parser.add_argument("--resume", action='store_true', help="Resume from saved timestamp (timetravel mode)")
    parser.add_argument("--before", type=int, default=None, help="Start before this timestamp (timetravel mode)")
    
    args = parser.parse_args()
    
    if args.mode == 'sequential':
        if args.start_id is None:
            print("Finding current ID range...")
            min_id, max_id = find_id_range(args.format)
            if max_id > 0:
                args.start_id = max_id - args.count
                print(f"Auto-detected range: starting from {args.start_id}")
            else:
                print("Could not auto-detect ID range. Please specify --start_id")
                exit(1)
        
        bulk_download(args.format, args.start_id, args.count, args.output, args.workers)
    
    elif args.mode == 'api':
        download_from_api(args.format, args.pages, args.output, args.min_rating)
    
    elif args.mode == 'timetravel':
        # Check for resume
        start_before = args.before
        if args.resume:
            resume_file = os.path.join(args.output, "_resume_timestamp.txt")
            if os.path.exists(resume_file):
                with open(resume_file, 'r') as f:
                    start_before = int(f.read().strip())
                print(f"Resuming from saved timestamp: {start_before}")
            else:
                print("No resume file found, starting from current time")
        
        download_time_based(args.format, args.count, args.output, args.min_rating, start_before)
