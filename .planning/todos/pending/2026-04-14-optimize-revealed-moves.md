---
created: 2026-04-14T06:54:21.491Z
title: Optimize revealed moves feature extraction
area: general
files:
  - train_action_predictor.py:680
---

## Problem

Processing ACTIVE `revealed_moves` features (Multi-Hot Encoding) takes extremely long. Identifying unique revealed moves and applying frequency pruning (`min_feature_replay_count=50`) iterates over ~713 unique moves against potentially millions of rows using regex string matching. This creates an O(N * M) operation which severely bottlenecks training and hogs CPU time.

## Solution

Replace the regex-based string match loop with Pandas vectorized string splitting and exploding:
1. Concatenate `p1` and `p2` revealed move strings.
2. Use `.str.split(',').explode()` to create a flat Series of all instantiated moves alongside their `replay_id`.
3. Use `.groupby('move')['replay_id'].nunique()` to compute replay frequency in a single pass.
This brings the time complexity down from hours to mere seconds.
