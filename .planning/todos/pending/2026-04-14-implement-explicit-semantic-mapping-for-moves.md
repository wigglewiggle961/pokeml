---
created: 2026-04-14T08:33:51.572Z
title: Implement Explicit Semantic Mapping for Moves
area: general
files:
  - train_action_predictor.py
---

> **⚠️ Superseded in scope by [DEEP-01](../../research/DEEP-01-representation-redesign.md)
> (2026-07-05).** The "do NOT map the species name" restriction below applied to the now-frozen
> LightGBM pipeline only — DEEP-01 proposes exactly that mapping for the embedding pipeline.
> The move-semantics idea itself lands as DEEP-01 Stage 4; close this todo when that ships.

## Problem

Currently, the model uses multi-hot encoding for `revealed_moves` (e.g., creating 150+ sparse binary columns for each move like `has_flamethrower=1`). This sparse representation causes LightGBM to struggle with overfitting, as it attempts to build deep, fragmented trees on highly sparse generic data rather than learning cohesive strategies. Embeddings (Move2Vec) would solve this but are complex to implement. 

## Solution

Replace the multi-hot encoded move names with **Explicit Semantic Mapping** using the local `poke-env` database.
Instead of adding columns for the move *names*, tally the mechanical properties of the opponent's revealed moves.
*   IMPORTANT: **Do NOT map the `species` name.** The user explicitly stated to keep the species as raw IDs/names and only apply this semantic mapping to the moves.
*   Example columns to generate: `revealed_physical_moves_count`, `revealed_special_moves_count`, `revealed_status_moves_count`, `revealed_fire_type_count`, `average_move_power`, etc.
*   This will collapse hundreds of sparse columns into ~20 dense, highly actionable numerical columns, significantly aiding the decision tree's logic without guessing via embeddings.
