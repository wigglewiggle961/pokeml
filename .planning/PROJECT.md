# PokéML

## What This Is
A Machine Learning pipeline for competitive Pokémon Showdown AI (Gen 9 OU). It processes replay data to train predictive models that suggest optimal moves and switches during live battles.

## Core Value
High-accuracy predictive strategy that generalizes across the competitive meta, rather than memorizing specific match fingerprints.

## Vision & Principles
- **Generalization Over Memorization**: Features should represent the *state* and *attributes* of the game, not match-specific IDs.
- **Parity is King**: Training and inference pipelines must share 100% identical pre-processing logic.
- **Transparency**: Model decisions should be traceable to game attributes (Types, Stats, HP bins).

## Requirements

### Validated
- ✓ [Dataset Parsing] — `process_replays.py` extracts flat state from raw battle logs.
- ✓ [Baseline Training] — LightGBM models achieve ~40% accuracy on Gen 9 OU moves.
- ✓ [Codebase Mapping] — Architecture and patterns are fully documented.

### Active
- [ ] [Overfitting Hardening] — Reduce the ~37% Train/Eval accuracy gap.
- [ ] [Semantic Representation] — Move from species names to Stats/Types for features.
- [ ] [Feature Pruning] — Implement frequency-based pruning for the 6,000+ binary move inputs.
- [ ] [Inference Parity] — Unify `sanitize_name` and normalization across all scripts.

### Out of Scope
- [Random Battles Support] — Project is focused specifically on Gen 9 OU competitive play.
- [Reinforcement Learning] — Current focus is on Supervised Learning from human replay data.

## Initial Context

### Background
The project has established a working pipeline but currently suffers from extreme overfitting (LogLoss gap ~1.9). Current efforts are focused on refining the "Medium" feature set.

### Specifics
- **Primary Data**: `30k.parquet` (parsed replay data).
- **Core Models**: LightGBM (Action Predictor), TensorFlow (Switch Predictor).
- **Top Goal**: Narrow the generalization gap to < 1.8 Eval LogLoss.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use LightGBM | High performance on sparse categorical data; handles NaNs naturally. | — Confirmed |
| Species-Agnostic Features | Prevents the model from using Species Names as unique match IDs (fingerprinting). | — Pending |
| Frequency-based Feature Pruning | Removes niche revealed moves that act as serial numbers for specific battles. | — Pending |

## Evolution
This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-04-12 after initialization via /gsd-new-project*
