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
| Judge models by margin above the prior-only floor (DEEP-01 Stage 0) | A zero-state per-species majority lookup scores 30.5% top-1 / 80.6% top-5 on Run 7's test split — raw accuracy mostly measures the species prior, not state reasoning. | ✅ Floor established 2026-07-05 (`baseline_prior_floors.py`, Run B0); Run 7's true margin is +8.0 pts top-1 |
| No class weighting for policy targets (DEEP-02 §1.3) | The model estimates a human policy; natural move frequencies are signal, not imbalance. `balanced` weights upweight rare moves ~100× and distort both argmax and calibration. | ✅ **Confirmed by Run 8 (2026-07-06): 46.3% top-1 vs Run 7's 38.5%, loss 2.10 vs 2.57 — new best model** |
| Bot winrate vs fixed baselines is the end-to-end KPI (DEEP-02 §5) | Offline accuracy can't detect compounding-drift failures or measure "plays well"; mirror-team battles vs Random/MaxBasePower/SimpleHeuristics do. | ✅ Harness written 2026-07-05 (`evaluate_bot_winrate.py`); first measurement pending local Showdown server |
| Adopt DEEP-09 speed items 1, 2, 5 before the Stage 3 grid (batch 1024 + LR 2e-3 defaults, feature cache in `data/cache/`, timings in metadata) | Stage 3 is ≥3 full runs; measured 8× step throughput at batch 1024 and 234s → ~seconds feature builds multiply directly. DEEP-09 §6: speedups must land *before* the experiment grids they multiply. | ✅ Landed 2026-07-06; cache verified bit-identical vs fresh build. Batch-size A/B vs Run 10 pending (first bs-1024 run doubles as it). Runs ≤10 were bs 256 / lr 1e-3 |

## Evolution
This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-07-05 (added prior-only-floor decision, DEEP-01 Stage 0)*
