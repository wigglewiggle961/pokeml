# Pitfalls Research (PokéML)

## Common Mistakes
1. **Feature Leakage**: Including `battle_winner` or `total_turns` in the features, allowing the model to "cheat" by seeing the future outcome.
2. **Species Fingerprinting**: Using names as features. In a small dataset, this is basically a Row ID.
3. **Normalization Drift**: Training on `Thunderbolt` and predicting on `thunderbolt`. Even 1 character difference breaks inference.
4. **Data Leakage in Time**: Failing to use `GroupShuffleSplit` (luckily PokéML already uses this).

## Prevention Strategies
- **Semantic Compression**: Always map names to stats.
- **Strict UAT**: Use `verify_parity.py` before every release.
- **Frequency Thresholds**: Never create a feature column for a move seen in only 1-2 battles.

## Phase Mapping
- **Phase 1**: Addresses Fingerprinting and Thresholds.
- **Phase 2**: Addresses Normalization Drift.
