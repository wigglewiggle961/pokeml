# Handoff: Implement DEEP-01 Stage 2 — Relational / Matchup Features

> Paste this file (or its path) into a fresh conversation. Written 2026-07-06 at the end of the
> session that shipped Stage 1. CLAUDE.md and Claude memory carry the general project context;
> this file carries what the next session needs to start building immediately.

## Read first (in order)

1. `.planning/research/DEEP-01-representation-redesign.md` — **§2.2 is the spec for this task**;
   §5 "Stage 2" is the work item. Stages 0–1 are marked done in the doc with results.
2. `species_data.py` — Stage 1's module. Stage 2 builds on its tables (`load_static_data`,
   `build_species_attribute_table`, the `_attr_` column pattern, `_make_resolver`).
3. `feature_engineering.py:build_medium_X` — where the new features get wired (pattern: the
   `attach_species_attrs` flag added for Stage 1; do the same with a `matchup_features` flag).
4. `train_action_predictor_embedding.py` — CLI flags / metadata pattern (`--species_attrs`,
   `--version`), and `is_attr_type_col`/`is_attr_ability_col` showing how dynamic columns are
   routed to embedding groups. Matchup features are **all numeric** — they just flow through
   `numerical_features` into the scaler; no embedding wiring needed.
5. `.planning/experiments.md` — run history and logging conventions.

## Current state (as of 2026-07-06)

| Run | What | Top-1 | Top-5 | Menu-restr. | Loss |
|---|---|---|---|---|---|
| B0 | Zero-state per-species majority lookup (floor) | 30.5% | 80.6% | — | — |
| 8 (v2) | Dropped balanced class weights | 46.3% | 89.5% | 51.7% | 2.104 |
| 9 (v3) | + species attribute bundles (`--species_attrs`) | 46.6% | 89.6% | 52.0% | 2.103 |

- Canonical run config: `data/100k.parquet --min_move_count 100` (277 classes; this dataset has
  **sanitized** move names — its twin `30k.parquet` has display names, don't use it).
- All models judged by **margin above the 30.5% floor**, never raw accuracy.
- Run 9's +0.3 was the predicted small bump: with species embeddings + usage columns still on,
  attributes are near-redundant for raw accuracy. **Stage 2 is the stage DEEP-01 expects to
  actually move the numbers** — the model currently has zero matchup/speed-order information.

## The task (DEEP-01 §2.2, priority order)

Implement `build_matchup_features(X)` (new function — natural home: `species_data.py` or a new
`matchup_features.py`; wire into `build_medium_X` behind a flag) producing, per row:

1. **Type effectiveness, both directions:** `p2_best_revealed_eff_vs_p1` = max over p2's revealed
   damaging moves of effectiveness(move type → p1 active typing, tera-aware), and
   `p2_best_revealed_dmg_proxy` = max of eff × basePower × STAB(1.5) × attack-side match (uses
   `atk_bias` from the attribute table). Mirror both for p1 vs p2. Four columns.
2. **Speed order:** `speed_edge` = sign + continuous ratio of adjusted speeds, using the attribute
   table's `speed_est` adjusted by boost multipliers (`p{1,2}_active_boost_spe`), paralysis ×0.5
   (`p{1,2}_active_status`), Tailwind ×2 (`p{1,2}_side_tailwind`), and flipped under Trick Room
   (`field_pseudo_weather`).
3. **KO pressure (crude is fine):** `p{1,2}_can_likely_ko_*` — dmg_proxy scaled against defender
   bulk (`phys_bulk`/`spec_bulk`) and current HP bin, thresholded.
4. **Team coverage (only if cheap):** `p1_has_safe_switchin`, `p2_walled_count` from bench attr
   types vs revealed moves.

## Implementation facts the last session learned (will save you time)

- **Static data is vendored at `data/static/`** (gen9pokedex/gen9moves/gen9typechart/natures
  .json, copied from the poke-env install). `data/gen9ou-0.json` is the Smogon stats.
  **`gen9typechart.json` is so far unused/uninspected — check its exact schema first** (damage
  multiplier encoding conventions vary: 0/1/2/3 codes vs floats).
- **Ordering constraint in `build_medium_X`:** the raw revealed-move strings
  (`p{1,2}_active_revealed_moves_str`, comma-separated) are **dropped** partway through the
  function (see `cols_to_drop`). Matchup features need those strings (or the parsed move lists)
  plus the active species/tera columns — so compute matchup features **before** that drop, or
  refactor the drop point. Don't rebuild moves from the pruned multi-hot columns (they lose
  niche moves).
- Species values in the parquet are sanitized but contain **parser-truncated names**
  (`ambipo` → Ambipom, cosmetic Alcremie formes). Always resolve via
  `species_data._make_resolver(table.index)`; `'absent'`/`'unknown'` are empty-slot
  placeholders, not bugs. Join coverage should stay ≥99.9% — print unmatched like Stage 1 does.
- Tera: `p{1,2}_active_tera_type` (capitalized type names, e.g. `Water`) +
  `p{1,2}_active_terastallized` (0/1). When terastallized, defensive typing = the tera type
  alone (mono); otherwise pokedex `type1`/`type2`.
- Move names in `revealed_moves_str` may need `sanitize_name` before lookup in `gen9moves.json`
  (whose keys are already sanitized). "Damaging move" = `category != 'Status'`.
- HP bins are categorical strings after `bin_hp` (`Full/High/Middle/Low/Critical/Sash/Fainted`)
  — map to rough fractions for the KO-pressure calc.
- **Validation pattern from Stage 1** (repeat it): a `__main__` CLI in the module that builds the
  features against `data/100k.parquet` and prints distributions/sanity checks (e.g. eff values
  only in {0, .25, .5, 1, 2, 4}; speed_edge flips under Trick Room rows), then an end-to-end
  smoke test: `python train_action_predictor_embedding.py data/test.parquet --species_attrs
  --matchup_features --epochs 2 --version smoketest` → **delete `models/*smoketest*` after**.
- **Do NOT launch the full training run** — the user runs those themselves. Leave a ready
  command as Run 10: `python train_action_predictor_embedding.py data/100k.parquet
  --min_move_count 100 --version v4 --species_attrs --matchup_features` (new `--matchup_features`
  flag; record it in the metadata JSON like `species_attrs` is).
- Results conveniently land in `models/action_tf_embedding_metadata_v4_medium.json` — read
  metrics from there after the user runs it; no logs needed.

## Bookkeeping when done (per CLAUDE.md — the last two sessions' pattern)

- experiments.md: Run 10 row (⏳ ready-to-run) + artifacts-table row when results exist.
- DEEP-01 §5 Stage 2 bullet: mark ✅ IMPLEMENTED with a short results note.
- `.planning/STATE.md`: append a dated status paragraph.
- `PROJECT_CONTEXT.md`: update the Key Files table (new module/flags).
- Claude memory: update `pokeml-project-status.md`.
- After Run 10, the next milestone is **Stage 3**: `--species_mode {embed, attrs_only, both}`
  ablation + held-out-species eval — that's where the attribute basis proves itself.
