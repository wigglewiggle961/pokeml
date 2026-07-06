# Handoff: Implement DEEP-01 Stage 3 — The Species-Mode Ablation (the decisive experiment)

> Paste this file (or its path) into a fresh conversation. Written 2026-07-06 at the end of the
> session that shipped Stage 2 and logged Run 10. Supersedes `.planning/HANDOFF-stage2.md`
> (Stage 2 is done). CLAUDE.md and Claude memory carry general project context; this file
> carries what the next session needs to start building immediately.

## Why this stage is now the whole ballgame

Run 9 (attribute bundles) was +0.3. Run 10 (matchup features — the stage DEEP-01 predicted
would "matter most") was **flat**. Conclusion logged in experiments.md: *augmenting* the
identity model is exhausted — with species embeddings + usage columns on, the identity prior
saturates the softmax model and even brand-new matchup/speed-order information adds nothing.
Stage 3 asks the real question: can attributes + matchup features **replace** the identity
channels rather than decorate them? It produces the DEEP-04 deliverable numbers.

## Read first (in order)

1. `.planning/research/DEEP-01-representation-redesign.md` — **§4 is the spec** (measurement
   protocol + success criteria); §5 "Stage 3" is the work item; §1.1 lists the two identity
   leaks that `attrs_only` must drop; §2.3 has the two-tower rationale. Stages 0–2 are marked
   done in the doc with results.
2. `train_action_predictor_embedding.py` — everything lands here. Key anchors:
   `EMBED_COLS_SPECIES` (line ~56, all 14 species token columns incl. bench),
   `usage_cols = [c for c in all_cols if '_active_usage_' in c]` (line ~491),
   `embed_dim()` giving species 48 dims (line ~186), the GroupShuffleSplit block (~451–478),
   the metadata dict (~676).
3. `.planning/experiments.md` — Run 10 section has the full flat-result interpretation;
   run-table + artifacts-table logging conventions.
4. `matchup_features.py` / `species_data.py` — Stage 2/1 modules (no changes expected; just
   know they exist and are flag-gated in `feature_engineering.build_medium_X`).

## Current state (as of 2026-07-06)

| Run | What | Top-1 | Top-5 | Menu-restr. | Loss |
|---|---|---|---|---|---|
| B0 | Zero-state per-species majority lookup (floor) | 30.5% | 80.6% | — | — |
| 8 (v2) | Dropped balanced class weights | 46.3% | 89.5% | 51.7% | 2.104 |
| 9 (v3) | + species attribute bundles | 46.6% | 89.6% | 52.0% | 2.103 |
| 10 (v4) | + matchup features | 46.5% | 89.7% | 51.9% | 2.101 |

- Canonical run config: `data/100k.parquet --min_move_count 100` (277 classes; the twin
  `30k.parquet` has unsanitized move names — don't use it).
- All models judged by **margin above the 30.5% floor**, never raw accuracy.
- **Expectation for attrs_only, stated up front so nobody panics (DEEP-01 §4):** raw top-1
  will likely DROP — the species prior is genuinely predictive and we're removing it. Success
  is defined by §4's three criteria (floor margin, held-out-species degradation <30% relative,
  menu-restricted accuracy), not by beating 46.5%.

## The task (DEEP-01 §5 Stage 3)

### A. `--species_mode {embed, attrs_only, both}` flag

- **`embed`** (default) = current behavior. Note: **Run 10 (v4) already IS the embed arm** for
  plain-test-set comparison — only retrain it if/when the held-out-species split is in play
  (all arms must share the same split to be comparable).
- **`attrs_only`** = drop BOTH identity leaks (§1.1): exclude all `EMBED_COLS_SPECIES` columns
  from `categorical_embed_cols` (bench tokens too) AND exclude the `{p1,p2}_active_usage_*`
  columns from `numerical_for_model`. Cheapest implementation: filter at the feature-group
  identification step (~line 484-506) — the columns get built but not fed. (Passing
  `--disable_smogon_features` instead would also skip building usage cols and save RAM, but
  `species_data.py` reads the Smogon JSON independently, so attribute priors are unaffected
  either way. Filtering is simpler and keeps one code path.)
- **`both`** = two-tower / "prior late" (§2.3, §4): species embeddings stay but shrink to
  **8 dims** (not 48) and their flattened outputs bypass the trunk — concatenate them with the
  trunk output **after** `drop3`, immediately before the final softmax Dense. The prior can
  adjust logits but can't become the representation trunk. Implement via a `species_mode`
  param on `build_embedding_model` (split `embed_outputs` into trunk vs late lists).
- `attrs_only`/`both` should **require `--species_attrs`** (error out otherwise) — without the
  bundles the model would be species-blind. Run all arms with `--species_attrs
  --matchup_features` so the only difference between arms is the identity channel.
- **Artifact collision warning:** paths are `..._{version}_{suffix}`; three arms with the same
  `--version` overwrite each other. Either bake the mode into the version tag by hand
  (`--version v5-attrsonly`) or auto-append the mode to `version` when `species_mode != embed`
  (recommended — foolproof). Record `species_mode` in the metadata JSON regardless.

### B. Held-out-species eval (§4.2 — the headline metric)

- Choose ~10 **mid-usage** species (Smogon rank ~30–80, each present in a few % of replays —
  enough eval turns, not so common that excluding them guts the training set). Select
  deterministically and hard-code the list as the flag default so every arm uses the same set;
  record it in metadata.
- Split change: a replay is **holdout** if any of its 12 `p{1,2}_slot{i}_species` matches a
  held-out species (resolve names via `species_data._make_resolver` — the parquet has
  truncated names like `ambipo`). Holdout replays are excluded from train/val/test entirely
  and form a fourth eval set. The existing GroupShuffleSplit then runs on the remainder,
  unchanged.
- Report: accuracy on holdout turns where the **acting mon is a held-out species**
  (`p1_active_species` — training rows are `player_to_move == 'p1'`; DEEP-01 §4.1 note), plus
  the same-model in-vocab test accuracy, plus the relative degradation Δ. Embed arm should
  collapse toward `__UNKNOWN__`/floor behavior; attrs arms should degrade gracefully (<30%
  relative loss = success criterion (ii)).
- Label-encoder caveat: fit the label encoder on the FULL filtered data (before holdout
  removal) so holdout-turn labels are encodable; a holdout mon's signature move may otherwise
  be missing from the class set.

### C. Eval extras (both are eval-only; a separate post-hoc script is fine)

- **Frequency-slope curve (§4.4):** per-species accuracy vs species training frequency,
  log-binned. Steep slope = memorization. One chart (or just the binned table printed/saved).
- **Counterfactual identity probe (§4.5, embed/both arms only):** on fixed test states, swap
  the species token holding attributes constant; mean KL shift of the output. High shift =
  identity still drives predictions. Natural home: a new `evaluate_species_generalization.py`
  that loads a saved model + its `_artifacts_` joblib — keeps the trainer lean.

## Implementation facts the last sessions learned (will save you time)

- **`prepare_inputs` now zero-fills non-finite numericals** — rows with no flagged active mon
  carry NaN boosts (np.select default in `build_medium_X`); one such row NaNs the loss. Fixed
  2026-07-06; don't re-introduce a raw-`.values` path.
- The smoke-test dataset is `data/test.parquet` (25k rows, includes NaN-boost rows — good
  canary). Pattern: `python train_action_predictor_embedding.py data/test.parquet
  --species_attrs --matchup_features --species_mode attrs_only --epochs 2 --version smoketest`
  → **delete `models/*smoketest*` after**. Smoke every arm incl. `both` (it changes the model
  graph — check the summary shows the 8-dim species embeddings joining after drop3).
- `build_vocab_encoders` groups species columns via `EMBED_COLS_SPECIES` membership; if you
  filter species cols out of `categorical_embed_cols`, everything downstream (vocab, inputs,
  model) follows automatically — that's the designed seam.
- `'absent'`/`'unknown'` species values are empty-slot placeholders, not bugs.
- **Do NOT launch the full training runs** — the user runs those himself. Leave ready
  commands logged in experiments.md as ⏳ rows, e.g.:
  `python train_action_predictor_embedding.py data/100k.parquet --min_move_count 100
  --species_attrs --matchup_features --species_mode attrs_only --version v5`
  (×3 arms; decide the holdout-split flag story and document it in the same rows).
- Results land in `models/action_tf_embedding_metadata_{version}_medium.json` — read metrics
  from there after runs; no logs needed.
- Optional knob if attrs_only shows base-stat memorization (§2.3 caveat): light Gaussian
  noise/dropout on the attribute inputs. Don't build it preemptively.
- **DEEP-09 items 1, 2, 5 landed 2026-07-06 (after Run 10):**
  - Defaults are now **batch 1024 / lr 2e-3** (Runs ≤10 were 256 / 1e-3). Measured 8× step
    throughput. **The A/B validating this is still pending** — the Stage 3 `embed` arm at the
    new defaults vs Run 10 (same features, old defaults) doubles as it; call that out when
    interpreting. If the embed arm lands far off 46.5%, suspect the batch change before
    suspecting the features.
  - **Feature cache** (`data/cache/<key>/`, DEEP-09 §1): `load_or_build_medium_features` in
    `feature_engineering.py` wraps load+filter+build; the trainer uses it. Auto-invalidates on
    any edit to feature_engineering/species_data/matchup_features .py, dataset mtime, or build
    params (incl. `min_move_count`). Verified bit-identical vs fresh build. `--no_cache`
    bypasses; `--prune_cache N` evicts to newest N (entries ~1–3 GB on 100k).
    **Stage 3 note:** all three arms share one cache entry per (dataset, flags) — species_mode
    is a *model*-side choice, not a build param, so the ablation grid pays the build once. The
    held-out-species split also happens after loading, so it doesn't fragment the cache either.
  - `timings` dict + `test_clean_ce` now in the metadata JSON. **Report clean CE for every
    Stage 3 arm** (DEEP-04: it's the per-run proxy metric; top-5 is saturated/retired).

## Cross-cutting debts from the other DEEP docs (decide, don't drift)

Stage grinding in the DEEP-01 lane left these open; each was flagged as high-value in its doc.
Don't silently skip them again — either pull them in or record the deferral:

1. **DEEP-05 §0.2 — 14 duplicate species embedding tables** (each species column gets its own
   `emb_{col}` table off one shared vocab; same duplication for status/HP/type). Called "the
   single highest-value/lowest-risk architecture fix available" (~1 day). **Decision needed
   before the ablation:** fixing it changes the embed arm's capacity; not fixing it means the
   ablation compares attrs against a handicapped identity baseline. Recommendation: land shared
   tables first (it touches `build_embedding_model` exactly where the two-tower work goes
   anyway), accept that Stage 3 arms are then compared *to each other*, not to Runs 8–10.
2. **DEEP-04 — the rest of the dashboard.** Clean CE is in (2026-07-06); the 8-number
   dashboard, `evaluate_action_model.py`, frozen baselines, and the winrate north star are
   not. The Stage 3 eval extras (frequency slope, counterfactual probe) are DEEP-04
   deliverables — build them as the start of `evaluate_action_model.py`, not throwaways.
3. **DEEP-06 §0.3 — the usage prior is the 0-cutoff (all-ratings) Smogon file.** One-file swap
   to the 1695-cutoff, pinned month. Touches the attribute priors AND the usage columns being
   ablated — do it *between* ablation rounds, never mid-grid (it invalidates the cache and the
   comparison).
4. **DEEP-03 F1 / DEEP-08 — the live bot almost certainly plays random moves** (three
   independent live-path defects). Consciously parked by Eric ("offline first"), but the
   winrate KPI is unmeasurable until fixed. Revisit the parking decision once Stage 3 settles
   the representation question — that's the natural "wire a real model into the bot" moment
   (DEEP-01 §5 Stage 5 note agrees).
5. **DEEP-09 §5 — the fast tier (`C1-v1-fast10`)** is not built. If Stage 3 spawns follow-up
   sweeps (noise levels, embedding dims), build the 10% replay-subsampled tier first instead
   of paying full-tier costs per hypothesis.

## Success criteria (DEEP-01 §4, verbatim intent)

An `attrs_only` or `both` model that (i) beats Floor B by more than the embed arm does,
(ii) loses <30% relative accuracy on held-out species vs the embed arm's expected collapse,
and (iii) improves menu-restricted accuracy — **even if raw top-1 lands under 46.5%**.

## Bookkeeping when done (per CLAUDE.md — the last three sessions' pattern)

- experiments.md: one ⏳ row per arm + artifacts-table rows when results exist + a Run-N
  section interpreting the three-arm comparison against the §4 criteria.
- DEEP-01 §5 Stage 3 bullet: mark ✅ IMPLEMENTED with results note. This stage produces the
  DEEP-04 deliverables — note the numbers there too if DEEP-04 has a results section.
- `.planning/STATE.md`: append a dated status paragraph.
- `PROJECT_CONTEXT.md`: update the trainer row in Key Files (new flags) + any new eval script.
- `.planning/PROJECT.md` Key Decisions: if the ablation settles the representation question,
  add the "species-ID basis → attribute basis" decision row (DEEP-01 §6 asks for exactly this).
- Claude memory: update `pokeml-project-status.md`.
- After Stage 3, next milestones per DEEP-01 §5: Stage 4 (move semantics) and Stage 5
  (candidate scorer) — but the Stage 3 numbers decide whether/how to proceed.
