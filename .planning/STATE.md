# Project State

## Current Milestone
Milestone 1: Generalization & Parity

## Current Phase
Phase 5: Hyperparameter Tuning

## Status
COMPLETED Phase 4. Next: Phase 5 (Hyperparameter Tuning).

2026-07-05: DEEP-01 (representation redesign) research doc landed; its **Stage 0 is done** —
prior-only floors computed (`baseline_prior_floors.py`, Run B0 in experiments.md: 30.5% top-1 /
80.6% top-5 zero-state vs Run 7's 38.5% / 84.5%), and Run 7's dataset lineage recovered
(`data/100k.parquet` + `--min_move_count 100`). Next DEEP-01 step: Stage 1 (species attribute table).

2026-07-05 (later): DEEP-02 first slice implemented: (1) class weighting dropped by default in
`train_action_predictor_embedding.py` + `--version` flag + lineage in metadata + menu-restricted
top-1 metric; (2) `evaluate_bot_winrate.py` harness written (needs a local pokemon-showdown
server to run).

2026-07-06: **Run 8 executed — new best model** (`action_tf_embedding_v2_medium.keras`):
46.3% top-1 / 89.5% top-5 / menu-restricted 51.7% / loss 2.104, vs Run 7's 38.5%/84.5%/2.567.
Margin over the Floor B lookup nearly doubled (+8.0 → +15.8 pts). Winrate harness deferred
(user decision) — offline track first. Note the live bot still runs the old v4 LightGBM.

2026-07-06 (later): **DEEP-01 Stage 1 implemented** — `species_data.py` (attribute table:
stats/types/speed_est/modal ability/item-class/role priors from vendored `data/static/` JSONs +
Smogon), wired into `build_medium_X(attach_species_attrs=...)` / `--species_attrs`. Join
validated (99.95%+ coverage; resolver handles parser-truncated names). Smoke-tested end-to-end.
**Run 9 done (2026-07-06): 46.6% / 89.6% / menu-restricted 52.0%** — +0.3 pts vs Run 8, the
predicted validation-level bump (identity channels still on, so attributes are near-redundant
for raw accuracy). Next: Stage 2 (relational/matchup features — expected to matter most), then
Stage 3 (`--species_mode` ablation + held-out-species eval, where the attribute basis proves out).
**Session handoff for Stage 2: `.planning/HANDOFF-stage2.md`** (spec pointer, current numbers,
implementation gotchas, ready-to-run commands).

2026-07-06 (Stage 2 session): **DEEP-01 Stage 2 implemented** — `matchup_features.py`
(12 relational columns: tera-aware type effectiveness + damage proxy both directions,
speed_edge/speed_ratio with boosts/paralysis/Tailwind/Trick-Room, crude KO pressure, p1 bench
coverage), wired into `build_medium_X(matchup_features=...)` / `--matchup_features`. Validated
on the full 100k parquet + smoke-tested end-to-end. Also fixed a latent NaN-loss bug
(`prepare_inputs` now zero-fills non-finite numericals — rows with no flagged active mon carry
NaN boosts). **Run 10 executed same day: 46.5% / 89.7% / menu-restricted 51.9% / loss 2.101 —
FLAT vs Run 9.** The stage DEEP-01 expected to move the numbers moved nothing while species
embeddings + usage columns stay on — augmenting the identity model is exhausted (see the Run 10
interpretation in experiments.md). **Next: Stage 3 is now the decisive experiment** —
`--species_mode {embed, attrs_only, both}` ablation (attrs_only drops species embeddings AND
usage columns) + held-out-species eval; success criteria in DEEP-01 §4.
**Session handoff for Stage 3: `.planning/HANDOFF-stage3.md`** (supersedes HANDOFF-stage2.md).

2026-07-06 (still later): after an explicit "are we straying from the deep analysis?" audit of
all nine DEEP docs: the stage-grind IS DEEP-01's own plan (intentional), but cheap cross-cutting
wins were idle. **Landed DEEP-09 items 1, 2, 5** (batch 1024 + lr 2e-3 defaults — Runs ≤10 were
256/1e-3, A/B pending; feature cache in `data/cache/`, bit-identical-verified; timings dict) and
**DEEP-04's clean CE** in eval. Decision row added to PROJECT.md. HANDOFF-stage3.md now carries a
"Cross-cutting debts" section: DEEP-05 shared embedding tables (decide BEFORE the ablation),
DEEP-04 dashboard remainder, DEEP-06 usage-file swap (between rounds only), DEEP-03/08 live-bot
fixes (parked by Eric), DEEP-09 fast tier.

## Accumulated Context

### Roadmap Evolution
- Phase 1 added: Training Optimization
- Phase 2 added: Sanitization & Parity Cleanup
- Phase 3 added: Bot Integration
- Phase 4 added: TensorFlow Embedding Model — Refactored Architecture (replaces one-hot with learned embeddings; introduces `feature_engineering.py` shared module; does NOT touch `train_action_predictor.py`)
- Phase 5 added: Hyperparameter Tuning
