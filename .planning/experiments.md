# PokéML Training Experiments

| Run ID | Strategy | Command Snippet | Train Acc | Eval Acc | Gap | Eval LogLoss | Status |
|---|---|---|---|---|---|---|---|
| **0** | Baseline | `--feature_set medium` | 80.5% | 40.1% | 40.4% | 1.134 | 🚩 Leaky |
| **1** | HP Binning + Reg | `--lgbm_estimators 6000` | 76.2% | 39.5% | 36.7% | 1.029 | ⚠️ Marginal |
| **2** | **Hardened** | `--smoothing 0.1 --replay_count 50` | 83.5% | 37.7% | 45.8% | 1.978 | ❌ Gap Widened |
| **4** | **Clean Baseline** | **Corrected Parser (Temporal Fix)** | **87.0%** | **37.1%** | **49.9%** | **2.146** | 🏁 Leak Fixed, Memorization High |
| **5** | **Metamon Scaling** | **9M Rows / 16GB RAM** | **83.6%** | **37.3%** | **46.3%** | **1.985** | ❌ Identity Memorization |
| **6** | **Aggressive Reg** (Proposed) | `--cat_smooth 1000 --reg_lambda 100` | TBD | TBD | TBD | TBD | 🤞 Targeting Gap < 0.4 |
| **7** | **TF Embedding** | **Embeddings + Swish Stack (Ph 4)** | **51.5%** | **38.9%** | **12.6%** | **2.567** | ✅ **Breakthrough** |
| **B0** | **Prior-only floors** (DEEP-01 Stage 0, zero state) | `baseline_prior_floors.py data/100k.parquet --min_move_count 100` | n/a | **30.5%** top-1 / **80.6%** top-5 (Floor B) | n/a | n/a | 📏 **Floor — all models judged by margin above this** |
| **8** | **No class weights** (DEEP-02 §1.3: natural freqs = policy prior) | `train_action_predictor_embedding.py data/100k.parquet --min_move_count 100 --version v2` | ❓ (logs not kept) | **46.3%** test / **89.5%** top-5 | ❓ | **2.104** (test) | ✅ **New best** — +7.8 pts top-1 over Run 7, one-line fix |
| **9** | **Species attribute bundles** (DEEP-01 Stage 1: stats/types/speed/ability/item/role priors) | `train_action_predictor_embedding.py data/100k.parquet --min_move_count 100 --version v3 --species_attrs` | ❓ | **46.6%** test / 89.6% top-5 | ❓ | 2.103 (test) | ✅ +0.3 pts vs Run 8, as predicted — attributes redundant *while identity channels stay on*; real test = Stages 2–3 |
| **10** | **Matchup features** (DEEP-01 Stage 2: type eff both directions, speed order, KO pressure, coverage) | `train_action_predictor_embedding.py data/100k.parquet --min_move_count 100 --version v4 --species_attrs --matchup_features` | ❓ | **46.5%** test / 89.7% top-5 | ❓ | 2.101 (test) | ➖ **Flat vs Run 9** (menu-restr. 51.9% vs 52.0%) — even matchup info is redundant while identity channels stay on. Stage 3 ablation is now the decisive test |

---
## Run 5: Metamon Scaling Summary
Scaling to 9 million rows significantly improved convergence but **did not stop the identity shortcut**. `p1_active_species` remains the dominant feature (2.6M importance), meaning the model is still a lookup table for "Most Common Move per Species." 

## Strategy for Run 6: Aggressive Regularization
Since mass data alone didn't break the memorization, we will now **Radically Regularize** the model:
1.  **Categorical Smoothing**: Set `cat_smooth=1000` to prevent species-based splits without overwhelming evidence.
2.  **Feature Blinding**: Drop `colsample_bytree` to **0.05** so the model is forced to learn from hazards/HP in 95% of trees.
3.  **Complexity Cap**: Strictly limit tree depth and increase `min_child_samples` to 10k.

---
## Run 7: TF Embedding Model (Phase 4)

**Significant Breakthrough**: By moving to learned entity embeddings for species and moves, we collapsed the generalization gap from **~50% down to 12.6%**. 

**Key Observations**:
- **Performance**: Hit a new record of **38.9% validation accuracy** (up from 37.3% ceiling).
- **Generalization**: The model is no longer "memorizing" match IDs. The top-5 accuracy of **84.5%** suggests the model reaches an expert-level understanding of the competitive "menu" even if it doesn't always pick the #1 move.
- **Loss Gap**: While the jump from 1.2 (Train) to 2.5 (Val) looks steep, the absolute Val Loss of **2.56** is the lowest recorded for this complexity of target space (277 classes).

**Next Steps**:
- **Increase Regularization**: Add higher Dropout (0.35) or weight decay to try and close that final 12% gap.
- **Architecture**: Test if "Wide & Deep" (concatenating numerical features directly to the final layer) helps preserve raw stat values.

---
## Run B0: Prior-Only Baseline Floors (2026-07-05, DEEP-01 Stage 0)

Two **zero-state** baselines, evaluated on the **identical test split** as Run 7 (same filter
chain, same `GroupShuffleSplit(random_state=42)`), conditioning only on the acting mon's species
(`p1_active_species` — rows are `player_to_move == 'p1'`; DEEP-01 §4.1's "p2_active_species" was
a perspective slip):

| Floor | Definition | Top-1 | Top-5 | Species coverage |
|---|---|---|---|---|
| A | argmax Smogon usage move per species (`gen9ou-0.json`, no training data) | **26.7%** | 76.1% | 95.9% |
| B | per-species majority move from the train split | **30.5%** | 80.6% | 99.9% |
| — | **Run 7 embedding model** (reference, same split) | **38.5%** | 84.5% | — |

**Interpretation (confirms DEEP-01 §1.2):** a lookup table with zero board state already gets
30.5% top-1 and 80.6% top-5. Run 7's real margin over the prior is **+8.0 pts top-1 / +3.9 pts
top-5** — most of its headline performance is species prior, not state reasoning. All future
runs (especially the DEEP-01 `attrs_only` / two-tower ablations) must report margin above Floor B,
not raw accuracy.

**Run 7 dataset lineage resolved (was ❓):** fingerprinted `data/100k.parquet` +
`--min_move_count 100` → 277 classes whose sanitized identities **exactly match**
`action_label_encoder_v1_medium.joblib` (277/277). `data/30k.parquet` is the same data
(30,168 replays / 1,647,831 raw rows) with unsanitized display-name moves; `100k.parquet`
stores sanitized names, so it is the one the training run consumed.

---
## Run 8: Drop Balanced Class Weights (DEEP-02 §1.3) — NEW BEST (2026-07-06)

`compute_class_weight('balanced')` over 277 classes upweights rare moves ~100×, training the
model *away* from the human policy π_human. Dropping it (now the script default; old behavior
behind `--balanced_class_weights`) was worth **+7.8 points top-1 for free**:

| Metric (same dataset, same split) | Run 7 (balanced) | Run 8 (none) | Floor B (lookup) |
|---|---|---|---|
| Top-1 | 38.5% | **46.3%** | 30.5% |
| Top-5 | 84.5% | **89.5%** | 80.6% |
| Menu-restricted top-1 | (not measured) | **51.7%** | — |
| Test loss (smoothed CE) | 2.567 | **2.104** | — |
| **Margin over Floor B (top-1)** | +8.0 | **+15.8** | — |

Menu-restricted top-1 of 51.7% means: when the model has the moveset right (true move in its
top-5, 89.5% of turns), it picks the actual choice half the time over a ~4-option menu — the
state-reasoning residual is real but this is the number DEEP-01's feature work must move.
Train-side accuracy/gap unrecorded (stdout not kept); metadata JSON has full config lineage.
The script changes shipped with this run: `--version` artifact tags, dataset lineage in
metadata, menu-restricted metric in eval.

Also new: `evaluate_bot_winrate.py` (DEEP-02 §5) — bot vs RandomPlayer / MaxBasePower /
SimpleHeuristics via poke-env `cross_evaluate` on a **local** Showdown server, mirror teams.
Winrate vs fixed baselines is the end-to-end KPI that offline accuracy can't fake. Not yet run
(needs a local `pokemon-showdown` server; see script docstring).

---
## Run 10 (prepared 2026-07-06): Matchup / Relational Features (DEEP-01 Stage 2)

New module `matchup_features.py` → 12 numeric columns, wired via
`build_medium_X(matchup_features=True)` / `--matchup_features` (recorded in metadata JSON).
This is the stage DEEP-01 expects to actually move the numbers: until now the model had
**zero** matchup/speed-order information. Columns:

- `p{1,2}_best_revealed_eff_vs_*` (tera-aware type effectiveness of revealed damaging moves,
  both directions) and `p{1,2}_best_revealed_dmg_proxy` (eff × BP × STAB × attack-side match).
- `speed_edge` / `speed_ratio` (attribute-table `speed_est` adjusted for boosts, paralysis ×0.5,
  Tailwind ×2; sign flips under Trick Room — verified by an exact-negation test).
- `p{1,2}_dmg_frac_vs_*` + `p{1,2}_can_likely_ko_*` (crude KO pressure vs bulk and HP bin;
  fires on ~16% of rows).
- `p1_has_safe_switchin` / `p2_walled_count` (p1 bench typing vs p2's revealed damaging moves).

Validated on all 1.65M rows of `data/100k.parquet` (33s): effectiveness values exactly in
{0, ¼, ½, 1, 2, 4}, distributions sane (`python matchup_features.py data/100k.parquet` reruns
the checks). Typechart schema decoded: defender-keyed, `damageTaken` codes 0/1/2/3 =
1×/2×/0.5×/0× (asserted against known matchups in the CLI).

Also fixed while smoke-testing: rows with no flagged active mon carry NaN boosts (np.select
default in `build_medium_X`) — a single such row NaN'd the whole loss on `data/test.parquet`
(pre-existing; 100k.parquet has no such rows post-filter). `prepare_inputs` now zero-fills
non-finite numericals before scaling.

**Result (run 2026-07-06): FLAT.** 46.5% top-1 / 89.7% top-5 / menu-restricted **51.9%** /
loss 2.101 — statistically indistinguishable from Run 9 (46.6 / 89.6 / 52.0 / 2.103). Margin
over Floor B unchanged at ~+16 pts.

**Interpretation:** DEEP-01 expected Stage 2 to be the stage that moves raw numbers; it did not
— *while species embeddings + usage columns are still on*. This is the strongest evidence yet
for the §1.2 thesis: the identity prior already saturates what the softmax model extracts from
the state, so even genuinely new information (the model previously had zero matchup/speed-order
signal) adds nothing on top of it. Two readings, not yet distinguishable: (a) the shortcut
crowds out state reasoning during SGD (gradient starvation — identity explains the loss first,
matchup features never get pulled in), or (b) 12 scaled columns among 445 numericals are too
quiet an input channel. Either way the conclusion is the same: **augmenting the identity model
is exhausted; Stage 3's `attrs_only` ablation (drop species embeddings + usage columns) is now
the decisive experiment** — it tests whether attributes + matchup features can *replace* the
prior rather than decorate it, with held-out-species eval as the headline metric.

---
## Infra note (2026-07-06, after Run 10): DEEP-09 items 1, 2, 5 landed

**Runs ≤10 used batch 256 / lr 1e-3; the trainer defaults are now batch 1024 / lr 2e-3**
(measured 8× step throughput, DEEP-09 §3). The A/B validating the change is pending — the next
run with Run-10 features at the new defaults doubles as it. Also landed: the DEEP-09 §1 feature
cache (`data/cache/`, ~4-min build → seconds on hit, verified bit-identical; `--no_cache` /
`--prune_cache N`), a `timings` dict in every metadata JSON, and **clean CE** (log-loss without
label smoothing — DEEP-04's per-run metric) in the eval output. Runs ≤10 predate clean CE; only
their smoothed `test_loss` exists. Decision logged in PROJECT.md Key Decisions.

---
## Artifacts & Reproducibility

> Since `data/` and `models/` are **gitignored** (git no longer stores the bytes), this
> table is the source of truth linking each result to the artifact + dataset that produced it.
> **Going forward: whenever you train, add a row here (artifact filename + exact dataset).**

| Model file (in `models/`) | Produced by | Feature set | Dataset | Result / Run |
|---|---|---|---|---|
| `action_lgbm_model_v4_medium_move_only.txt` (+ `_feature_info_`, `_scaler_`, `action_label_encoder_v4_medium_move_only`) | `train_action_predictor.py --model_type lightgbm` | medium (move-only target) | ❓ unrecorded (10k/30k parquet lineage) | LightGBM baseline, Runs 0–4 (~37% eval) |
| `action_tf_model_v4_medium.keras`, `action_tf_model_v4_medium_move_only.keras` (+ `action_tf_preprocessor_v4_*`) | `train_action_predictor.py --model_type tensorflow` | medium | `data/10k.parquet` (per `archive/abababa.txt` cmd) | plain TF dense, ~30% val |
| `action_tf_embedding_v1_medium.keras` (+ `action_tf_embedding_metadata_v1_medium.json`, `action_tf_embedding_artifacts_v1_medium.joblib`, `action_label_encoder_v1_medium.joblib`) | `train_action_predictor_embedding.py --feature_set medium` | medium (277 classes) | `data/100k.parquet` + `--min_move_count 100` (recovered 2026-07-05 via exact 277-class label-encoder fingerprint; see Run B0 notes) | **Run 7 — best.** 38.5% test / 84.5% top-5 / loss 2.567 |
| `switch_predictor_lgbm_model_v2_simplified.txt` (+ `_feature_info_`, `_scaler_`) | `train_switch_predictor.py --feature_set simplified` | simplified | ❓ unrecorded | Binary switch classifier (2025-05-04) |
| `switch_target_predictor_lgbm_model_simplified_moves.txt` (+ `_feature_info_`, `_scaler_`, `_label_encoder_`) | `train_pokemon_switch_predictor.py --feature_set simplified` | simplified_moves | ❓ unrecorded | Switch-target classifier (2025-10-03) |
| `baseline_prior_floors_v1_medium.json` (results only, no model) | `baseline_prior_floors.py --min_move_count 100` | n/a (species → move lookup) | `data/100k.parquet` + `--min_move_count 100` (Run 7's split, replicated) | **Run B0 floors.** Smogon 26.7% / train-majority 30.5% top-1 |
| `action_tf_embedding_v2_medium.keras` (+ `_metadata_v2_`, `_artifacts_v2_`, `action_label_encoder_v2_medium.joblib`) | `train_action_predictor_embedding.py --min_move_count 100 --version v2` (no class weights) | medium (277 classes) | `data/100k.parquet` (recorded in metadata JSON) | **Run 8 — new best.** 46.3% test / 89.5% top-5 / menu-restricted 51.7% / loss 2.104 |
| `action_tf_embedding_v3_medium.keras` (+ `_metadata_v3_`, `_artifacts_v3_`, `action_label_encoder_v3_medium.joblib`) | `train_action_predictor_embedding.py --min_move_count 100 --version v3 --species_attrs` | medium + attr bundles (277 classes) | `data/100k.parquet` (in metadata JSON) | **Run 9.** 46.6% test / 89.6% top-5 / menu-restricted 52.0% / loss 2.103 |
| `action_tf_embedding_v4_medium.keras` (+ `_metadata_v4_`, `_artifacts_v4_`, `action_label_encoder_v4_medium.joblib`) | `train_action_predictor_embedding.py --min_move_count 100 --version v4 --species_attrs --matchup_features` | medium + attr bundles + matchup features (277 classes) | `data/100k.parquet` (in metadata JSON) | **Run 10.** 46.5% test / 89.7% top-5 / menu-restricted 51.9% / loss 2.101 — flat vs Run 9 |

**Note — bot vs. best model:** `predict_action.py` currently loads the **v4 LightGBM** action model
(`action_lgbm_model_v4_medium_move_only`), **not** the Run-7 embedding model. Wiring the embedding
model into the live bot is deferred work (Phase 3 / ensemble).

**Reproducibility convention:** artifacts are named `<task>_<framework>_<version>_<feature_set>[_<target>]`.
Datasets used to be loose parquet files in the repo root; they now live in `data/` and are gitignored,
so their provenance lives here, not in git.
