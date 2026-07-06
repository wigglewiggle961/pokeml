# Fable Deep-Dive Prompts (PokéML)

Nine high-reasoning prompts for the Fable model, ordered by priority. **Feed them one at a
time.** Each is self-contained but assumes Fable can read this repo (run it in Claude Code inside
the project so it can open the files listed). Later prompts may reference earlier outputs — if a
`DEEP-0x` file already exists in `.planning/research/`, tell Fable to read it first.

**Shared context to know (true as of 2026-07):**
- Project = Gen 9 OU Pokémon Showdown battle AI. Predicts the opponent's action from board state.
- **Central unsolved problem:** the move model memorizes opponent *species identity* instead of
  learning game state. In the LightGBM one-hot models `p1_active_species` dominated feature
  importance (~2.6M) — the model became a "most common move per species" lookup. Train/eval
  accuracy gap was ~40–50%.
- **Best model so far:** the TF embedding model (`train_action_predictor_embedding.py`) replaced
  one-hot with learned entity embeddings; the gap collapsed to ~12.6% and it hit **38.5% test acc /
  84.5% top-5 / val_loss 2.567 across 277 move classes.** This is the current frontier.
- Feature set "medium": slot-based team representation (6 slots/side: species, hp%, status,
  boosts, tera, revealed moves), active-Pokémon features, revealed-moves multi-hot (~6000 binary
  columns), field state (weather/terrain/pseudo), hazards, side conditions, last moves, turn number.
- Split = `GroupShuffleSplit` by `replay_id`. Perspective augmentation flips p1/p2
  (`augment_perspectives.py`). Inference filters predicted moves by Smogon usage (`data/gen9ou-0.json`).
- Orientation files to read: `PROJECT_CONTEXT.md`, `CLAUDE.md`, `.planning/PROJECT.md`,
  `.planning/experiments.md`, `.planning/ROADMAP.md`.

---

## DEEP-01 — Representation redesign (break the species-memorization shortcut)

> **Read first:** `PROJECT_CONTEXT.md`, `feature_engineering.py`, `process_replays.py` (the feature
> builders and slot encoding), `train_action_predictor_embedding.py` (how features enter the model),
> `.planning/experiments.md`, and the pending todo
> `.planning/todos/pending/2026-04-14-implement-explicit-semantic-mapping-for-moves.md`.
>
> **The problem.** Even the embedding model still leans on species identity to predict moves — it's
> learned a smarter lookup table, not a theory of the game. The project's stated principle is
> *"generalization over memorization: features should represent the state and attributes of the game,
> not match-specific IDs."*
>
> **Your task.** Design a fundamentally better, species-*agnostic* feature representation that forces
> the model to reason from game attributes rather than identity. Think from first principles about
> what information a strong human player actually uses to predict an opponent's move. Address:
> 1. A critique of the current representation — where exactly does identity leak in, and why does
>    embedding-per-species only partially fix it?
> 2. A concrete proposed representation: how to encode Pokémon as *stats + types + role + ability +
>    item + revealed moveset* rather than a species token. What derived/relational features (type
>    matchups, speed tiers, offensive/defensive roles, threat/coverage relationships) matter most?
> 3. How to handle moves themselves semantically (category, type, power, priority, effect class)
>    instead of 6000+ binary move columns — and how that interacts with the prediction target.
> 4. The trade-off: how much does removing species identity cost in raw accuracy vs. gain in
>    generalization, and how would you *measure* that this time (see DEEP-04)?
> 5. A staged implementation path that a solo developer can execute incrementally without a rewrite.
>
> Be concrete and opinionated. Name the specific columns/functions in `feature_engineering.py` you
> would change. **Deliverable:** write your analysis to `.planning/research/DEEP-01-representation-redesign.md`.

---

## DEEP-02 — Problem-framing critique (is exact-move prediction even the right target?)

> **Read first:** `PROJECT_CONTEXT.md`, `train_action_predictor_embedding.py` (target construction,
> `num_classes=277`), `predict_action.py` (how predictions are consumed by the live bot),
> `.planning/PROJECT.md`, `.planning/experiments.md`.
>
> **The problem.** The action predictor is a 277-way classifier over exact move names, plus separate
> switch models. A ceiling around ~38% top-1 may partly reflect that *exact move* is an ill-posed
> target: multiple moves are near-equivalent, the "right" move is often a distribution not a point,
> and the metric doesn't credit picking a reasonable alternative.
>
> **Your task.** Critically evaluate the ML problem formulation and propose alternatives. Address:
> 1. Is top-1 exact-move classification the right framing, given the goal is a bot that *plays well*
>    (not one that guesses the human's exact click)? What's the irreducible entropy here?
> 2. Evaluate alternative targets: move *archetype/intent* (setup / pivot / attack-strongest /
>    status / hazard), a value/EV formulation, or a policy over legal actions. Pros/cons of each for
>    both accuracy and downstream bot decisions.
> 3. Should move-prediction, switch-decision, and switch-target be one unified action head over the
>    legal action set rather than three stitched models? What would that cost/buy?
> 4. Does supervised imitation of human replays structurally cap performance, and if so where would
>    you augment it (search, self-play, value estimation) — scoped realistically for a solo project?
> 5. A recommended target reformulation with justification, and what it implies for labels in
>    `process_replays.py`.
>
> **Deliverable:** write to `.planning/research/DEEP-02-problem-framing.md`. If `DEEP-01` exists,
> read it and stay consistent with (or explicitly challenge) its representation choices.

---

## DEEP-03 — Leakage & pipeline-correctness audit

> **Read first:** `process_replays.py` in full (parsing → per-turn state → action labeling),
> `feature_engineering.py`, `augment_perspectives.py`, the split logic in
> `train_action_predictor_embedding.py`, and `.planning/experiments.md` (note Run 4 = "Corrected
> Parser (Temporal Fix)" — a temporal leak was already found and fixed once).
>
> **The problem.** History shows subtle leaks here before. A leak inflates train metrics and is a
> prime suspect whenever generalization looks suspiciously easy or hard.
>
> **Your task.** Do a rigorous correctness and leakage audit of the data pipeline. Address:
> 1. **Temporal integrity:** verify that every feature for a labeled action is derived strictly from
>    state *before* that action was chosen. Hunt for any field (revealed moves, HP, last_move,
>    boosts, terastallization, hazards) that could be populated with post-decision information for
>    the same turn. Cite exact lines in `process_replays.py`.
> 2. **Split integrity:** does `GroupShuffleSplit` by `replay_id` fully prevent leakage, given
>    perspective augmentation duplicates a battle as p1 and p2? Could both perspectives of one battle
>    land on opposite sides of the split?
> 3. **Label correctness:** are `move:X` / `switch:Y` labels always attributed to the right player
>    and the right turn? Any off-by-one between state and action?
> 4. **Normalization parity:** confirm `sanitize_name`/species normalization is identical between
>    training and `predict_action.py` inference (this is REQ-02).
> 5. Any silent data-quality issues (dropped turns, forfeits, team-preview, forced switches after
>    faint, Zoroark/illusion, diagnostics) that pollute labels.
>
> Rank findings by severity with concrete repro reasoning. **Deliverable:** write to
> `.planning/research/DEEP-03-pipeline-leakage-audit.md`.

---

## DEEP-04 — Evaluation methodology (define the north-star metric)

> **Read first:** `train_action_predictor_embedding.py` (metrics, the Optuna `objective` ~line 535
> which minimizes `val_loss`), `.planning/ROADMAP.md` (Phase 5 target: val_acc > 40%, top-5 > 75%),
> `.planning/experiments.md`, `predict_action.py`.
>
> **The problem.** Phase 5 tuning minimizes `val_loss`, but the acceptance criteria are stated in
> accuracy — and neither cleanly measures "does this bot play well." Best trial's accuracy was never
> even recorded. There's no agreed north-star metric.
>
> **Your task.** Design the evaluation methodology for this project. Address:
> 1. What metric(s) actually reflect a good *opponent-action predictor* and, separately, a good
>    *bot*? Discuss top-1 vs top-k accuracy, calibrated log-loss, per-class/macro performance given
>    heavy class imbalance, and metrics conditioned on situation (lead turn vs endgame, forced vs
>    free choice).
> 2. Should the tuning objective be val_loss, val_accuracy, top-5, or a composite? Justify, and
>    reconcile it with the Phase 5 criteria.
> 3. How to evaluate the *bot* beyond next-move accuracy — e.g. ladder win-rate, agreement with
>    strong play, regret vs a reference policy — and what's feasible for a solo dev.
> 4. A concrete, minimal evaluation harness proposal (what to log per trial/run, held-out slices,
>    baselines to beat such as "always most-common-move-for-species").
> 5. A single recommended north-star metric plus a small dashboard of secondary metrics.
>
> **Deliverable:** write to `.planning/research/DEEP-04-evaluation-methodology.md`. Keep it
> consistent with DEEP-01/02 if those exist.

---

## DEEP-05 — Model architecture review (is the network itself right, given the data and a solo dev?)

> **Read first:** `train_action_predictor_embedding.py` (the whole model:
> `build_embedding_model`, embedding dims, the flat-concat → 512→256→128 head, Optuna search
> space), `train_action_predictor.py` (the LightGBM baseline it must beat),
> `.planning/research/DEEP-01-representation-redesign.md` and
> `.planning/research/DEEP-02-problem-framing.md` (these fix the inputs and outputs — DEEP-05 is
> about the middle), `.planning/experiments.md`.
>
> **The problem.** DEEP-01/02 redesigned what goes *into* the model (attribute bundles, relational
> features) and what comes *out* (unified candidate-scoring policy), but the network between them
> has never been examined. It is currently the simplest possible thing: flatten every input,
> concatenate, three Dense layers. Run 7 changed encoding *and* architecture at once, so we don't
> even know how much the architecture contributed. Known unknowns: the model sees a single turn
> snapshot (no history beyond `last_move_*`); the 12 team slots are order-sensitive flat inputs
> even though slots are interchangeable; embedding dims (species=48, moves=32) are guesses; and
> nobody has checked whether deep beats a well-regularized LightGBM *on the same features* at our
> data scale (~1–9M rows).
>
> **Your task.** Review the model architecture from first principles, constrained hard by
> feasibility (solo dev, consumer GPU/CPU, bursts of work). Address:
> 1. **Snapshot vs. sequence.** How much predictive signal plausibly lives in turn history
>    (reveals, momentum, opponent tendencies within a game) that a single-snapshot model can never
>    see? Compare options: keep snapshot; snapshot + engineered history summaries (cheap); GRU or
>    small transformer over the turn sequence (expensive — changes data loading, training, and the
>    live bot's state tracking). Recommend when, if ever, the sequence jump is worth it.
> 2. **Flat concat vs. structured/set encoder.** Evaluate a shared per-Pokémon encoder applied to
>    all 12 slot attribute-bundles with pooling/attention (permutation-invariant over bench,
>    parameter-shared) vs. today's flat concatenation. What does it buy for generalization and
>    parameter count, and what does it cost in code complexity and inference parity?
> 3. **Ablation discipline.** Design the minimal experiment grid that separates encoding gains
>    from architecture gains — including "LightGBM on the DEEP-01 attribute features" as a
>    mandatory baseline arm, since if it matches the deep model, the deep model isn't earning its
>    complexity yet.
> 4. **Capacity and budget fit.** Are 48/32-dim embeddings, the 512→256→128 head, batch 256, and
>    the current Optuna search space (lr, dropouts, layer widths only) sensible for this data
>    scale? What should the search space actually contain, and what single tuning objective (per
>    DEEP-04 if it exists) should it minimize?
> 5. **Fit with the DEEP-02 endgame.** Which architecture choices here carry forward into the
>    unified candidate scorer (shared state trunk + per-candidate scorer MLP), and which would be
>    throwaway? Prefer choices that survive the transition.
> 6. A ranked recommendation: the ordered list of architecture changes worth making, each with
>    expected payoff, implementation cost in days, and the experiments.md row that would validate
>    it.
>
> **Deliverable:** write to `.planning/research/DEEP-05-architecture.md`. Stay consistent with
> DEEP-01/02 (or explicitly challenge them where the architecture evidence disagrees).

---

## DEEP-06 — Data strategy & meta drift (which replays deserve to be trained on?)

> **Read first:** `process_replays.py` (what metadata is parsed vs discarded — note player
> ratings in `|player|` lines are currently dropped), the data-collection scripts in the repo
> root, `.planning/experiments.md` (Run 5 = 9M-row metamon scaling; also the "❓ unrecorded"
> dataset provenance rows), `data/` contents, and `.planning/research/DEEP-01/02/04` if present.
>
> **The problem.** Nobody has ever asked *which* replays should be in the training set. Gen 9 OU
> in 2023 and 2026 are different games (DLC waves, bans, tera-policy votes), so old replays teach
> outdated play. `data/gen9ou-0.json` is a single undated Smogon snapshot. The metamon corpus vs.
> our own scrapes have unknown rating and era composition. And evaluation splits randomly by
> replay, so we never measure the thing deployment actually faces: playing in *next month's*
> meta. Data composition usually moves accuracy more than model changes — and it interacts
> directly with DEEP-02's rating-weighted BC proposal.
>
> **Your task.** Design the project's data strategy. Address:
> 1. **Corpus audit method:** how to characterize what we currently train on (rating
>    distribution, date/era distribution, format legality, source mix metamon-vs-scraped) given
>    what the parquet schema does and doesn't record. What must `process_replays.py` start
>    emitting to make this auditable (ties into DEEP-02 §5's parser additions)?
> 2. **Meta drift:** how stale is too stale? Propose concrete era boundaries for Gen 9 OU
>    (DLC/ban watersheds), and an experiment to measure drift cost — e.g., train on era N,
>    evaluate on era N+1, vs. train on mixed.
> 3. **Time-based evaluation:** should the canonical test split become "most recent X months"
>    instead of (or alongside) random-by-replay? Reconcile with the GroupShuffleSplit hygiene
>    and the DEEP-04 metric suite.
> 4. **Quality vs. quantity:** rating-filtered subsets (e.g., ≥1500) vs. the full 9M rows —
>    propose the learning-curve experiment that settles how much data is enough and whether
>    high-rated-only beats everything-weighted.
> 5. **Freshness operations:** versioning for the Smogon usage JSON (which month's stats, pinned
>    and recorded per experiments.md conventions), a re-scrape/refresh cadence that fits burst
>    development, and what "retrain the deployed model" should be triggered by.
> 6. A recommended canonical training corpus definition (sources, era window, rating floor,
>    weighting) that all future runs reference by name in experiments.md.
>
> **Deliverable:** write to `.planning/research/DEEP-06-data-strategy.md`.

---

## DEEP-07 — Search & consumption design (how the bot should *use* the models)

> **Read first:** `.planning/research/DEEP-02-problem-framing.md` §4 (the augmentation ladder —
> this prompt designs its step 2–3), `predict_action.py` (`choose_move`, the current
> argmax-of-cloned-policy consumption), poke-env's docs/source for what battle state, stats, and
> damage-relevant data it exposes at runtime, and DEEP-04's harness if it exists.
>
> **The problem.** DEEP-02 recommends flipping the model from "play the human move" to "opponent
> model inside a 1-ply expectimax with a damage calculator" — but that's a paragraph, not a
> design. Pokémon turns are simultaneous-move (not alternating minimax), damage depends on
> hidden EVs/items, and the bot must decide within the Showdown timer. Building this naively
> could easily produce a bot *worse* than the argmax clone.
>
> **Your task.** Produce the design for the search-based bot, scoped to a solo developer.
> Address:
> 1. **Decision rule:** formalize the 1-ply computation — for each of our legal actions, expected
>    value over the opponent's predicted action distribution. How to handle the simultaneous-move
>    structure honestly (we exploit a *predicted* distribution rather than solving for
>    equilibrium — when does that break, and does a best-response-to-prediction bot get punished
>    by mixing opponents?).
> 2. **Damage model fidelity:** what accuracy does the damage calc need before search beats the
>    clone? Tiers: DEEP-01's crude eff×BP×stat-ratio proxy → full formula with Smogon modal
>    spreads/items → range-aware with unknown-set uncertainty. Where's the knee of the curve?
> 3. **Leaf evaluation:** combining immediate damage/KO outcomes with the replay-trained win-prob
>    model V(s) from DEEP-02 §4.3 — or, before V(s) exists, a hand-rolled positional heuristic
>    (HP totals, hazards, speed control). What's good enough to start?
> 4. **Risk and mixing:** should the bot maximize expectation always, or shade toward
>    variance-seeking when V(s) says it's losing? Should it *sample* among near-equal actions to
>    avoid exploitable determinism (DEEP-02 §4's critique)?
> 5. **Budget:** per-turn latency envelope under the Showdown timer with poke-env overhead;
>    what depth/branching is affordable in Python; when (if ever) 2-ply is worth it.
> 6. **Validation plan:** the experiment sequence proving each increment helps — clone vs.
>    clone+filter vs. 1-ply-crude vs. 1-ply-full-calc vs. +V(s) — all measured in the DEEP-04
>    winrate harness, with experiments.md rows.
>
> **Deliverable:** write to `.planning/research/DEEP-07-search-design.md`.

---

## DEEP-08 — Live-bot parity & robustness audit (does deployment see what training saw?)

> **Read first:** `predict_action.py` **in full** — especially `map_battle_to_dataframe_row`
> (rebuilding training features from live poke-env `Battle` objects), `load_smogon_moves` (note:
> a *separate reimplementation* of the Smogon-usage logic that also lives in
> `feature_engineering.py:get_smogon_usages_df` — two copies that can drift),
> `_prepare_data_for_model` / `_prepare_data_for_move_model`, and every `except` block. Then
> `feature_engineering.py` and `process_replays.py` for the training-side counterparts, and
> `.planning/research/DEEP-03-pipeline-leakage-audit.md` if it exists (DEEP-03 audits the
> training pipeline; this audits the *deployment* path).
>
> **The problem.** "Parity is King" is a stated project principle, but the live path has never
> been verified end-to-end: training features come from replay-log parsing, live features from
> poke-env object mapping — two entirely different code paths that must produce identical rows.
> Meanwhile the bot's exception handlers silently fall back to random moves, so live breakage is
> invisible; and the three deployed models don't even share a feature set (`medium` for moves,
> `simplified` for the switch models).
>
> **Your task.** Audit the deployment path and design the guardrails. Address:
> 1. **Column-by-column parity review:** walk `map_battle_to_dataframe_row` against
>    `process_replays.py`'s flattening and `build_medium_X`'s expectations. Hunt specifically:
>    species/forme normalization differences (live poke-env species strings vs.
>    `normalize_species_name`'s suffix stripping), HP binning timing, revealed-moves accounting
>    (does the live tracker match the parser's strictly-temporal reveal semantics DEEP-03
>    checks?), status/boost/hazard naming, and the duplicated Smogon-usage injection. Cite lines.
> 2. **The duplicated-logic problem:** `predict_action.py` reimplements sanitization, usage
>    loading, and feature prep instead of importing `feature_engineering.py`. Propose the
>    consolidation (what moves into the shared module, what the bot imports) so drift becomes
>    impossible rather than merely audited.
> 3. **A mechanical parity test:** design the harness — log every live decision's raw feature
>    row during a test battle, then rebuild the same turns from the battle's replay log through
>    the training pipeline, and diff column-wise. Define pass/fail tolerance. This should become
>    a repeatable check, not a one-off.
> 4. **Silent-failure inventory:** every `except`-and-continue in `choose_move`'s path, what each
>    hides, and a logging scheme (per-turn decision record: features hash, model outputs, chosen
>    action, fallback reason) that makes live failures diagnosable after the fact.
> 5. **Version skew:** the bot loads v4 LightGBM + v2/v1 switch models trained on different
>    feature sets and datasets. What invariants should artifact metadata carry (feature list
>    hash, dataset id, `feature_engineering.py` version) and be *checked at load time*?
> 6. Ranked findings + fixes, severity-ordered, with the usual experiments.md / PROJECT.md
>    logging hooks.
>
> **Deliverable:** write to `.planning/research/DEEP-08-live-parity.md`.

---

## DEEP-09 — Training speed & iteration velocity (make experiments cheap)

> **Read first:** `train_action_predictor_embedding.py` (data loading, `build_medium_X` call,
> `prepare_inputs` materializing full numpy dicts in RAM, `model.fit` with `epochs=100000` +
> `EarlyStopping(patience=10)`, the Optuna loop re-running full trainings per trial),
> `feature_engineering.py:build_medium_X` (the multi-hot encoding and Smogon joins that re-run
> from scratch on *every* training invocation), `.planning/experiments.md`. Also check what
> hardware is actually available (GPU model/VRAM, RAM) before recommending anything —
> run `nvidia-smi` / inspect TF device logs rather than assuming.
>
> **The problem.** Every proposal in DEEP-01–08 turns into *runs*, and runs are currently
> expensive: feature building is repeated identically for every experiment, the whole dataset
> lives in RAM as float32 numpy dicts, trials pay full training cost, and nothing records how
> long a run takes. For a burst-development solo project, iteration speed *is* research speed —
> the constraint isn't GPU-hours, it's how many hypotheses fit in one weekend burst. (Overlap
> note: DEEP-05 owns model-size choices; this prompt owns everything *around* the model.)
>
> **Your task.** Design the fast-iteration setup. Address:
> 1. **Profile first:** instrument where wall-clock actually goes for one representative run
>    (parquet load → build_medium_X → prepare_inputs → per-epoch time → total-to-early-stop).
>    Propose the timing log format so every experiments.md row gets a duration column.
> 2. **Feature caching:** design a cache for built features — key = (dataset id, feature-set,
>    build params, feature_engineering.py version-hash), value = the built X + feature lists +
>    encoders, stored under `data/cache/`. Invalidation rules; how DEEP-01's attribute-table
>    additions plug in. This is likely the single biggest win — spec it concretely.
> 3. **Input pipeline:** in-RAM dicts vs. `tf.data` from memory-mapped/parquet shards; where the
>    9M-row metamon scale breaks the current approach (Run 5 already needed 16 GB); mixed
>    precision and XLA on the available hardware — measured, not assumed.
> 4. **Convergence economics:** is `patience=10` at batch 256 leaving 2× on the table? Evaluate
>    larger batches + scaled LR, cosine/one-cycle schedules vs. ReduceLROnPlateau, and
>    Optuna with aggressive pruning + small-subset proxy trials (do subset rankings transfer?).
> 5. **The two-tier protocol:** define the official "fast tier" (e.g., 10% replay-subsampled
>    corpus, fixed seed, for feature/architecture screening) vs. "full tier" (milestone runs
>    only), including the rule for when a fast-tier win must be confirmed at full tier before
>    being believed — and how both tiers are recorded in experiments.md without confusion.
> 6. Ranked list of speed changes with expected speedup, implementation cost, and risk of
>    changing results (anything that alters convergence must re-validate against a known run).
>
> **Deliverable:** write to `.planning/research/DEEP-09-training-speed.md`.
