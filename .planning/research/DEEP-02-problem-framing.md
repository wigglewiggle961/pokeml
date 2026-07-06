# DEEP-02 — Problem-Framing Critique: Is Exact-Move Prediction the Right Target?

> Written 2026-07-05. Companion to [DEEP-01-representation-redesign.md](DEEP-01-representation-redesign.md)
> (representation) and DEEP-04 (measurement). Consistent with DEEP-01's endgame — the candidate
> scorer in its §3.2 — but this document extends it: the scorer should be a **unified head over
> the full legal action set** (moves + switches + tera), trained with outcome/rating weighting,
> and consumed by a shallow search rather than played argmax-raw.

---

## 0. TL;DR / Opinionated summary

1. **First, an identity crisis the docs paper over:** PROJECT_CONTEXT says the project "predicts
   opponent actions," but `predict_action.py:choose_move` uses the model as the **bot's own
   policy** — it maps the battle with the bot as p1, predicts "what a human would click in my
   seat," and plays it. That's behavioral cloning, not opponent modeling. The same model can
   serve both roles (perspective augmentation already trains both seats), but the roles have
   *different* requirements: a policy wants good argmax; an opponent model wants a **calibrated
   distribution**. Neither goal is served by chasing raw top-1.
2. **~38% top-1 is not obviously bad — much of the remaining 62% is irreducible.** The label is a
   draw from a stochastic, heterogeneous human policy. Mixed strategies are game-theoretically
   *correct* in a simultaneous-move imperfect-information game; ladder replays mix skill levels;
   and the features can't see hidden information the player used. A realistic top-1 ceiling is
   plausibly ~50–60%, not 100%. The fix is not to squeeze top-1 but to change what we optimize
   and how we consume it.
3. **Concrete bug-level finding:** `compute_class_weight('balanced')` over 277 classes
   (train_action_predictor_embedding.py:521-530) is wrong for policy cloning. It massively
   upweights rare moves, so the fitted distribution is no longer an estimate of π_human and both
   top-1 and calibration suffer. Drop it. This is a free improvement, one line, next run.
4. **Three stitched models is the worst of the available designs.** The 0.85 switch threshold is
   an arbitrary, miscalibrated seam (the bot only switches when 85% sure → structurally
   move-biased); move probabilities are conditioned on "given not switching" so the heads can't
   be compared; three feature-prep paths violate the project's own "Parity is King" principle.
   And the current action space omits Terastallization entirely — the bot literally never teras
   (`create_order(chosen_action)` with default flags). Unify into one head over legal actions.
5. **Recommended target: a unified legal-action policy** — softmax over the ≤9 legal candidates
   (4 moves + 5 switches, tera as a small side head initially), rating/outcome-weighted, with a
   move-intent auxiliary head, evaluated by log-loss + menu-restricted accuracy + **bot winrate
   against fixed baselines** (the only metric that measures "plays well").
6. **Imitation does cap performance, and the cap binds at the tactical layer.** A reactive clone
   of median ladder play cannot count damage or look ahead. The realistic solo-dev path beyond
   it is: (a) rating-filtered BC now, (b) 1-ply expectimax using the model *as opponent model*
   plus a damage calc — this is where the "predict opponent" framing finally earns its keep —
   (c) a replay-trained value function as leaf eval, (d) self-play RL only if/when that plateaus.

---

## 1. Top-1 exact-move classification: what it is and what it can't be

### 1.1 What the current formulation actually optimizes

`train_action_predictor_embedding.py` filters to `player_to_move == 'p1'` and `move:` rows,
label-encodes 277 move names, and minimizes (class-weighted, label-smoothed) cross-entropy. This
estimates π_pop(move | state): the move distribution of the *replay population* — every rating
band, every playstyle, tilted by the class weights (see §1.3). The bot then plays
argmax-filtered-to-legal of this estimate.

### 1.2 Irreducible entropy: why 100% was never on the table

Four separate noise sources sit between the features and the label:

- **Game-theoretic mixing.** Pokémon turns are simultaneous-move with hidden information; the
  equilibrium policy in many states is genuinely mixed (that's what "50/50" means in VGC/OU
  speak). When the true policy is 60/40 between two lines, perfect modeling scores 60% on those
  states *by definition*.
- **Population heterogeneity.** A 1100-rated and an 1800-rated player click different moves in
  the same state. The model fits their mixture. (This is also a data-quality lever — see §4.)
- **Hidden context.** The player knows their own EVs/item/full set and remembers reveals from
  team preview and chat-of-the-mind; our features don't. Some choices are inexplicable given
  our observation, hence noise to the model.
- **Near-equivalent actions.** Close Combat vs. Headlong Rush into a target both KO; Protect
  first vs. hazard first; order-swapped setup lines. Exact-match scoring gives zero credit for
  picking the co-optimal twin.

Rough calibration of the ceiling: Run-7's val loss 2.567 nats (label smoothing inflates this
somewhat) is perplexity ≈ 13 over 277 classes; top-5 = 84.5% says the effective menu is ~4–5.
If π_human conditional on the menu carries 1.5–2 bits (typical for a 4-option decision with a
favorite), the best achievable top-1 sits around 50–60%. Published Showdown imitation systems
report the same 30–40% band we're in, for the same reason. **Implication: the distance from
38.5% to the ceiling is real but modest; the distance from "clone of median ladder" to "plays
well" is much larger and lives outside the top-1 metric entirely.**

### 1.3 Two self-inflicted wounds in the current setup

- **Balanced class weights (fix immediately).** `compute_class_weight('balanced')` on a 277-way
  policy target means a move seen 100× less often gets 100× the gradient weight. The model is
  being trained to *not* be π_human. For a policy/opponent model we want natural frequencies —
  the prior over moves is signal, not imbalance. Expect both top-1 and log-loss to improve when
  dropped; label smoothing (already present) is the right regularizer to keep.
- **The metric ignores the consumer.** The bot takes argmax after legality filtering; training
  never sees the legality mask (277-way softmax spends probability mass on moves the mon can't
  even have), and evaluation never measures the thing the bot needs (rank quality *within* the
  legal menu). DEEP-01's menu-restricted accuracy is the honest per-decision metric; bot winrate
  vs. fixed opponents is the honest end-to-end one.

---

## 2. Alternative targets, evaluated

### 2.1 Move archetype / intent (setup, pivot, attack, status, hazard, heal, protect…)

- **Pros:** ~277 → ~10 classes; credits near-equivalent choices; directly reuses DEEP-01 §3.1's
  move effect-class map (zero extra labeling — derive intent from `gen9moves.json` at training
  time, **no parser change needed**); interpretable error analysis ("model knew it was a pivot
  turn but picked the wrong pivot move").
- **Cons:** lossy for the bot — you still must emit an actual move, so intent alone needs a
  second stage that reintroduces the original problem; class boundaries are mushy (Knock Off is
  attack *and* item-removal utility; Flip Turn is attack *and* pivot); and the accuracy gain is
  partly an illusion — collapsing classes raises the number without adding decision-relevant
  information.
- **Verdict: auxiliary head and evaluation lens, not the target.** A second softmax over intent
  classes sharing the trunk regularizes the representation toward strategy-shaped features
  (synergizes with DEEP-01's whole thesis) at ~zero cost. As a *sole* target it's a downgrade.

### 2.2 Value / EV formulation (V(s) or Q(s,a) from game outcomes)

- **Pros:** optimizes the actual goal ("win"), not mimicry; a value function is exactly the leaf
  evaluator a search needs (§4); binary win/loss labels are free in the replays
  (`battle_winner` is already parsed).
- **Cons:** credit assignment across 30–60 turn games from off-policy, mixed-skill data is
  high-variance; Q(s,a) additionally needs action coverage the data doesn't have (we only see
  the action the human took); as the *only* objective it discards the enormous prior encoded in
  human move choice. Full offline-RL on this (the Metamon project's route) is a research
  program, not a solo burst.
- **Verdict: yes to V(s) as a second model** (binary "did the player in this seat win," trained
  with the same GroupShuffleSplit hygiene — it's cheap, and it unlocks search), **no to
  replacing the policy target with values.** Also adopt the cheap proxy immediately:
  outcome/rating **weighting** of the BC loss (§4.1) gets some of the value signal into the
  policy without any credit-assignment machinery.

### 2.3 Policy over the legal action set (candidate scoring)

DEEP-01 §3.2 already argues this for moves; stated here as the target reformulation:

- **Pros:** the softmax support becomes the *actual decision set* (≤9 options), so all
  probability mass is meaningful; scoring per-candidate with mechanics features
  (effectiveness-vs-target, priority-vs-speed-gap, what-the-switch-in-walls) is species-agnostic
  and transfers to unseen moves/mons; log-loss over candidates is a clean, comparable metric
  turn-over-turn; and it naturally absorbs switches (§3) and, later, tera variants.
- **Cons / honest difficulties:** (a) constructing the candidate set in training data — see §5's
  labeling notes: the acting player's eventually-revealed moveset defines the menu, which is
  correct label-side information (the player knew their own moves) but *under*-covers negatives
  (moves never revealed were never chosen — pad negatives with Smogon-likely moves so the scorer
  learns to reject plausible-but-unchosen options); (b) a real architecture change (shared
  scorer MLP over [state ⊕ candidate]) — new script, per DEEP-01 Stage 5; (c) top-1 numbers stop
  being comparable to the old 277-way runs (feature, not bug — but log it clearly in
  experiments.md).
- **Verdict: this is the right target.** It is also the formulation under which "prediction
  accuracy" and "bot quality" finally align, because the training-time object *is* the
  decision-time object.

---

## 3. One unified action head vs. three stitched models

Current live pipeline (`choose_move`, predict_action.py:1148-1212): binary switch model →
if P(switch) > 0.85 → switch-target model (5 bench slots) → else 277-way move model → legality
filter → argmax.

**What stitching costs today:**

- **Incoherent probabilities.** The move model is trained only on move-rows, i.e., it estimates
  P(move | state, chose-to-move). Its outputs are not comparable with the binary head's, and no
  joint distribution over actions exists anywhere in the system.
- **The 0.85 threshold is a hand-tuned seam** that makes the bot structurally switch-averse and
  interacts badly with the binary model's own calibration (LightGBM probabilities at the tail
  are the least reliable part of the model).
- **Triple parity surface.** Three artifact sets, three `_prepare_data_for_model` code paths in
  predict_action.py, three places for train/inference drift — against the project's own "Parity
  is King" principle. The switch models still run on the older `simplified` feature set while
  the move model uses `medium`, so the three heads don't even see the same state.
- **The action space is incomplete.** `action_taken` is only `move:X | switch:Y`; the tera
  decision is unmodeled and the bot never terastallizes — in Gen 9 that alone is worth real
  winrate.
- **Data inefficiency.** The switch/move split throws away the shared structure (the same state
  encoder should serve both), and each head trains on a fraction of the rows.

**What unification buys:** one softmax over the ≤9 legal candidates gives a coherent
π(a | state); forced switches become a mask (rows where `|switch|` follows a faint train the
same head — recovering training data the current move-only filter discards); the threshold
disappears; one artifact set, one prep path; switch-vs-move calibration is learned from data
instead of hard-coded. **Costs:** full retrain of the deployed stack; move/switch base-rate
imbalance must be left *unbalanced* (per §1.3 — the ~20% natural switch rate is the correct
prior); switch candidates need their own feature block (the DEEP-01 bench attribute bundles slot
in directly); and the live bot's Stage-1/Stage-2 logic gets rewritten (a simplification, but
touching the fragile parity code).

**Verdict: unify — but as the Stage-5 scorer, not as a retrofit of the 277-way softmax.**
Interim step that's worth it now: keep the three models but replace the 0.85 threshold with
comparing P(switch) against a calibrated cutoff fit on validation data (or just lower it toward
the natural switch rate) — one constant, measurable in bot games. Tera: start as a separate
small binary head ("tera this turn?") over the unified trunk; folding tera-variants into the
candidate set doubles move candidates for marginal early value.

---

## 4. Does supervised imitation cap performance? Yes — here's where, and what to do

**Where the cap binds:**

1. **You clone the mean, mistakes included.** π_pop includes 1100-rated misplays. Argmax of a
   mixture of good and bad players is not a good player; it's a confident median player.
2. **Determinism is exploitable.** Argmax collapses the (correctly) mixed human policy into a
   pure strategy; a human opponent who notices the bot always makes the "obvious" play can
   exploit it indefinitely. (Sampling from the distribution instead of argmax is a two-line
   change in `_find_best_valid_move` and worth an experiment on its own.)
3. **Compounding drift.** BC is reliable on-distribution; a bot that misplays turn 8 lands in
   states humans rarely reach, where the clone is worst. Classic imitation-learning failure —
   and invisible to *any* offline accuracy metric, which is why bot-vs-baseline winrate must
   enter the eval suite.
4. **No lookahead.** Strong play is heavily tactical arithmetic — damage ranges, KO counting,
   speed order — which humans compute *explicitly* per turn. A reactive policy must have
   memorized the arithmetic's outputs; it cannot verify them. This is the hardest ceiling.

**Realistic augmentation ladder for a solo developer (in order; each step is independently
shippable):**

1. **Rating/outcome-weighted BC (days).** Showdown replay logs carry player ratings in the
   `|player|` lines — `process_replays.py` currently discards them (it stores only names).
   Parse rating into per-row metadata; weight the loss by rating (e.g., soft threshold above
   ~1500) and/or by whether that seat won. Cheapest possible move toward "clone *good* play."
   Requires the §5 parser additions and a re-parse, batched with DEEP-01 Stage 6.
2. **1-ply expectimax consumption (a week, no retraining).** Flip the model into its *opponent
   model* role: for each of our legal actions, evaluate expected outcome against the opponent's
   predicted action distribution using a damage calculator, pick the best. This uses the
   distribution (calibration now matters — another reason to drop class weights) and injects
   exactly the tactical arithmetic BC lacks. A crude damage proxy (DEEP-01 §2.2's) already
   helps; poke-env exposes enough (stats, moves, typechart) to do a serviceable calc.
3. **Replay-trained V(s) as leaf eval (a week).** Binary win-prob head on the shared state
   encoder. Turns the 1-ply search into "damage arithmetic + positional judgment." Also yields
   the win-prob trace over a game — an excellent debugging/analysis artifact.
4. **Self-play fine-tuning (out of scope, deliberately).** PROJECT.md rules RL out; keep it out
   until steps 1–3 plateau *in measured winrate*. If that day comes, the Metamon line of work
   (offline RL on Showdown at scale) is the reference point — and its datasets are already in
   this repo's orbit (`metamon_*/`).

---

## 5. Recommended reformulation and label implications

**Target:** unified candidate-scoring policy π(a | s) over the legal action set (available
moves ∪ available switches), trained on both seats' decisions (drop the `player_to_move == 'p1'`
filter in favor of proper two-seat emission — perspective augmentation already half-does this),
with:

- **Loss:** cross-entropy over candidates, label smoothing ~0.05, **no class weights**,
  sample weights from rating/outcome (§4.1).
- **Auxiliary head:** intent class of the chosen action (~10 classes from the DEEP-01 move
  effect-class map, `switch` as its own intent) — regularization toward strategy features.
- **Tera:** separate binary head on the shared trunk, phase 2.
- **Primary metrics:** candidate log-loss (calibration), menu-restricted top-1, held-out-species
  accuracy (DEEP-01 §4), and **winrate vs. poke-env's RandomPlayer / MaxBasePower /
  SimpleHeuristics** as the non-negotiable end-to-end gate. Raw 277-way top-1 retires.

**Label/parser implications for `process_replays.py`** (batch all of these with the Stage-6
re-parse from DEEP-01 so the datasets are rebuilt once):

1. **Keep `action_taken` unchanged** (`move:X` / `switch:Y`) — downstream compatibility; the
   candidate framing is additive.
2. **Emit `p{n}_rating`** from the `|player|` lines (currently discarded at
   process_replays.py:236-242) — enables §4.1 weighting. Single highest-value parser change.
3. **Emit `action_was_forced`** (the decision followed a faint / U-turn completion) so forced
   switches can be masked or trained separately in the unified head.
4. **Candidate sets need no new parser columns.** The acting player's menu = the active mon's
   *eventually-revealed* moveset for that game (legitimate label-side info — the player knew
   their own four moves; the per-turn `revealed_moves` field stays strictly temporal for
   *features*) ∪ alive bench species (already in slot state). Both are reconstructible at
   training time from existing columns plus a per-replay terminal pass — implement in the
   training script, not the parser, to keep the parquet schema stable. Pad move negatives with
   Smogon-likely moves so unrevealed-but-plausible options exist as negatives.
5. **Optional, with #2:** emit `battle_rated`/format metadata if present, for dataset filtering.

**Sequencing relative to DEEP-01** (they interleave; neither blocks the other):

| Step | Source | Cost |
|---|---|---|
| Drop `class_weight='balanced'`; rerun Run 7 config | this doc §1.3 | one line + one run |
| Prior-only floors + menu-restricted metric | DEEP-01 Stage 0 / §4 | ½ day |
| Bot-vs-baseline winrate harness (poke-env cross_evaluate) | this doc §5 | 1 day, reusable forever |
| Attribute + relational features | DEEP-01 Stages 1–3 | days |
| Parser re-parse: ratings, forced flag, items/abilities | this doc §5 + DEEP-01 Stage 6 | 1–2 days + re-parse |
| Unified candidate scorer (moves + switches), intent aux head | DEEP-01 Stage 5 + this doc | ~a week |
| 1-ply expectimax bot consuming the scorer as opponent model | this doc §4.2 | ~a week |

The first three rows are a weekend and would already change how every subsequent experiment is
judged. Log each as an experiments.md row per CLAUDE.md; the reformulation itself belongs in
PROJECT.md's Key Decisions when adopted ("target moved from 277-way move ID to unified
legal-action policy; winrate vs. baselines added as primary KPI").
