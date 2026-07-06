# DEEP-04 — Evaluation Methodology: The North-Star Metric

> Written 2026-07-05. Consistent with DEEP-01 (§4 measurement protocol — this doc absorbs and
> operationalizes it), DEEP-02 (winrate as the honest KPI; calibration matters because the model
> will be consumed as a distribution, not just an argmax), and DEEP-03 (two caveats inherited:
> every recorded eval number is *slightly* optimistic due to pre-split pruning (F5), and **no
> bot-level evaluation is meaningful until the F1 NameError fix lands** — the live bot has
> likely been playing random moves).

---

## 0. TL;DR

- **Project north-star: win-rate vs. poke-env's `SimpleHeuristicsPlayer`, 400 games on a local
  Showdown server.** It is the only metric that measures "plays well," it's free, deterministic
  to set up, and sensitive across the whole skill range we currently occupy. Provisional target:
  **≥ 55%** (calibrate after the first measurement).
- **Per-run model metric (the cheap proxy you optimize between winrate measurements): clean test
  cross-entropy** — log-loss *without* the label-smoothing term, computed on natural
  (unweighted) labels. It is a proper scoring rule, it serves both consumer roles (argmax policy
  *and* opponent-distribution for the DEEP-07 search), and it is far lower-variance than
  accuracy for Optuna.
- **Raw top-1 stops being a target** (DEEP-02: the ceiling is ~50–60% and the distance to it is
  mostly irreducible); **top-5 retires entirely** (at ~4-option menus it saturates and measures
  moveset knowledge the Smogon prior supplies for free — its Phase-5 target of 75% was already
  dead on arrival at 84.5%).
- Every model run reports an **8-number dashboard** (§5) against **4 frozen baselines** (§4.2),
  logged as a standardized experiments.md row. Two small scripts (`evaluate_action_model.py`,
  `evaluate_bot.py`) constitute the whole harness.
- **Phase 5's acceptance criteria should be rewritten** (§2.3): "val_acc > 40%, top-5 > 75%"
  becomes "clean val CE beats the frozen Run-7 reference by ≥ 0.05 nats **and** skill margin
  over the species-majority baseline improves" — with the winrate gate reserved for
  bot-touching phases.

---

## 1. What to measure — predictor vs. bot (prompt item 1)

The project evaluates two different objects and conflating them caused the current metric
muddle:

### 1.1 The predictor: distribution quality, not exact-guess rate

The model estimates π(action | state). Given DEEP-02 §1.2 (mixed strategies, population
heterogeneity, hidden information), the honest question is "how good is the *distribution*,"
not "how often does argmax equal the human's click."

- **Log-loss (clean CE) — primary.** The proper scoring rule for distribution estimation;
  strictly decreases with genuine improvement; usable at every scale. Two implementation notes:
  (a) Keras `val_loss` currently *includes* the label-smoothing term, so it isn't comparable
  across smoothing settings — log a separate clean `keras.metrics.CategoricalCrossentropy`
  (no smoothing) as the reported number; (b) it is only meaningful once the
  `class_weight='balanced'` distortion is removed (DEEP-02 §1.3 — the fitted distribution
  should estimate π, not a reweighted fiction).
- **Top-1 accuracy — reported, not targeted.** Keep it for continuity with Runs 0–7 and
  because it's legible. Do not tune toward it: near the entropy floor it rewards prior-sharpening
  over state understanding.
- **Top-5 — sanity check only.** With effective menus of ~4–5 moves, top-5 ≈ "did the model
  identify the moveset," which the usage prior already provides. It saturated (84.5%) the
  moment embeddings landed. Watch it only for *regressions* (a drop signals something broke in
  reveal/usage features).
- **Menu-restricted top-1 — the state-reasoning residual (DEEP-01 §4.3).** Accuracy computed
  only among the moves the model itself ranked in its top-5 (or, better post-DEEP-02, among the
  candidate set). This isolates "given you know the menu, can you read the board?" — the number
  this whole project is trying to move. Currently ~38.5/84.5 ≈ 46% — barely above
  uniform-over-menu; that framing should appear on every run report.
- **Skill margin — the anti-lookup meter.** Test top-1 minus baseline B3's top-1 (per-species
  train-majority move, §4.2) *on the same rows*. A model that only memorizes species priors has
  margin ≈ 0 regardless of its raw accuracy. This is the single best scalar for the project's
  stated "generalization over memorization" principle.
- **Calibration — ECE (15-bin) + a reliability curve per milestone run.** Nobody looks at this
  today, but the DEEP-07 search consumes probabilities; an over-confident predictor makes
  expectimax reckless. Cheap to compute; becomes load-bearing the day the bot goes 1-ply.
- **Per-class/macro metrics — explicitly rejected as targets.** Under natural 277-way imbalance,
  macro-averaging demands the model inflate rare-move probabilities — the exact distortion
  DEEP-02 §1.3 diagnosed in the class-weight bug. Use per-class numbers *diagnostically* only
  (e.g., top-20 most-confused move pairs among same-menu alternatives), never as an objective.

**Situation-conditioned slices** (each is the same dashboard computed on a subset):

| Slice | Why | Available |
|---|---|---|
| Turn buckets: 1, 2–5, 6–15, 16–30, 31+ | Lead/early game is prior-driven; endgame is state-driven. A generalizing model should *gain* relative to baselines as turns increase | now |
| Species-frequency buckets (log-scaled by train count) | DEEP-01 §4.4's frequency-slope: memorizers are great on common mons, awful on rare ones | now |
| Free vs. forced decisions | Choice-locked/Encored/Struggle rows (DEEP-03 F8) are trivially predictable *noise*; free decisions are the real test | after the `action_was_forced` re-parse |
| High-rated seats only (≥1500, ≥1700) | "Agreement with strong play" — cheap proxy for play quality | after the ratings re-parse |
| Held-out-species arm | DEEP-01 §4.2's headline generalization test | when the Stage-3 arm runs |

### 1.2 The bot: winrate, full stop

Next-move accuracy cannot capture compounding drift, exploitability, or the value of the
decisions the model gets *confidently* right vs. wrong (DEEP-02 §4). The bot is evaluated by
playing games — §3.

---

## 2. The tuning objective (prompt item 2)

### 2.1 Recommendation: minimize clean validation cross-entropy

- **Why not val_accuracy:** a 0/1 metric on a ~4-way effective decision is high-variance per
  epoch and per trial; Optuna's pruner will kill good trials on accuracy noise. Accuracy also
  can't see distribution quality below the argmax.
- **Why not top-5:** saturated (§1.1); optimizing it tunes toward a solved sub-problem.
- **Why not a composite:** composites need weights, weights need justification, and every
  component worth including is monotonically improved by CE anyway. Complexity without benefit.
- **Why CE specifically:** proper scoring rule → improving it improves *whatever* the model is
  later used for (argmax policy, top-k filtering, expectimax input). It is the pre-metric of
  both consumer roles identified in DEEP-02.

So the current Optuna `objective` (train_action_predictor_embedding.py:535-586) minimizing
`val_loss` is *directionally right* — with three concrete fixes:

1. **Return clean CE, not smoothed loss** (else trials with different `label_smoothing` are
   incomparable, and the reported number overstates loss by roughly the smoothing entropy term).
2. **Log the full dashboard per trial** via `trial.set_user_attr(...)` (top-1, top-5, menu-top-1,
   best epoch, epochs run, wall-clock). The prompt's complaint — "best trial's accuracy was
   never even recorded" — is fixed by five lines here. The sqlite study (`tuning_history.db`)
   then contains everything; add a tiny `tuning_report.py` that dumps the study to a table.
3. **Fix the val set before tuning:** all trials must share the identical GroupShuffleSplit
   (they do today via `random_state=42` — keep it that way deliberately, and say so in the
   experiments.md row).

### 2.2 Guardrail: tune on CE, *gate* on the dashboard

A hyperparameter search can reduce CE by sharpening priors. After tuning, the champion trial
must not *regress* skill margin or menu-restricted top-1 versus the pre-tuning model by more
than noise. If it does, the search found prior-sharpening, not understanding — reject and
constrain the search space.

### 2.3 Reconciling Phase 5's acceptance criteria (ROADMAP.md:33)

"Validation Accuracy > 40%, Top-5 > 75%" has one dead criterion (top-5 achieved before Phase 5
began) and one misaimed one (40% val-acc is reachable via better prior-fitting without a single
point of board-reading improvement — and sits uncomfortably close to the DEEP-02 entropy
ceiling to be a *tuning* deliverable). Replace with:

> **Phase 5 acceptance (revised):** starting from the frozen Run-7 reference evaluation
> (re-measured once with clean CE and no class weights):
> (a) clean val CE improves by ≥ 0.05 nats;
> (b) skill margin (vs. B3) does not decrease, and
> (c) menu-restricted top-1 does not decrease.
> Report the full §5 dashboard for the champion.

Raw top-1 will very likely also improve (dropping class weights alone should help), but it is
no longer the goalpost. When this lands, update ROADMAP.md and note the criteria change in
PROJECT.md Key Decisions per CLAUDE.md.

---

## 3. Evaluating the bot (prompt item 3)

**Precondition: DEEP-03 F1 (predict_action.py:756 NameError → random fallback) must be fixed
first; until then all bot games measure a random player.** Also log a fallback counter per game
(DEEP-08 will formalize this) so a silent regression to random can never invalidate an
evaluation again.

Feasible ladder for a solo dev, in order of cost:

1. **Fixed-baseline gauntlet (the workhorse — hours to build, minutes to run).**
   Local Showdown server + poke-env's `cross_evaluate`. Opponents, frozen forever as the
   reference set: `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`.
   Sample sizes: 400 games vs. SimpleHeuristics (95% CI ≈ ±5 pp at p=0.5) for the north-star
   number; 100–200 vs. the weaker two (they only need to confirm dominance).
   Fix both teams (the bot's current fixed team) so results are comparable across runs; add a
   second team later as a robustness slice, not a variable.
2. **Champion A/B (self-play between versions).** New bot vs. previous-best bot, same teams,
   400 games. This is the *decision* metric for "ship the new model into the bot" — beats
   arguing from offline deltas. Costs nothing beyond running two processes.
3. **High-rated agreement (offline, free).** Once ratings are parsed (DEEP-03 §7 re-parse):
   dashboard computed on ≥1700-rated seats only. This is the cheap stand-in for "agreement
   with strong play" — no GM-annotated dataset needed.
4. **Ladder Elo (sparingly — milestone-only).** Real-ladder games are the ultimate test but:
   noisy (±100+ Elo swings over tens of games), slow (timer-bound, ~10–15 games/hour),
   confounded (opponent pool drifts), and account-sensitive. Use ~50-game ladder sessions at
   milestone boundaries for a reality check, never for iteration. Record the Elo trajectory in
   experiments.md but hang no acceptance criteria on it until games-per-session can exceed ~200.
5. **Regret vs. a reference policy — rejected.** Requires an oracle (search+value stack) we
   don't have; revisit only after DEEP-07's bot exists, at which point "1-ply bot vs. clone
   bot" A/B *is* the regret measurement in practice.

---

## 4. The minimal harness (prompt item 4)

Two scripts + one convention. Everything lands in `models/`-adjacent JSON and experiments.md
rows; no dashboards, no services.

### 4.1 `evaluate_action_model.py` (offline; runs after every training run)

Input: model artifacts + a test parquet (or the split spec). Output: one JSON block
(`models/eval_<run_id>.json`) + a formatted experiments.md row printed to stdout.

Computes: clean CE, top-1, top-5, menu-restricted top-1, ECE(15), skill margin vs. each
baseline, and every §1.1 slice available for the dataset's columns. Baselines are computed
in-script on the same rows (they're all trivial lookups — no training):

### 4.2 The frozen baselines

| ID | Baseline | What it isolates |
|---|---|---|
| B0 | Uniform over classes | absolute floor / sanity |
| B1 | Global train move-frequency | "no state, no species" |
| B2 | Smogon argmax per active species | "public prior only" — what the inference filter already knows |
| B3 | Per-species train-majority move | **the memorization ceiling** — a pure lookup table's score; skill margin = model − B3 |

B2/B3 are DEEP-01 Stage 0's "prior-only floors" — this section is where they live permanently.
B3 must be fit on **train-split rows only** (DEEP-03 F5 discipline).

### 4.3 `evaluate_bot.py` (online; runs before any "ship to bot" decision and at milestones)

Spins up/expects a local Showdown server, runs the §3.1 gauntlet + optional champion A/B,
outputs winrates with Wilson CIs per matchup + fallback-event counts, appends to a
`bot_evals.md` table (or an experiments.md section).

### 4.4 The experiments.md row (standardized)

Extend the current table with fixed columns so runs stay comparable:

```
| Run | Strategy | Dataset (exact file) | Clean CE (val/test) | Top-1 | Menu-Top-1 | Skill vs B3 | Top-5 | ECE | Gap | Duration | Bot WR vs SH |
```

Rules: the Dataset cell is mandatory (three "❓ unrecorded" rows in the current table are
already an active liability per DEEP-03); `Bot WR` stays "—" unless `evaluate_bot.py` ran;
baseline rows (B0–B3) get their own one-time entries so every margin is checkable.

### 4.5 One-time re-baselining

Because DEEP-03's fixes (F5 pruning hygiene, class-weight removal, F2 re-augmentation) each
shift numbers slightly, the first action after building the harness is: **re-evaluate the
frozen Run-7 artifacts and the B0–B3 baselines under the new metrics** and record those rows.
That snapshot becomes the reference all Phase-5+ deltas are measured against. Do not compare
new-metric runs against the old 38.5%/2.567 numbers.

---

## 5. North-star + dashboard (prompt item 5)

**North-star (project level): win-rate vs. SimpleHeuristicsPlayer, 400 games, fixed teams,
local server.** Provisional target ≥ 55%; recalibrate after the first honest measurement
(post-F1-fix, which may reveal the current true level is well below 50%). This is the number
that appears in STATE.md and defines milestone success.

**Per-run dashboard (all cheap, all from `evaluate_action_model.py`):**

1. Clean test CE ← the per-run optimization proxy
2. Skill margin vs. B3 ← the anti-memorization meter
3. Menu-restricted top-1 ← the board-reading meter
4. Top-1 (continuity)
5. Top-5 (regression tripwire only)
6. ECE ← readiness for search consumption
7. Train/val clean-CE gap (continuity with the project's founding metric)
8. Turn-bucket margin trend (does the model beat B3 by *more* in late game?)

Secondary, when available: held-out-species delta (DEEP-01 arm), high-rated-slice dashboard,
forced/free split, champion-A/B winrate.

**Decision rules, stated once:** models iterate on (1); a model is *interesting* if (2) and (3)
move; a model **ships to the bot** only if champion A/B winrate ≥ 50% − CI; a milestone is
**done** only on the north-star. Hyperparameter tuning may never trade (2)/(3) down for (1)
(§2.2 gate).

Per CLAUDE.md, adopting this doc means: ROADMAP.md Phase 5 criteria rewritten (§2.3), the
experiments.md table gains the §4.4 columns, PROJECT.md Key Decisions gets "north-star =
winrate vs SimpleHeuristics; tuning objective = clean CE," and STATE.md tracks the north-star
number once first measured.
