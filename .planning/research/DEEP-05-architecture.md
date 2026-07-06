# DEEP-05 — Model Architecture Review: The Middle of the Network

> Written 2026-07-05. Inputs are fixed by DEEP-01 (attribute bundles, relational features),
> outputs by DEEP-02 (unified candidate-scoring policy); this reviews everything between, under
> hard feasibility constraints measured on this machine (§4.4): **TF 2.19 on Windows runs
> CPU-only — the RTX 3060 Laptop GPU (6 GB) is idle during training** — with 16 GB system RAM.
> Metrics language follows DEEP-04 (clean CE, skill margin, menu-restricted top-1).

---

## 0. TL;DR / Opinionated summary

1. **The flat-concat MLP was the right scaffold and is the wrong endgame.** It won Run 7 because
   embeddings + a smooth shared function class beat per-leaf tree memorization — not because
   three Dense layers are a good match for the game's structure. Keep it as the control arm;
   evolve, don't rewrite.
2. **Found a concrete architectural flaw while auditing `build_embedding_model`:** the 14
   species columns share one vocabulary encoder (`build_vocab_encoders` fits one LabelEncoder
   for the group) but each gets its **own separate `Embedding` table**
   (train_action_predictor_embedding.py:202-205 creates `emb_{col}` per column). The model
   learns what Great Tusk is 14 times; bench-slot tables see each species too rarely to train
   well; species parameters alone are ~14× larger than needed. Same duplication for status
   (14 tables), HP (14), type, field. **Sharing one table per group is a ~1-day change and the
   single highest-value/lowest-risk architecture fix available.**
3. **Snapshot beats sequence for now — because Pokémon is nearly Markov given good state.**
   Most "history" is already accumulated in-state (reveals, boosts, hazards, tera). The biggest
   genuinely-missing history is DEEP-03 F9a's untracked volatiles (Sub, Taunt, Encore, locks) —
   which are *snapshot features* awaiting a parser fix, not a reason to build a GRU. Sequence
   models are deferred behind three cheaper things (§1).
4. **The set encoder over the 12 slots is worth building — and it is the candidate scorer's
   trunk.** A shared per-Pokémon encoder + pooled bench is permutation-invariant over a true
   symmetry of the game, multiplies the effective training signal per species, and the same
   entity encoder later scores switch candidates in the DEEP-02 unified head. Build it once,
   use it twice (§2, §5).
5. **Deep has not yet earned its complexity on equal terms.** Run 7 changed encoding *and*
   architecture together; the heavily-regularized LightGBM baseline (num_leaves 24, depth 5,
   cat_smooth 500, colsample 0.15 — train_action_predictor.py:329-339) has never seen the
   DEEP-01 attribute features. The ablation grid in §3 is mandatory before further deep
   investment.
6. **Fix the training environment before any expensive architecture.** TF ≥ 2.11 has no native
   Windows GPU support; every run to date was CPU. Either WSL2 (keeps the TF code) or a PyTorch
   port (native Windows CUDA) unlocks the 3060. Final call belongs to DEEP-09, but every
   recommendation below is sized to remain feasible on CPU, with the sequence option explicitly
   gated on GPU access (§4.4).

---

## 1. Snapshot vs. sequence (prompt item 1)

### 1.1 How much signal actually lives in history?

Decompose "history the snapshot can't see":

| History component | Already in snapshot? | Best carrier |
|---|---|---|
| What moves/mons the opponent revealed | ✅ `revealed_moves`, slot species/HP/status | — |
| Accumulated board changes (hazards, boosts, tera, weather) | ✅ | — |
| Volatile constraints: Sub, Taunt, Encore, Disable, partial trap, perish, sleep turns | ❌ — **parser gap (DEEP-03 F9a), not a model gap** | snapshot features after re-parse |
| Choice-lock state ("locked into Earthquake since turn 12") | ❌ but *derivable*: last move + hasn't switched since + choice-item likelihood (DEEP-01 item priors) | engineered feature |
| Within-game opponent tendencies (doubles a lot, sacks early, protect-scouts) | ❌ genuinely sequential | history summaries or sequence model |
| Momentum/tempo (who's been forced to react) | ❌ partially proxied by hazards/HP totals | history summaries |

The first four rows — the bulk of the signal — are snapshot-shaped. The genuinely sequential
residual (rows 5–6) is real but modest at mid-ladder, where play is less stylistically
consistent; DEEP-02's population-heterogeneity argument cuts here too: a per-game
tendency-reader mostly pays off against *consistent* opponents.

### 1.2 The three options

- **A. Keep snapshot (baseline).** Zero cost. Correct default while DEEP-01/02/03 work lands.
- **B. Snapshot + engineered history summaries (recommended, cheap).** ~10 columns computed
  per-row at feature-build time from the same replay's earlier rows (train) / tracked live
  (bot): opponent switch count and switches-per-turn, protect uses, last-3 opponent moves
  (3 extra move-embedding inputs), turns-current-mon-has-stayed-in, `likely_choice_locked`
  flag, damage-taken trend over last 3 turns, boosts-lost-to-phazing count. All trivially
  paritable in `predict_action.py` since the bot already tracks per-turn events
  (`_update_last_moves` exists; extend it). Expected gain: small but real — order +0.02–0.05
  clean CE; the choice-lock flag alone should pay for the batch.
- **C. GRU / small transformer over the turn sequence (deferred).** Costs, concretely: dataset
  restructure to per-replay sequences (padding/truncation, new loader), loss masking, live-bot
  stateful sequence tracking (a new parity surface DEEP-08 would have to audit), LightGBM
  comparability lost, and **CPU training cost roughly 5–15×** current — unaffordable before the
  GPU environment fix. Payoff is speculative on top of B (B already captures the cheap 80% of
  rows 5–6).

### 1.3 Recommendation

**A + volatiles-in-parser + B now; C only after** (i) the GPU environment exists, (ii) the
candidate scorer has shipped, and (iii) the DEEP-04 winrate harness shows the snapshot stack
plateauing. If C happens, prefer a 2–4-layer transformer over per-turn *decision tokens*
(the per-turn state encoding from §2 reused as the token) with a 32-turn window — not a
raw-event GRU — so the §2 encoder investment carries over yet again.

---

## 2. Flat concat vs. structured/set encoder (prompt item 2)

### 2.1 What's wrong with the flat concat, precisely

- **Slot order is noise.** Slots are assigned in team-preview/reveal order
  (process_replays.py `|poke|` handling) — an arbitrary ordering the model must memorize
  around. `p1_slot3_species` and `p1_slot5_species` are the same *kind* of thing with
  different weights.
- **No parameter sharing** (finding #2 above): 14 embedding tables per group + 14 separate
  input pathways. Rare species get gradient only from the slots they happened to occupy.
- **Interactions must be rediscovered per slot pair.** "My bench has a water immune to their
  active's STAB" is one fact; the flat MLP must learn it ~36 slot-pair times.

### 2.2 The proposal (one design, two stages)

**Stage A — shared tables (1 day, no topology change):** one `Embedding` per vocab group
(species, move, status, hp, type, field), referenced by all columns in the group. Everything
else identical. Cuts total parameters roughly 3× (~1.5M → ~0.6M, dominated by the 14 species
tables today) and gives every species 14 columns' worth of gradient into one vector. This is
separable from — and should precede — the set encoder, because it isolates "sharing" gains
from "structure" gains in the ablation grid.

**Stage B — per-Pokémon set encoder (2–3 days):** build one token per Pokémon:
`[species emb (or DEEP-01 attribute bundle) ⊕ hp-bin emb ⊕ status emb ⊕ boosts(5) ⊕
flags(active, fainted, terastallized) ⊕ tera-type emb]` → shared 2-layer MLP (~128 wide) →
- the two **active** tokens pass through unpooled (they're special — concatenate),
- the 10 **bench** tokens pool per side with mean+max (permutation-invariant),
- concat [p1_active, p2_active, p1_bench_pool, p2_bench_pool, field/hazard/context vector,
  revealed-move pools] → existing 512→256→128 head.

Buys: bench permutation invariance (a true symmetry), 12× sharing of the encoder, rare-species
robustness, and — decisive — **the entity encoder is exactly what the DEEP-02 candidate scorer
needs to encode switch candidates.** Skip attention for now; add "active attends over opposing
bench" (one attention layer) only if the pooled version shows coverage-related errors, and only
after it's measurable in the DEEP-04 dashboard.

Costs, honestly: input-plumbing rewrite (dict-of-14-columns → stacked `(batch, 12, token_dim)`
tensor + masks for absent/fainted handling), matching changes in the live bot's feature
mapping (a new DEEP-08 parity item), and slot-feature comparability with LightGBM ends (LGBM
keeps eating the flat frame — that's fine, it's a different arm).

---

## 3. Ablation discipline (prompt item 3)

Rule: **one change per arm; every arm on the same dataset (named in experiments.md), same
GroupShuffleSplit seed (42), reported on the DEEP-04 dashboard** (clean CE, skill margin vs B3,
menu-top-1). The grid:

| Arm | Encoder | Features | Purpose |
|---|---|---|---|
| R7′ | flat MLP (as-is) | medium | Run 7 re-run minus class weights, clean CE — **the new reference** (DEEP-04 §4.5) |
| A1 | LightGBM (frozen script's params) | medium | tree baseline on old features (≈ Runs 4/5, re-metered) |
| A2 | **LightGBM** | **DEEP-01 attributes** | **mandatory arm** — if A2 ≈ A4, deep isn't earning complexity |
| A3 | flat MLP + shared tables (§2 Stage A) | medium | isolates the sharing gain |
| A4 | flat MLP + shared tables | attributes | isolates the DEEP-01 encoding gain |
| A5 | set encoder (§2 Stage B) | attributes | isolates the structure gain |
| A6 | A5 + history summaries (§1 B) | attributes + history | isolates the history gain |

Sequencing note: A3 is nearly free once Stage A lands and should run immediately; A2/A4 wait
for DEEP-01 Stages 1–2; A5/A6 follow. If A2 matches A4 within noise, the honest response is to
ship LightGBM on attributes to the bot (it's also the easier parity story) and let the deep
line justify itself at A5 — the set encoder is where an MLP can do things trees structurally
cannot (shared entity encoding, pooling). If even A5 can't beat A2, deep investment pauses
until the candidate scorer (whose per-candidate scoring genuinely doesn't map onto LightGBM).

LightGBM arms use the frozen `train_action_predictor.py` **unchanged** per CLAUDE.md's
stability guardrail — its `--feature_set` input just points at the new attribute parquet;
if any code change is unavoidable, it happens in a copied script, not the frozen one.

---

## 4. Capacity and budget fit (prompt item 4)

### 4.1 Embedding dims

species 48 / moves 32 / others 8 are reasonable *ceilings* and wrong *shapes*: with duplicated
tables (finding #2), 48 dims × 14 tables is over-parameterized in exactly the way that
memorizes; after sharing, one 32-dim species table (vocab ~800 post-forme-stripping) is likely
sufficient — 48 is defensible, tune once at A3. HP/status/field at 8 dims is harmless but
pointless (vocabs of 7–15); 4 dims or even one-hot would do — not worth a run to find out.

### 4.2 Head and batch

512→256→128 + BN + swish + dropout is fine at 1M rows and not obviously limiting at 9M; width
is the *last* thing to tune (A5's structure matters more than 512 vs 1024). Batch 256 on CPU
under-utilizes oneDNN — try 1024 with lr scaled ~2× as a pure-throughput change (validate
against R7′ per DEEP-09's "anything that alters convergence re-validates" rule).

### 4.3 The Optuna search space (current: lr, 3 dropouts, 3 widths)

Wrong priorities: widths are low-leverage, and three independent dropouts mostly add search
dimensions. Post-A3 space, 6 dims total:
`lr` (log 3e-4–3e-3), `weight_decay` (AdamW, log 1e-6–1e-3 — currently absent and the Run-7
next-steps note already wanted it), one shared `dropout` (0.1–0.4), `label_smoothing`
{0, 0.05, 0.1}, `batch` {512, 1024, 2048}, `species_dim` {24, 32, 48}. Objective: **clean val
CE** per DEEP-04 §2 — with all trial metrics logged via `set_user_attr` and the DEEP-04 §2.2
no-regression gate on skill margin / menu-top-1 applied to the champion.

### 4.4 The hardware reality (measured this session)

- `nvidia-smi`: RTX 3060 Laptop, 6144 MiB. `tf.config.list_physical_devices('GPU')` → **`[]`**
  on TF 2.19/Windows (native Windows GPU support ended at TF 2.10). Every training run so far,
  including Run 7 and all Optuna trials, was CPU-bound. RAM 16 GB (Run 5's documented ceiling).
- Implications: (a) everything recommended in §1–3 is deliberately CPU-feasible (sub-1M-param
  models, tabular batches); (b) the sequence option and any candidate-scorer scale-up want the
  GPU; (c) 6 GB VRAM is ample for every model in this document — the constraint is software,
  not silicon. Options: **WSL2 + TF-GPU** (keeps all code; moderate one-time setup) vs.
  **PyTorch port** (native Windows CUDA; a rewrite of ~400 lines that would naturally fold into
  the candidate-scorer script anyway). Decision and benchmarking belong to DEEP-09; DEEP-05's
  input to it: if the candidate scorer is going to be a new script regardless (DEEP-01 Stage 5),
  writing *that* in PyTorch while the current TF stack rides out its remaining arms is the
  path that wastes the least.

---

## 5. Fit with the DEEP-02 endgame (prompt item 5)

| Component | Survives into the candidate scorer? |
|---|---|
| Shared vocab/embedding tables (§2 Stage A) | ✅ verbatim — candidates embed moves/species from the same tables |
| Per-Pokémon entity encoder (§2 Stage B) | ✅ the trunk; switch candidates are *literally* bench tokens it already encodes |
| Numerical/field/context branch | ✅ folds into the state vector |
| History summaries (§1 B) | ✅ state-side features, head-agnostic |
| 512→256→128 classifier head | ⚠️ trunk layers survive; the 277-way softmax is replaced by the per-candidate scorer MLP (small loss — it's ~180k params) |
| 14-column dict input plumbing | ❌ dies with Stage B (good riddance — it's also the ugliest parity surface) |
| Class-weight logic | ❌ dies now (DEEP-02 §1.3) |
| Optuna study as configured | ❌ objective and space replaced (§4.3) |

The through-line: **every recommended change is chosen so the candidate scorer inherits it.**
That's the tiebreaker used throughout — e.g., set-encoder-before-attention, and
transformer-over-decision-tokens rather than GRU-over-events if sequences ever happen.

---

## 6. Ranked recommendations (prompt item 6)

| # | Change | Expected payoff | Cost | experiments.md row |
|---|---|---|---|---|
| 1 | Drop class weights, add clean-CE metric, re-run Run 7 config | calibration fixed; top-1 likely +1–2pp; creates the reference | 0.5 d | **R7′** (new reference) |
| 2 | **Share embedding tables per vocab group** | params ~3×↓; bench slots informative; CE −0.03–0.08 (est.) | 1 d | **A3** |
| 3 | LightGBM arms on old + attribute features | decision information: is deep earning its keep? | 1–2 d (mostly waiting on DEEP-01 Stage 1) | **A1, A2** |
| 4 | Flat MLP on attribute features | isolates DEEP-01 encoding gain | with DEEP-01 Stage 3 | **A4** |
| 5 | Per-Pokémon set encoder, pooled bench | structure gain + the scorer trunk; the biggest plausible single win (CE −0.05–0.15, wide error bars) | 2–3 d | **A5** |
| 6 | History summaries (+ choice-lock flag); volatiles when parser reopens | small-but-cheap; CE −0.02–0.05 (est.) | 1–2 d | **A6** |
| 7 | Optuna space/objective rework (§4.3) | better champions per GPU/CPU-hour | 0.5 d | tuning study v2 |
| 8 | GPU environment (WSL2 vs PyTorch-for-scorer) | unlocks sequence work + fast scorer iteration | DEEP-09's call | — |
| 9 | Sequence model (transformer over decision tokens) | speculative; gated on #8 + scorer + winrate plateau | 1–2 wk | deferred |

Items 1–2 are a weekend and independent of everything else in flight. All payoff estimates are
clean-CE guesses with honest error bars wider than the estimates themselves — which is exactly
why the grid in §3 exists. Log every arm per CLAUDE.md (row + artifact + exact dataset), and
when #5 lands, PROJECT.md Key Decisions gets "state encoder = shared per-Pokémon token encoder
with pooled bench (DEEP-05)."
