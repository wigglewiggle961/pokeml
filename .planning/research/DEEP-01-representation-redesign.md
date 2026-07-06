# DEEP-01 — Representation Redesign: Breaking the Species-Memorization Shortcut

> Written 2026-07-05. Companion to DEEP-04 (measurement). Supersedes the scope restriction in
> `.planning/todos/pending/2026-04-14-implement-explicit-semantic-mapping-for-moves.md` — that todo
> said "do NOT map the species name," but it was written for the now-frozen LightGBM pipeline
> (`train_action_predictor.py`). This document proposes mapping species to attributes in the
> embedding pipeline, which is exactly the experiment that todo deferred.

---

## 0. TL;DR / Opinionated summary

1. **The Run-7 embedding model is a better lookup table, not a different kind of model.** The
   48-dim species embedding is a free identity code; SGD fills it with move-prior sufficient
   statistics because that is the shortest gradient path to reducing loss. The gap collapsed from
   ~50% to 12.6% because a *smooth, shared* lookup table generalizes across replays far better
   than per-leaf LightGBM memorization — not because the model learned game mechanics.
2. **The `p1/p2_active_usage_*` Smogon columns are a second, sneakier identity channel.** A
   species' 100-dim move-usage vector is a near-unique fingerprint. Even with species tokens
   deleted, those columns reconstruct identity — and worse, they hand the model a pre-computed
   answer prior, so it learns to re-rank a prior instead of reasoning from state.
3. **The fix is to change the basis, not to ban information.** Species is legitimately observable;
   humans use it constantly. The goal is that species enters the model *through its mechanics*
   (stats, types, ability, item, role, revealed moves) so that (a) gradients flow through features
   that transfer to unseen/rare species, and (b) the state-dependent part of the decision — the
   part that actually wins games — has to be learned from state.
4. **The single highest-value unbuilt feature group is relational:** type-effectiveness of the
   opponent's revealed moves vs. my active/team, and a speed comparison. The current model sees
   *zero* matchup information; every "does Great Tusk click Headlong Rush or Knock Off here?"
   decision is mediated entirely by identity priors.
5. **All the raw material is already on disk.** `poke_env/data/static/` ships
   `gen9pokedex.json` (base stats, types, abilities), `gen9moves.json` (type, category, basePower,
   priority, flags), and `gen9typechart.json`. `data/gen9ou-0.json` has per-species Abilities,
   Items, Spreads, and Tera-Types distributions we currently ignore. No new data collection needed.
6. **Before building anything, measure the prior-only baseline** (argmax Smogon usage move per
   species, zero state). We have never computed it. If it scores ~30%+ against our 38.5%, it
   proves most current capacity is lookup and gives DEEP-04 its floor. Half a day of work.

---

## 1. Critique: where exactly identity leaks in

### 1.1 Direct leaks

| Leak | Where it lives | Severity |
|---|---|---|
| Active species tokens | `p1_active_species`, `p2_active_species` — built in `build_medium_X` (`base_active_features`, feature_engineering.py:153), consumed as 48-dim embeddings via `EMBED_COLS_SPECIES` (train_action_predictor_embedding.py:56) | **Critical** — the primary shortcut |
| Bench species tokens | `{p1,p2}_slot{1..6}_species`, added in the `bench_cols` loop (feature_engineering.py:159-167), 12 more embedding inputs sharing the species vocab | High — encodes team archetype by ID, not by role |
| Smogon usage vectors | `{p1,p2}_active_usage_{move}` × top-100 moves, injected at feature_engineering.py:322-349 | **Critical and underappreciated** — a per-species answer prior AND an identity fingerprint |
| Free move embeddings | `last_move_p1/p2` (32-dim learned move embeddings) and the `move_pool_p1/p2` Dense(32) over the multi-hot (train_action_predictor_embedding.py:216-225) | Medium — same problem one level down: moves are also IDs with learned free vectors, so nothing ties Flamethrower to Fire Blast |
| Revealed-move multi-hot | ~thousands of `{p1,p2}_active_revealed_move_{move}` binary columns (`new_binary_move_cols`, feature_engineering.py:253-320) | Medium — legitimate observed state, but as raw IDs it doubles as a species fingerprint and shares no strength across similar moves |

### 1.2 Why embedding-per-species only partially fixed it

The LightGBM one-hot models failed *twice over*: species identity was the dominant split feature
(~2.6M importance), and tree leaves memorized replay-specific conditional distributions — hence
the 40–50% gap. Moving to embeddings fixed the **second** failure: an embedding forces all
occurrences of a species through one shared 48-dim bottleneck, so the model can't carve out
per-replay leaves, and rare species shrink toward the shared prior. That's real regularization
and it's why the gap collapsed to 12.6%.

But it did nothing about the **first** failure. Nothing constrains those 48 dimensions to encode
anything about the game. The loss-minimizing content for `emb_p1_active_species[greattusk]` is a
compressed code for "Great Tusk's conditional move distribution" — i.e., the lookup table, now
differentiable. Evidence consistent with this:

- **Top-5 = 84.5% vs top-1 = 38.5% on 277 classes.** Top-5 on a mon with a 4-move set is
  approximately "the model knows the moveset" — which the usage columns literally hand it. The
  hard residual — *which* of the 4 moves given the board — is exactly where top-1 sits, and 38.5%
  over a ~4-option effective menu is barely above uniform-over-menu (~25–30% depending on menu
  size). The state-reasoning component of the current model is small. (§4 defines the metric that
  makes this precise.)
- **Structural argument:** the model has no input from which move-vs-matchup reasoning is even
  *representable*. There is no type-effectiveness feature, no speed comparison, no damage proxy.
  Whatever "state reasoning" it does must be re-derived from raw species/HP/boost tokens per
  species — which is memorization by another name.
- **Failure modes it cannot escape:** a new DLC Pokémon maps to `__UNKNOWN__` (index 0) → the
  model is blind. A meta shift (same species, new set) leaves a stale embedding. A rare species
  seen in 3 training replays gets a noisy embedding with no way to fall back on "it's a fast
  Fire-type with Choice Specs usage."

### 1.3 What is *not* wrong

- The slot-based team structure, hazards, side conditions, field state, boosts, HP binning, and
  the GroupShuffleSplit-by-replay hygiene are all sound. Keep them.
- Using species-conditional information at *inference* is fine and necessary (the bot already
  filters predictions by Smogon usage). The problem is species-as-input-basis during *learning*.

---

## 2. Proposed representation: Pokémon as attribute bundles

**Principle:** every place the pipeline currently emits a species token, emit instead a fixed
**attribute bundle** derived from static data, keyed by the sanitized species name. Species
identity survives only as an optional small residual embedding (8-dim, not 48) that we can ablate
to zero — see §4 for the two-tower placement that keeps it from re-becoming the trunk.

### 2.1 The attribute bundle (per Pokémon)

Built once as a lookup table `species → row` by a new module (see §5, Stage 1). Sources:
`gen9pokedex.json` (poke-env static), `gen9ou-0.json` (Smogon), `gen9moves.json`.

**A. Intrinsic (pokedex):**
- `basestat_{hp,atk,def,spa,spd,spe}` — 6 floats, /255 normalized.
- `type1`, `type2` — two categorical slots into the existing shared type vocab
  (`EMBED_COLS_TYPE` already exists; extend its column list). `type2='none'` for monotypes.
- Derived: `atk_bias = (atk − spa)/(atk + spa)` (physical vs special attacker, in [−1,1]),
  `phys_bulk = hp·def/255²`, `spec_bulk = hp·spd/255²`.

**B. Statistical priors (Smogon — mechanics-space, not move-ID-space):**
- `speed_est` — speed stat computed from base speed + the modal Spread's nature/EVs
  (`Spreads` field, e.g. `Jolly:0/252/0/0/4/252`). This is the number a human actually uses;
  base speed alone misranks half the meta.
- `ability_modal` — categorical, top ability from `Abilities` (vocab ~300, 8-dim embedding).
  Overridden by revealed ability once the parser tracks it (Stage 6, optional).
- `item_class_probs` — collapse `Items` into ~12 mechanic classes and emit the probability mass
  of each: `choice_scarf, choice_band_specs, boots, leftovers_lifeorb_split..., assault_vest,
  sash, weakness_policy, hazmat(rocky helmet etc.), other`. Choice-item probability is the
  single most action-predictive item fact (locked mons repeat moves; scarf flips speed order).
- `role_scores` (~8 floats) — usage-weighted probability the species' set contains a move in
  each mechanic class, computed by joining `Moves` usage with `gen9moves.json` semantics:
  `hazard_setter, hazard_remover, recovery, pivot (uturn/voltswitch/flipturn/partingshot),
  setup_phys, setup_spec, status_spreader, priority_user`. **This is the legitimate replacement
  for the raw usage vector:** it summarizes what the species *does* in mechanics space (~8 dims)
  instead of leaking a 100-dim move-ID fingerprint.

**C. Observed in-battle (already parsed, unchanged):** hp bin, status, boosts, tera state,
revealed moves (re-encoded semantically per §3), is_fainted, is_active.

Apply the full bundle to both actives. For the 12 bench slots, use a reduced bundle (types,
speed_est, atk_bias, bulk, role_scores, hp, status) — bench mons matter mainly as switch-in
threats and win-condition context, not in full detail.

### 2.2 Relational features (the actual point)

A strong human predicts the opponent's move from **the matchup**, not from the species in a
vacuum. All of these are cheap once the bundle exists; none exist today. Priority order:

1. **Type effectiveness, both directions.** Using `gen9typechart.json`:
   - `p2_best_revealed_eff_vs_p1` = max over p2's revealed damaging moves of
     effectiveness(move.type → p1 active typing/tera).
   - `p2_best_revealed_dmg_proxy` = max of effectiveness × basePower × STAB(1.5 if move.type ∈
     p2 types) × attack-stat-side match (uses `atk_bias`).
   - Mirror both for p1's revealed moves vs p2. Four columns, enormous signal: "I threaten a
     4× weak target" ≈ "I click the attack"; "I do nothing to this wall" ≈ "I switch or status."
2. **Speed order.** `speed_edge = sign(p2_speed_adj − p1_speed_adj)` plus the continuous ratio,
   where `_adj` applies boost multipliers, paralysis ×0.5, Tailwind ×2, and flips sign under
   Trick Room (`field_pseudo_weather`). Faster-and-threatening plays differently than
   slower-and-threatened; today the model literally cannot know who moves first.
3. **KO pressure (crude is fine).** `p2_can_likely_ko_p1` ≈ dmg_proxy scaled against defender
   bulk and current HP bin, thresholded; and its mirror. Predicts desperation moves, sacks, and
   free-turn setup better than anything in the current feature set.
4. **Team-level coverage.** For each p1 bench mon: best effectiveness p2's revealed moves achieve
   against it → summarize as `p1_has_safe_switchin` (min over bench of incoming eff < 1) and
   `p2_walled_count`. Whether the opponent *expects a switch* is the main driver of doubles,
   hazard turns, and status fishing.

### 2.3 What this buys

- Unseen/rare species land at a meaningful point in mechanics space instead of `__UNKNOWN__`.
- The model can finally represent the *reason* for a move choice, so the residual
  which-of-the-menu task has learnable structure.
- Honest caveat: the base-stat tuple is itself a near-unique species fingerprint, so a large
  network can still memorize "(131,131,115,53,53,87) → Headlong Rush." Attribute encoding makes
  the shortcut *no easier than* the generalizing path rather than impossible. The defenses are
  (a) the held-out-species eval in §4, which detects it, (b) light Gaussian noise/dropout on the
  attribute inputs during training, and (c) the candidate-scoring formulation in §3.2, which
  structurally routes the decision through move mechanics.

---

## 3. Moves: semantics in, semantics out

### 3.1 Feature side — replace the multi-hot and the free move embeddings

Every move becomes a fixed **semantic vector** from `gen9moves.json`:
`[type (shared type vocab), category one-hot(3), basePower/150, accuracy/100, priority/5,
effect_class one-hot(~14)]` where effect_class ∈ {hazard, removal, recovery, pivot, setup_phys,
setup_spec, status_inflict, screens, protect, taunt_encore(disruption), weather_terrain, phazing,
attack_plain, attack_secondary}. Build the class map once with a small hand-check pass over the
~300 OU-relevant moves (flags + heuristics get ~90% of it; hand-fix the rest — this is a solo-dev
afternoon, and the todo from 2026-04-14 already sketched it).

- **Revealed moves:** instead of `new_binary_move_cols` (feature_engineering.py:253-320), each
  active mon's ≤4 revealed moves → semantic vectors → shared Dense projection → **mean+max pool**
  (set encoder). Kills thousands of sparse columns, ties Flamethrower to Fire Blast for free, and
  the type-effectiveness features in §2.2 consume the same vectors. Interim LightGBM-friendly
  variant (per the pending todo): aggregate counts/maxes per class and type — ~25 dense columns.
- **`last_move_p1/p2`:** keep the token embedding if you like, but *add* the semantic vector of
  that move. "Opponent just clicked a Choice-locked physical Ground move" is the state a human
  tracks; the token alone can't share that across moves.

### 3.2 Target side — from 277-way softmax to candidate scoring (the endgame)

The global 277-class softmax forces the network to rediscover the legal menu on every forward
pass — that is *why* identity priors are so valuable to it. Reformulate:

- Build a candidate set per state: revealed moves ∪ top Smogon moves for the species, capped at
  K≈6 (this mirrors exactly what `predict_action.py` already does with the usage filter at
  inference — we'd be moving that logic into training where it belongs).
- Each candidate = its move semantic vector ⊕ candidate-specific relational features
  (effectiveness vs p1 active, STAB, blocked-by-current-conditions, priority-beats-speed-gap).
- A shared scorer MLP takes [state encoding ⊕ candidate encoding] → scalar; softmax over the K
  candidates; cross-entropy on the observed choice.

This makes the task "pick among these mechanically-described options," which is species-agnostic
by construction, transfers to never-seen moves, and eliminates the 277-class × species-prior
crutch entirely. It is also the natural place to later unify move and switch prediction (switch
targets become candidates too), collapsing the three-model bot into one scorer. It's a new script
(`train_action_scorer.py`), not a modification — do it last (§5 Stage 5), after the feature work
proves out under the softmax model, so feature effects and architecture effects stay separable.

---

## 4. The trade-off, and how to measure it (feeds DEEP-04)

**Expectation, stated up front so we don't panic at the first run:** deleting species tokens +
usage columns will *drop* raw test accuracy, plausibly 38.5% → 33–36%, because the species prior
is genuinely predictive and we're removing it before the state features are strong enough to
compensate. That is the cost of buying generalization. Two mitigations:

- **Two-tower (“prior late”) architecture:** keep an 8-dim species embedding + role priors in a
  side branch that joins at the *last* layer (Wide-&-Deep style, already on the Run-7 next-steps
  list), while all deep layers see only attributes/state. The prior can adjust logits but can't
  become the representation trunk. This should recover most of the raw accuracy.
- **Never remove the inference-time Smogon legality filter** — species knowledge at decision
  time is free and correct; the fight is only over what the network trains on.

**Measurement protocol** (run for every arm: `embed` = current, `attrs_only`, `both`/two-tower):

1. **Prior-only floors (do this first, before any building):**
   (a) argmax Smogon usage move per acting species; (b) per-species majority move from the
   training split. Zero state input. If (b) scores near 38.5%, the current model is quantified
   as a lookup table — and every future model must be judged by its margin above this floor,
   not by raw accuracy.
   *(Correction 2026-07-05: this originally said `p2_active_species`, but training rows are
   filtered to `player_to_move == 'p1'`, so the acting mon is `p1_active_species`.)*
   **✅ DONE 2026-07-05** (`baseline_prior_floors.py`, Run B0 in experiments.md): on Run 7's
   exact test split, (a) = **26.7%** top-1 / 76.1% top-5; (b) = **30.5%** top-1 / **80.6%**
   top-5, vs Run 7's 38.5% / 84.5%. The §0.6 prediction ("~30%+") was right on: Run 7's margin
   over the zero-state lookup is +8.0 pts top-1 and only +3.9 pts top-5.
2. **Held-out-species generalization (headline metric):** exclude from training all replays where
   any of ~10 chosen mid-usage species appear; evaluate accuracy on test turns where the acting
   mon is a held-out species. The identity model degrades to `__UNKNOWN__` performance; the
   attribute model should degrade gracefully. Report Δ vs. in-vocab accuracy.
3. **Menu-restricted accuracy:** accuracy conditional on the true move being in the model's top-5
   (≈ conditional on knowing the moveset). This isolates the state-reasoning residual that raw
   top-1 blurs; it is the number §1.2 argues is currently near-floor, and it is the number this
   whole redesign is trying to move.
4. **Frequency-slope curve:** per-species accuracy vs. species training frequency (log-binned).
   Memorization ⇒ steep slope; attribute reasoning ⇒ flat. One chart, very legible.
5. **Counterfactual identity probe:** on fixed states, swap the species token (embed arms only)
   holding attributes constant; measure mean prediction shift (KL). High shift = identity still
   drives the output beyond what mechanics justify.
6. Keep reporting the train/val gap and val loss as before (experiments.md table).

**Success criterion, concretely:** an `attrs_only` or two-tower model that (i) beats the
prior-only floor by more than the current model does, (ii) loses <30% relative accuracy on
held-out species (vs. the identity model's expected collapse), and (iii) improves
menu-restricted accuracy — even if raw top-1 lands a couple points under 38.5% initially.

---

## 5. Staged implementation path (solo, incremental, no rewrite)

All stages preserve the guardrails: `train_action_predictor.py` untouched; every change lands in
`feature_engineering.py` + `train_action_predictor_embedding.py`; every run gets an
`experiments.md` row with the exact dataset (fix the standing ❓ while at it).

- **Stage 0 — Baselines & bookkeeping (½ day, no model code).** Script the two prior-only floors
  (§4.1) against the existing test split. Log to experiments.md. This alone reframes every number
  we have. **✅ DONE 2026-07-05** — see §4.1 results and experiments.md Run B0. Bonus: the
  standing ❓ on Run 7's dataset lineage was resolved while replicating the split
  (`data/100k.parquet` + `--min_move_count 100`, verified by exact label-encoder class match).
- **Stage 1 — Attribute table (1–2 days).** New `species_data.py` (or a section in
  `feature_engineering.py`): `load_static_data()` reading `gen9pokedex.json` / `gen9moves.json` /
  `gen9typechart.json` — **copy them into `data/static/` from the poke-env install** so the
  pipeline doesn't depend on a site-packages path — plus `gen9ou-0.json`;
  `build_species_attribute_table()` → DataFrame keyed by sanitized species (reusing
  `sanitize_name`; forme-stripping in `normalize_species_name` already matches most Smogon keys);
  `attach_species_attributes(X)` adding `{p1,p2}_active_*` bundle columns (§2.1) and reduced
  bench bundles. Wire into `build_medium_X` behind a flag; feed stats/roles as numericals and
  type1/type2 through the existing type-embed path. Species embeddings stay ON. Run → expect a
  small bump; this validates the join (watch for species that fail the attribute lookup — log
  them, they're forme-normalization bugs).
  **✅ IMPLEMENTED 2026-07-06** (`species_data.py`; `build_medium_X(attach_species_attrs=True)`;
  `--species_attrs` flag). Join validated against `data/100k.parquet`: 99.95%+ row coverage,
  0 unresolved species after the resolver (the dataset has parser artifacts — gender-marker
  truncation like `ambipo`→`ambipom` and cosmetic-forme suffixes like `alcremiematchacrea` —
  handled by `species_data._make_resolver`; `absent` is an empty-slot placeholder, not a bug).
  Bundle = 204 numeric + 30 categorical columns (attr types share the tera-type vocab; modal
  ability gets its own shared 8-dim vocab). **Run 9 (2026-07-06): 46.6% top-1 / 89.6% top-5 /
  menu-restricted 52.0% — +0.3 pts over Run 8, the predicted small bump.** With species
  embeddings and usage columns still on, attributes are near-redundant for accuracy; their value
  is claimed for generalization and must be proven by the Stage 3 ablation + held-out-species
  eval. Stage 2 (relational features) is the stage expected to move raw numbers.
- **Stage 2 — Relational features (1–2 days).** `build_matchup_features(X)` implementing §2.2
  items 1–3 (effectiveness, speed_edge, KO pressure); team coverage (item 4) if cheap. Columns
  like `p2_best_revealed_eff_vs_p1`, `speed_edge`, `p2_can_likely_ko_p1`. Run. This is the stage
  I expect to matter most.
  **✅ IMPLEMENTED 2026-07-06** (`matchup_features.py`; `build_medium_X(matchup_features=True)`;
  `--matchup_features` flag). All four §2.2 items shipped — 12 numeric columns including team
  coverage (`p1_has_safe_switchin`, `p2_walled_count`), which turned out cheap by reusing the
  exploded revealed-move arrays. Validated against all 1.65M rows of `data/100k.parquet` (33s):
  eff values exactly in {0,¼,½,1,2,4}, Trick-Room speed flip exact, KO pressure fires ~16% of
  rows. Typechart decoded: defender-keyed, damageTaken codes 0/1/2/3 = 1×/2×/0.5×/0×.
  **Run 10 (2026-07-06): 46.5% top-1 / 89.7% top-5 / menu-restricted 51.9% — FLAT vs Run 9.**
  The "stage expected to matter most" added nothing *with identity channels still on* — the
  strongest evidence yet for §1.2: the identity prior saturates the softmax model, so even
  brand-new matchup/speed-order information (previously zero) is redundant on top of it.
  Augmenting the identity model is exhausted; Stage 3's `attrs_only` ablation is now the
  decisive experiment (can attributes + matchup *replace* the prior, not decorate it?).
- **Stage 3 — The ablation that answers the question (½ day of code, plus runs).** Add
  `--species_mode {embed, attrs_only, both}` to `train_action_predictor_embedding.py`:
  `attrs_only` drops all `EMBED_COLS_SPECIES` inputs *and* the `{p1,p2}_active_usage_*` columns
  (both leaks, per §1.1); `both` = two-tower with the 8-dim late-joining species branch.
  Implement the §4 measurement suite (held-out species split needs a small change to the
  `GroupShuffleSplit` block; menu-restricted accuracy and frequency-slope are eval-only).
  **This stage produces the DEEP-04 deliverable numbers.**
- **Stage 4 — Move semantics (2–3 days).** Build the move→semantic-vector table + effect-class
  map (§3.1). Replace the multi-hot block in `build_medium_X` (lines 253-320) with pooled
  semantic encodings; add `last_move_*` semantic vectors. Retires ~thousands of sparse columns
  and the pending 2026-04-14 todo (close it, noting it landed here).
- **Stage 5 — Candidate scorer (a week, new script `train_action_scorer.py`, v2 artifacts).**
  §3.2. Only after Stages 1–4 show the attribute basis holds up. Also the natural moment to
  finally wire a modern model into `predict_action.py`, since the scorer consumes the same
  candidate set the bot already builds for its usage filter.
- **Stage 6 (opportunistic) — Parse revealed items/abilities.** `process_replays.py` currently
  ignores `|-item|`, `|-enditem|`, `|-ability|` lines entirely (confirmed by inspection —
  `get_initial_pokemon_state()` at process_replays.py:145 has no item/ability fields). Add
  `revealed_item`/`revealed_ability` to the slot state and handlers for those events. Requires
  re-parsing datasets, so batch it with the next data refresh; until then the Smogon modal
  ability/item-class priors from Stage 1 carry the load.

---

## 6. Decision log hooks

When stages land, per CLAUDE.md: experiments.md gets a row per run (with dataset!), PROJECT.md
Key Decisions gets "representation moved from species-ID basis to attribute basis (DEEP-01)",
and PROJECT_CONTEXT.md's Feature Engineering section must be updated when `build_medium_X`
changes shape. The prior-only floor from Stage 0 should be added to the experiments table as its
own row — it is the number every later run is compared against.
