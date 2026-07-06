# DEEP-03 — Leakage & Pipeline-Correctness Audit

> Written 2026-07-05. Scope: `process_replays.py` (full), `feature_engineering.py`,
> `augment_perspectives.py`, split logic in `train_action_predictor_embedding.py`; plus targeted
> checks of `train_switch_predictor.py`, `train_pokemon_switch_predictor.py`, and the inference
> mapping in `predict_action.py` where the prompt's items 3–4 required it. Line numbers refer to
> current working-tree files.

---

## 0. Executive summary

The **Run-4 temporal fix is sound for the move model** — I could not break it. Frozen turn-start
snapshots, post-snapshot reveal updates, and previous-turn `last_move` are all correctly ordered,
and **split integrity holds** even under perspective augmentation (replay_id survives the swap, so
GroupShuffleSplit keeps both perspectives of a battle on the same side; the trainer's overlap
asserts would catch a regression). The headline Run-7 numbers are not inflated by a same-turn
temporal leak.

But the audit found nine real issues. The three worst are not in the move-model path, which is
why they've stayed invisible:

| # | Severity | Where | One-liner |
|---|---|---|---|
| F1 | **Critical (live bot)** | predict_action.py:756 | `player_prefix` is undefined → NameError → every turn falls back to a random move whenever Smogon stats are loaded |
| F2 | **High (training data)** | augment_perspectives.py:26-36 | `last_move_p1/p2` are never swapped → "my last move" is the opponent's in every augmented row |
| F3 | **High (switch-target model)** | train_pokemon_switch_predictor.py:422-440 | `shift(-1)` labels don't check the actor → model often trained to predict the *opponent's* switch |
| F4 | **High (switch models) / n/a (move model)** | process_replays.py:766, turn-0 path | Mid-turn decisions (post-KO replacements, pivot targets) get stale turn-start state; turn-0 lead rows leak their own action |
| F5 | Medium (eval hygiene) | feature_engineering.py:273-293; trainer:387-393 | Feature pruning and class filtering computed on the full dataset before the split (test-set peeking) |
| F6 | Medium (blocks plans) | process_replays.py:427-435, 796-799 | `battle_winner` is null on every emitted row — the column is dead; DEEP-02's outcome weighting has nothing to join on |
| F7 | Medium (REQ-02 parity) | predict_action.py:727/761/785 vs parser | Multiple train/inference normalization mismatches — opponent species arrive Title-cased and forme-unstripped at inference |
| F8 | Low-Med (label semantics) | process_replays.py:722-746 | Forced non-decisions (choice-locked, Encore, Struggle, Sleep Talk's called move) recorded as free choices |
| F9 | Low (coverage/rare) | various | Volatile statuses untracked; Zoroark `|replace|`, Court Change, Revival Blessing edge cases |

Fixes are ranked in §7. F1 and F2 are each a few lines and should be fixed before the next bot
session / training run respectively.

---

## 1. Temporal integrity (prompt item 1) — PASS for move rows, FAIL for mid-turn switch rows

### 1.1 What the Run-4 fix does, verified line by line

- `|turn|` handler deep-copies the whole state into `turn_start_state` **before any of the
  turn's events resolve** (process_replays.py:418-425), and freezes `last_move` as of the
  previous turn's end (:409, :425).
- When an action is recorded, the snapshot is built from `turn_start_state`, not live state
  (:762-773). Both players' move rows in a turn therefore see the identical pre-turn board —
  the correct observation set for simultaneous choice.
- The acting player's move is added to `revealed_moves` and `last_move_seen` **after** the
  snapshot is appended (:775-782), so a move never appears in its own row's features, and the
  second mover's row does not contain the first mover's same-turn move. ✅
- Damage, boosts, status, tera, hazards, weather all mutate `current_state` only; end-of-turn
  residuals land in `current_state` and are correctly visible from the *next* turn's snapshot. ✅
- `battle_winner` is set on `current_state` only after `|win|`, which occurs after the last
  appended snapshot — no outcome leak into any row (see F6 for the flip side). ✅

**Conclusion: for `move:` rows — the entire training population of the embedding model — I found
no temporal leak.** The 12.6% Run-7 gap is not explained by same-turn leakage.

### 1.2 F4 — where temporal integrity *does* fail: actions decided mid-turn

The snapshot rule "state = turn start" is only correct for decisions *made* at turn start.
Two classes of recorded actions are decided mid-turn:

- **Post-KO replacement switches.** A `|faint|` resolves during turn N's events; the replacement
  `|switch|` line arrives before `|turn| N+1`, so its row gets `base_state = turn_start_state`
  (:766) — the board *before* the KO. The row claims the fainted mon is alive, active, at
  pre-hit HP, while the label is the replacement choice the player made looking at a post-KO
  board. This is a large fraction of all switch actions.
- **Pivot switches (U-turn / Volt Switch / Flip Turn / Parting Shot targets).** Same mechanism:
  the incoming-target choice is made after seeing the turn's damage, but the row shows the
  pre-turn board.

This is *stale-state mislabeling* rather than future-information leakage — arguably worse,
because it teaches the switch models systematically wrong state→choice mappings instead of
inflating their metrics.

- **Turn-0 lead rows are worse.** Before the first `|turn|`, `turn_start_state` is `None`, so
  the fallback uses **live** `current_state` (:766) — and for a `|switch|` line the state-update
  branch (:596-630) has *already executed* in the same loop iteration before action recording.
  Net effect: p1's lead row contains its own action applied (the lead is already active —
  direct label leak), and p2's lead row additionally contains p1's lead — future information
  for what is a simultaneous team-preview decision.
- **Mitigations in place today:** the embedding move model filters to `move:` rows, so it never
  sees any of these. `train_switch_predictor.py` defaults `--min_turn 1` (:1057), excluding
  turn-0 rows from the binary model. The switch-target model's exposure is worse — see F3 — and
  its `min_turn` filtering happens *after* the shift-based label construction, so turn-0 rows
  can still have served as label sources.

**Fix direction:** for `switch` actions occurring after a faint or pivot, snapshot
`current_state` (the decision-time board) instead of `turn_start_state`, and emit an
`action_was_forced` flag (DEEP-02 §5.3 wants this column anyway). Do not record turn-0 lead
switches as action rows at all — lead selection is a team-preview problem with a different
observation set, not a turn decision.

---

## 2. Split integrity (prompt item 2) — PASS

- `augment_perspectives.py` renames only `p1_*`/`p2_*` columns (:26-36); `replay_id` passes
  through untouched. Both perspectives of a battle share one `replay_id`.
- `train_action_predictor_embedding.py` groups by `replay_id` in both GroupShuffleSplits
  (:433-450) and asserts zero overlap of replay sets across train/val/test (:456-458). Both
  perspectives of a battle therefore always land on the **same side** of every split. ✅
- Even the degenerate case — feeding a directory containing both the original and the augmented
  parquet, duplicating p1 rows — cannot cross the split boundary (same `replay_id` ⇒ same side);
  it would only double-weight those battles.

**One caveat outside the split mechanism's control:** if the corpus mixes sources (own scrapes +
metamon exports) and the *same ladder battle* appears under two different replay-id formats,
grouping cannot see the duplication. That's a dataset-provenance question → DEEP-06's corpus
audit; unverifiable from code alone.

### F2 — the augmentation bug the swap misses (High)

`swap_perspectives` builds its rename map from the prefixes `p1_`/`p2_` only. The columns
**`last_move_p1` and `last_move_p2` don't start with those prefixes and are never swapped**
(:26-36). In every augmented row, the player relabeled as "p1" carries the *original* p1's last
move in `last_move_p1` — i.e., the features "my last move" and "opponent's last move" are
**crossed in 100% of swapped rows**, which is ~50% of an augmented dataset. `battle_winner`
values (`'p1'`/`'p2'` strings) are likewise unswapped, currently moot because of F6.

Impact: `last_move_*` are two of the model's context features (and 32-dim embedding inputs in
the TF model). Half the training data actively teaches the wrong association for them. This
doesn't inflate metrics (it degrades them symmetrically), but it partially poisons whatever
signal last-move context carries — e.g., exactly the choice-lock/momentum patterns DEEP-02
cares about. Whether Run 7 was trained on augmented data is unrecorded (the experiments.md
dataset column for Run 7 is "❓" — this audit is another reason to fix that); if it was, a rerun
after the fix is warranted.

Fix: add `{'last_move_p1': 'last_move_p2', 'last_move_p2': 'last_move_p1'}` to the rename map,
and swap `battle_winner` values if F6 is ever fixed. Three lines.

---

## 3. Label correctness (prompt item 3) — PASS for parser, FAIL for switch-target trainer

### 3.1 Parser-side attribution — correct

- `|move|` rows: actor from the identifier (`parse_pokemon_identifier`, :728), label
  `move:{normalize_move_name(...)}` (:733-734), snapshot from the same turn's frozen state,
  `player_to_move` set to the actor (:772). Move names share one normalizer with
  `revealed_moves` storage (:733 vs :782 — same `normalize_move_name` output). No off-by-one:
  state is turn-start, action is that turn's choice, `last_move_*` is previous-turn. ✅
- `|switch|` labels use `normalize_species_name` (:758), identical to how slot species are
  stored (:620), so label space and feature space agree. `|drag|` (Roar/Whirlwind) is correctly
  a state update but **not** an action (:596 vs :748) — forced displacement is never labeled as
  a choice. ✅

### 3.2 F3 — switch-target trainer label bug (High)

`train_pokemon_switch_predictor.py` builds its labels via
`df.sort_values(['replay_id','turn_number']) … df['next_action'] = df['action_taken'].shift(-1)`
(:422-429), keeps rows where `next_action` starts with `switch` in the same replay (:433), then
filters the *feature* rows to `player_to_move == 'p1'` (:456). Two defects:

1. **The actor of `next_action` is never checked.** Every turn emits two rows (p1's and p2's
   action). The row after p1's turn-N row is frequently *p2's* turn-N row — so the label
   attached to p1's state is **p2's switch**. The `player_to_move` filter constrains only the
   feature row, not the label row. A large, unquantified fraction of this model's training
   labels are the opponent's switch decisions.
2. **The sort is not stable** (`sort_values` defaults to quicksort). Rows within one
   (`replay_id`, `turn_number`) key can be reordered arbitrarily, so even the "which row is
   next" relation is nondeterministic across runs.

The deeper point: **the shift construction is unnecessary post-Run-4.** A `switch:` row's own
snapshot already *is* the pre-decision state (that was the whole point of the temporal fix).
The correct construction is simply: rows where `action_taken.startswith('switch')` and
`player_to_move == 'p1'`, label = that row's own action. That one change fixes actor
attribution, the stability issue, and shrinks the code. (Then F4's stale-state fix determines
*which board* those rows carry — the two fixes compose.)

The binary model (`train_switch_predictor.py`) labels each row with its **own** action (:472)
and is unaffected by F3 — only by F4's stale states.

---

## 4. Normalization parity (prompt item 4 / REQ-02) — FAIL at inference

Training-side normalization is internally consistent (parser stores what the feature builder
and target encoder consume; the Smogon join applies `sanitize_name` at join time on both sides,
feature_engineering.py:331). The breakages are all on the live-inference side,
`map_battle_to_dataframe_row`:

- **F1 (Critical, and bigger than a parity issue):** the Smogon-usage injection block inside
  the p1 slot loop references **`player_prefix`, which is not defined anywhere in the method**
  (predict_action.py:756 — the only other `player_prefix` bindings live in different functions,
  :296 and :937). The moment a mapped p1 Pokémon is found in `smogon_usages_df.index` — i.e.,
  the first turn of any game, for any common species — the method raises `NameError`, which
  `choose_move`'s wrapper catches as "FATAL: Error during master feature mapping" (:1166-1169)
  and **returns a random move**. Consequence: whenever usage stats load successfully, the bot
  most likely plays randomly *every turn* while printing an error line per turn. Any past live
  evaluation of the bot is suspect until this is checked. Note the block is also redundant —
  proper usage injection already happens downstream in `_prepare_data_for_model` (:1015-1034) —
  and wrongly writes `_active_usage_` keys from *bench* slots. The right fix is deletion of the
  block (:750-757), not repair.
- **F7a — opponent species case:** p2 species are emitted as `pkmn.species.title()` (:785 —
  'Greattusk'), while training vocab is lowercase ('greattusk'); p1 uses
  `.title().lower().replace(" ","")` (:727) and is fine. Every opponent Pokémon maps to
  `__UNKNOWN__` / unseen category at inference. For the LightGBM path categories fall to
  default-bin handling; for the future embedding-model wiring this would zero out the single
  most informative input. One `.lower()` fixes it.
- **F7b — revealed moves case:** live moves are Title-cased `m.id.title()` (:761 — 'Knockoff')
  vs. the parser's lowercase `normalize_move_name` ('knockoff'). Downstream multi-hot matching
  is case-sensitive at the `str.get_dummies` / column-name level — live reveal features
  silently miss their columns.
- **F7c — forme stripping is train-side only:** the parser aggressively strips `FORME_SUFFIXES`
  (process_replays.py:54-78 — 'urshifurapidstrike'→'urshifu'); the live mapping does no
  stripping, so forme species miss the training vocab even with correct casing.
- **F7d — tera-type case is luck, not design:** the parser stores the raw log token ('Water',
  :465) while flatten's comment claims it's "already normalized lowercase" (:835 — false); the
  live `.title()` (:742) happens to match. Any future "normalize to lowercase" cleanup on one
  side only would silently break the other. Pick one canon (lowercase) and enforce in both.
- **F7e — two normalizers disagree on punctuation:** `normalize_species_name` removes only
  hyphens/spaces (keeps `'` and `.` — 'sirfetch'd'), `sanitize_name` strips all non-alphanumerics
  ('sirfetchd'), and poke-env ids are fully sanitized. Affects only punctuation species (rare in
  current OU), but it means the parser vocab and live ids can never fully agree until the parser
  adopts `sanitize_name`-equivalent stripping.

REQ-02 ("unify sanitize_name and normalization across all scripts") is therefore **not
currently satisfied**. The systematic column-by-column live audit and the consolidation design
belong to DEEP-08; the five items above are the confirmed instances.

---

## 5. Silent data-quality issues (prompt item 5)

- **F6 — `battle_winner` is a dead column.** Snapshots are appended at action time; `|win|`
  arrives after the last action; the winner is written only to `current_state` (:427-435), which
  is never re-emitted. The finalize comment (:796-799) frames non-broadcast as deliberate
  anti-leak policy — good instinct, but the net result is a column that is `None` in every row.
  Anything downstream that assumes winner labels exist (DEEP-02's outcome-weighted BC, a value
  model, per-outcome analysis) has no data. Fix without touching feature rows: emit a per-replay
  sidecar (`replay_id, winner, p1_rating, p2_rating`) — combine with DEEP-02 §5.2's rating
  parsing since it's the same `|player|`/`|win|` lines.
- **F8 — forced non-decisions labeled as choices.** Choice-locked repeats, Encore/Taunt-coerced
  moves, Struggle, and recharge/locked turns (Outrage) are recorded identically to free
  decisions. Sleep Talk is subtler: the `[from]` tag arrives as a separate pipe segment, so
  parts[3] is the *called* move — the row's label is a move the player never clicked. Together
  these inflate apparent predictability (locked moves are trivially predictable *if the lock
  were observable* — it isn't, since items/volatiles aren't tracked) and blur the "decision"
  semantics DEEP-02's reformulation depends on. The `action_was_forced` flag (§1.2 fix) plus a
  `[from]`-tag check covers most of it.
- **F9a — volatile battle state is entirely untracked.** Substitute, Leech Seed, confusion,
  Taunt, Encore, Disable, partial-trap, Salt Cure, perish count: none are parsed, yet several
  *hard-constrain* the action space (Taunt bans status moves; Encore forces repetition). For a
  move predictor this is missing state that shows up as irreducible error. Not leakage — a
  coverage gap; belongs on DEEP-01's feature roadmap (its §5 stages don't currently include
  volatiles — flag for when the parser is next reopened, Stage 6).
- **F9b — Zoroark/Illusion:** `|replace|` events are not handled; until the reveal, damage,
  moves, and reveals are attributed to the disguised species, and *after* it, no correction is
  applied to the earlier rows of that battle. Rare (Zoroark-Hisui exists in the tier's orbit);
  accept as noise, but worth a parser warning counter.
- **F9c — Revival Blessing:** a `|-heal|p1: Name|…` for a benched fainted mon carries no
  position letter; the handler resolves via `active_slot` (:474) and heals the *active* mon
  instead, leaving the revived one fainted in-state. Rare; noted.
- **F9d — Court Change (`-swapsideconditions`) unhandled** — hazards/screens stay on the wrong
  side afterward. **`-swapboost`/`-copyboost`/`-invertboost` unhandled** — stale boosts after
  rare moves. Low frequency; parser warning counters would quantify.
- **F9e — forfeits/short games:** early forfeits still emit their few rows; no minimum-length
  filter exists. Harmless for leakage; relevant to DEEP-06's corpus definition.
- **F9f — latent forme-suffix risk:** single-character suffixes `'f'`/`'m'` in `FORME_SUFFIXES`
  (:49) will strip the last letter of any ≥4-char species ending in f/m that isn't a gender
  forme. No current OU species triggers it (checked the common tier list); it's a time bomb for
  future dex additions. Cheap guard: only strip if the base is a known dex species (DEEP-01's
  Stage-1 attribute table provides exactly that lookup).

### F5 — pre-split statistics (eval hygiene)

Two selection steps compute statistics over the **full dataset including test replays** before
the split:

- Revealed-move feature pruning: `min_feature_replay_count` counts replays per move over all of
  `X` (feature_engineering.py:273-293) — feature selection informed by test data.
- `min_move_count` target filtering in the trainer (:387-393) — class-list selection informed by
  test data. (The target LabelEncoder also fits pre-split (:413-414), which is benign class
  enumeration, but the *filter* is not.)

The bias is small (these are coarse frequency thresholds, not fitted parameters) but it is
textbook peeking, and it slightly flatters every recorded eval number. Fix: split replay-ids
first, compute both prunings on train-side replays only. Note in experiments.md when fixed so
before/after numbers aren't compared naively.

---

## 6. What is verified sound (for confidence, and to stop re-auditing)

- Turn-start snapshot mechanics, reveal/last-move ordering, simultaneous-choice correctness for
  `move:` rows (§1.1).
- Group split + augmentation interaction; overlap asserts (§2).
- `battle_winner` neither leaks into features (never selected in `build_medium_X`) nor exists in
  rows at all (F6).
- Hazard layer caps (Spikes 3 / T-Spikes 2 / others 1, :669), Defog/Rapid Spin removal (:677-694),
  weather/terrain lowercase normalization (:633-639), status lowercase (:497) with matching
  `-curestatus` comparison (:521).
- Boost clamping to ±6 (:540), Belly-Drum `-setboost` (:545), Haze `-clearallboost` (:574),
  switch-out boost reset with tera persistence (:625-627).
- Vocab encoders and StandardScaler fit on train only (trainer :492, :496-510); class weights on
  train only (:521-525).

---

## 7. Ranked fix list

| Pri | Finding | Fix | Effort | Invalidates past results? |
|---|---|---|---|---|
| 1 | F1 NameError → random bot | Delete the redundant usage-injection block (predict_action.py:750-757); injection already occurs at :1015-1034 | minutes | Any live-bot impressions/win estimates |
| 2 | F2 last_move not swapped | Add the two-entry swap to `rename_map` in augment_perspectives.py; regenerate augmented parquet | minutes + re-augment | Runs trained on augmented data (Run 7? — dataset unrecorded) |
| 3 | F3 switch-target labels | Drop the shift(-1) construction; use own `switch:` rows directly | ~1 h | Switch-target model entirely |
| 4 | F5 pre-split pruning | Split replay-ids before pruning/class filters | ~half day | All recorded evals, slightly (re-baseline on next run) |
| 5 | F4 stale mid-turn state + turn-0 leads | Decision-time snapshot for post-faint/pivot switches; skip turn-0 lead rows; add `action_was_forced` | ~1 day, **re-parse** | Both switch models' training data |
| 6 | F6 winner sidecar (+ratings) | Per-replay sidecar emit in parser | with #5's re-parse | Nothing (new capability) |
| 7 | F7a-e parity | One-liners now (.lower(), move case); full canon unification in DEEP-08 | minutes now / DEEP-08 later | Live bot behavior |
| 8 | F8/F9 | `[from]` handling, forced flag, warning counters, volatiles → DEEP-01 Stage 6 scope | with next re-parse | — |

Items 5–6 (and ideally 8) share one re-parse batch — the same batch DEEP-01 Stage 6 and DEEP-02
§5 already planned (ratings, items/abilities, forced flags). That consolidated re-parse is now
justified by four separate documents; it should be scheduled as its own phase task in
ROADMAP.md rather than done piecemeal.

Per CLAUDE.md: when fixes land, log the re-parse dataset lineage in experiments.md (this audit
makes the "❓ unrecorded" rows an active liability — F2's blast radius is unknowable *because*
Run 7's dataset is unrecorded), and record the F1/F3 fixes in PROJECT.md's Key Decisions.
