# DEEP-08 — Live-Bot Parity & Robustness Audit

> Written 2026-07-05. Companion to DEEP-03 (training-pipeline audit; its F1/F7 findings are
> subsumed and extended here) and DEEP-07 (whose search bot inherits every guardrail designed
> below). API behaviors were **verified against the installed poke-env this session**, not
> inferred: `Battle.side_conditions → Dict[SideCondition, int]` (enum keys),
> `Battle.weather → Dict[Weather, int]`, `Battle.fields → Dict[Field, int]`.

---

## 0. Executive summary — the bot is almost certainly playing random moves

Three independent live-path defects each suffice to break or blind the bot; together they mean
**no live game to date reflects the trained models' ability**:

1. **Every-turn crash #1 (DEEP-03 F1):** undefined `player_prefix` at predict_action.py:756 →
   `NameError` during feature mapping whenever the Smogon usage table loaded (it always does) →
   `choose_move`'s catch → random move, every turn.
2. **Every-weather-turn crash #2 (new):** `battle.weather.name.title()` at :881 assumes the old
   poke-env API (`Optional[Weather]`); the installed version returns a **dict**, which is truthy
   when weather is active and has no `.name` → `AttributeError` → random move for the rest of
   any weather game. (Empty dict is falsy, so clear-skies turns survive this one.)
3. **Hazards & screens are always zero (new):** :897-912 checks
   `if cond in HAZARD_CONDITIONS_MAP` — enum keys against a lowercase-string dict; verified
   `SideCondition.STEALTH_ROCK in {'stealthrock': 1} == False`. Even on turns that don't crash,
   the bot has **never seen Stealth Rock, Spikes, Reflect, or Tailwind on either side.**

Beyond these, the audit found a systematic pattern: the live mapper was written against an
older poke-env and a remembered (not shared) normalization spec, then never verified — exactly
the failure mode "Parity is King" was supposed to prevent. §2's consolidation makes the drift
class impossible rather than re-auditable; §3's harness makes regressions loud.

Also flagged in passing (not parity, but found while reading): **hardcoded Showdown credentials
with a real password at predict_action.py:57-58, committed to a git-tracked file.** Rotate the
password, strip the default from the code, and rely on the env vars only.

---

## 1. Column-by-column parity review (prompt item 1)

Training-side truth: parser flatten (process_replays.py:808-901) → `build_medium_X`
(feature_engineering.py). Live side: `map_battle_to_dataframe_row` (predict_action.py:682-917)
→ `_prepare_data_for_move_model`/`_prepare_data_for_model` (:919-1068).

| Column group | Training emits | Live emits | Verdict |
|---|---|---|---|
| `p1_slotN_species` | lowercase, **forme-stripped** (`normalize_species_name`, process_replays.py:54-78) | `.title().lower().replace(" ","")` (:727) — lowercase, **no forme strip** | ⚠️ forme species miss vocab (DEEP-03 F7c) |
| `p2_slotN_species`, `p2_active_species` | same | `.title()` **only** (:785) — 'Greattusk' | ❌ every opponent species is an unseen category (F7a) |
| `*_hp_perc` | int 0-100; binned to labels *inside* `build_medium_X` (training) / expected raw by v4 feature-info | `round(fraction*100)` (:731) | ✅ value-parity; binning timing consistent for v4 (LightGBM feature_info drives it); re-verify per-model in §3 harness |
| `*_status` | lowercase (`brn`, `fnt`) | `.name.lower()` (:732) | ✅ |
| `*_boost_*` | −6..6 ints | poke-env boosts (:746-748) | ✅ |
| `*_terastallized` / `*_tera_type` | int / **raw log case** 'Water' (process_replays.py:465; flatten comment claiming lowercase is false) | `getattr(...,'is_terastallized', pkmn.is_dynamaxed)` fallback (:738 — dynamax as a proxy is wrong-but-rare) / `.name.title()` 'Water' (:742) | ⚠️ tera-type case matches by luck (F7d); the dynamax fallback should die |
| `*_revealed_moves` (strings) | lowercase normalized ids, comma-joined (parser :842) | `m.id.title()` (:761, :805) — 'Knockoff' | ❌ case mismatch in the string columns — **but see next row** |
| `*_active_revealed_move_*` multi-hot | columns named with `sanitize_name` (lowercase) | reverse-engineered: column-stem → `.replace("_"," ").title()` compared against the Title-cased live set (:948, :1000) | 😬 **works by coincidence** — both sides Title-case the same sanitized stem. Zero-margin brittleness: any move whose sanitized id isn't `.title()`-stable, or any future column naming change, silently zeroes features |
| Own-side revealed moves (semantics) | strictly *revealed-so-far* (DEEP-03 §1.1 verified) | `pkmn.moves` for **our own team = all four moves from the team sheet**, turn 1 onward | ⚠️ systematic distribution shift: live p1 reveal features are always "fuller" than training saw. Options: mask to actually-used moves live, or (better, per DEEP-02 — the player *does* know their own moves) retrain with own-side full movesets. Decide once, document in the canon (§2) |
| `field_weather` | lowercase log token ('raindance', process_replays.py:634) | `.name.title()` on a **dict** (:881) | ❌ crashes when weather active (#2 above); even the intended path yields 'Raindance' ≠ 'raindance' |
| `field_terrain`, `field_pseudo_weather` | tracked from log (:637-652) | **hardcoded `'none'`** (:885-889) with a comment claiming poke-env can't see them — false: `battle.fields → Dict[Field, int]` exposes both | ❌ model is blind to terrain/Trick Room live |
| `p1/p2_hazard_*`, `p1/p2_side_*` | layer counts / 0-1 (:877-889) | enum-vs-string dead check (:897-912) → all zeros | ❌ finding #3 |
| `last_move_p1/p2` | previous-turn, normalized lowercase, 'none' default (DEEP-03-verified semantics) | raw 'Knock Off' format; `'None'` (capital) initial (:572-573); p1/p2 assignment **hardcoded-swapped** on the protocol role (:646-654) — a per-battle coin flip since the bot isn't always protocol-p2 | ❌ triple fault: wrong vocab (never matches), wrong default token, and randomly wrong seat. These features are pure noise live |
| Smogon usage columns | `get_smogon_usages_df` + `sanitize_name` (feature_engineering.py:53-102) | `load_smogon_moves` — a **third sanitizer** (`'%'→'perc'`, no regex; :195, :208) + the dead-crash injection block (:750-757) + a working duplicate in `_prepare_data_for_model` (:1015-1034) keyed by Title-case p2 species → opponent usage lookups miss | ❌ delete the :750-757 block (DEEP-03 fix 1); unify the sanitizer (§2) |
| Missing columns at reindex | n/a | silent default-fill `'Unknown'`/0 (:958-960) | ⚠️ hides every schema drift; must become a counted, logged event |
| Extra live-only columns | n/a | `p1_active_slot` (:843) — never in training | benign (dropped by reindex) but symptomatic |

**Net effect at inference for the v4 move model:** opponent species unseen, both last-move
features garbage, hazards/screens/terrain/Trick Room all zero, opponent usage vector zero —
the model runs on a fraction of its trained signal even on the turns it doesn't crash. The
LightGBM's robustness to unseen categories is the only reason output was ever non-random.

---

## 2. Killing the duplicated-logic class (prompt item 2)

The audit found **three sanitizers** (`sanitize_name` regex; `normalize_species_name`/
`normalize_move_name` with forme-stripping; `load_smogon_moves`'s ad-hoc `'%'→'perc'` variant),
**two Smogon loaders**, and **two feature-prep stacks**. `verify_parity.py` (repo root) exists
precisely because this drift was suspected — but it only compares sanitizer outputs on move-name
sets, and Phase 2's "acceptance: verify_parity.py passes" was never made meaningful.

Consolidation design (target state, one burst of work):

1. **One normalization module.** Move `normalize_species_name`, `normalize_move_name`,
   `FORME_SUFFIXES`, and `sanitize_name` into `feature_engineering.py` (or a new tiny
   `normalization.py` imported by parser, trainer, and bot). `process_replays.py` imports it —
   the parser keeps zero private copies. Delete `load_smogon_moves`'s sanitizer;
   the bot imports `get_smogon_usages_df` and derives `pokemon_valid_moves` from it in five
   lines.
2. **A written canon, enforced at the boundary:** every string feature is lowercase
   sanitize-form (species forme-stripped, moves normalized, tera/weather/terrain lowercase,
   absent = `'none'`, unknown = `'unknown'`). The v4 artifacts predate the canon (they contain
   'Water' tera and raw-case categories), so canon lands **at the next retrain (v5)**; until
   then the live mapper must emit v4's exact legacy expectations — which §3's harness, not
   memory, confirms.
3. **One state mapper, seat-parameterized.** Extract `map_battle_to_dataframe_row` into
   `state_mapper.py: map_battle(battle, seat) -> dict`, fixing every §1 defect, using
   enum-keyed lookups (`SideCondition.STEALTH_ROCK`), `battle.fields` for terrain/Trick Room,
   and `battle.player_role` for last-move seat attribution. DEEP-07 §1.1 needs exactly this
   function for opponent-seat prediction — build it once. `predict_action.py` (and the future
   `search_player.py`) become thin consumers.
4. **Feature assembly shared too:** the multi-hot/usage/reindex logic in
   `_prepare_data_for_model` duplicates `build_medium_X` semantics by reverse-engineering
   column names. Replace with a `FeatureAssembler` in `feature_engineering.py` that takes
   (flat row, feature_info) and produces the model frame using the *same* forward
   sanitization as training — no `.title()` round-trips. Dead code (`prepare_input_data_medium`,
   `predict_moves_with_filter`, `get_switch_slot` — none called by `choose_move`) is deleted,
   not migrated.

After this, parity failures require editing one shared function *and* ignoring a failing
harness — drift stops being a latent state and becomes a loud event.

---

## 3. The mechanical parity harness (prompt item 3)

`parity_check.py`, run against a local Showdown server (the DEEP-04 harness's server does
double duty):

1. **Record:** play N=3 scripted games (bot vs `RandomPlayer`). Per decision turn, dump the
   JSONL decision record (§4) including the complete flat feature row.
2. **Rebuild:** take the server's saved replay `.log` for each game, run it through
   `process_replays.parse_showdown_replay` + the identical feature path, select the rows for
   the bot's seat and decision turns.
3. **Diff:** align on (battle, turn, forced-flag), compare column-wise. Report per-column
   mismatch counts and a per-game summary.
4. **Pass/fail:** exact match required for all categorical/int columns; `hp_perc` tolerance
   ±1 (rounding paths differ); a short *explicit* allowlist for known-and-accepted semantic
   gaps (currently: own-side revealed-moves fullness until the §1 decision is made; timer-
   related `turn 0` rows are excluded). Anything not allowlisted fails. Exit code drives it —
   this becomes Phase 2's real acceptance criterion, replacing the current
   `verify_parity.py` (absorb its move-set check as one sub-test).
5. **Cadence:** run at every burst start, after any change to `state_mapper.py`,
   `feature_engineering.py`, `process_replays.py`, or a model artifact swap. It's ~3 minutes
   of wall clock.

The deep value: the harness also catches *future* poke-env API drift (the exact class of
finding #2/#3) because enum-keyed lookups that silently change shape will diff against the
log-derived truth immediately.

---

## 4. Silent-failure inventory & the decision record (prompt item 4)

Every swallow point in the live path today, and what it hid:

| Site | Catches | Hid in practice |
|---|---|---|
| `choose_move` mapping try (:1165-1169) → random | **any** mapper exception | findings #1 and #2 — the bot's entire brain, every turn |
| Stage-1 try (:1174-1181) → "defaulting to move" | binary-model prep/predict errors | feature drift in the simplified set |
| Stage-2 switch try (:1190-1196) → fall through to move | switch-target errors | DEEP-03 F3's garbage model would never be noticed |
| Stage-2 move try (:1199-1205) → random fallback | move-model errors | unseen-category storms |
| `_handle_battle_message` warn-and-continue (:675-678) | last-move tracker errors | the seat-swap bug produced no signal |
| `load_smogon_moves` broad excepts (:224-232) → empty structures | JSON issues | filtering silently off |
| reindex default-fill (:958-960) | schema drift | every missing column, forever |

**Design rule replacing all of this: fallbacks may keep the bot alive mid-game, but every
fallback is a counted, typed, logged event — and the harness fails on nonzero unexplained
counts.**

Per-turn decision record (JSONL, `logs/decisions/<battle_tag>.jsonl`), one line per
`choose_move`: `ts, battle_tag, turn, seat, feature_row (full dict), feature_hash,
model_ids {move, switch_bin, switch_target}, switch_prob, top5_moves+probs, valid_top5_flags,
chosen_action, order_sent, fallback_reason ∈ {none, mapping_error, prep_error, model_error,
no_valid_prediction, timeout}, exception_str, latency_ms`. End-of-battle summary line:
fallback counts by type + result. `evaluate_bot.py` (DEEP-04) asserts
`fallback_rate < 1%` before accepting any winrate as meaningful — retroactively, this single
assertion would have caught everything in §0. The same records feed §3's harness (they *are*
its live half) and DEEP-07's disagreement diagnostics.

---

## 5. Version-skew invariants (prompt item 5)

Today the bot loads v4 move (medium) + v2 binary switch (simplified) + v1 switch-target
(simplified_moves): three feature sets, three unknown datasets, zero load-time checks —
`feature_info` carries only the ordered feature list.

Every future artifact set ships `<stem>_manifest.json`:

```
{ artifact: "action_lgbm_v5_medium_move_only", task, feature_set,
  feature_list_sha256,            # of the ordered feature_names_in_order
  dataset: "C1-v1",               # DEEP-06 corpus name — never a loose filename
  usage_file: "gen9ou-1695-2026-06.json",
  normalization_canon: "v1",      # §2's spec version
  feature_engineering_git: "<hash>", trained_at, metrics: {clean_ce, ...} }
```

Load-time checks in the bot (hard-fail, not fallback — a misconfigured bot must refuse to
queue): (a) recompute feature-list hash from the loaded feature_info == manifest; (b) the
usage file on disk == every manifest's `usage_file` (one pinned prior for all three models —
today they'd silently disagree the moment one is retrained); (c) all manifests share a
`normalization_canon`; (d) log all manifest ids into every decision record (§4), so any logged
game is attributable to exact artifacts. Legacy v4/v2/v1 artifacts get hand-written manifests
once, marked `canon: "legacy-v4"`, which the mapper uses to select legacy emit mode (§2.2).

---

## 6. Ranked findings & fixes (prompt item 6)

| Pri | Finding | Fix | Effort |
|---|---|---|---|
| 0 | Credentials in git (:57-58) | rotate password now; env-vars only; consider history scrub | 15 min + rotate |
| 1 | Three every-game breakers: `player_prefix` NameError (:756), weather-dict crash (:881), enum-keyed hazards dead check (:897-912) | delete :750-757; `battle.weather` dict handling; enum-keyed maps | ~1 h total |
| 2 | Blind features live: terrain/pseudo hardcoded (:885-889), p2 Title case (:785), last-move triple fault (:572-573, :646-654) | `battle.fields`; `.lower()`; role-aware normalized tracker | ~half day |
| 3 | Decision record + fallback counters (§4) | new logging module; wraps existing paths | ~half day |
| 4 | Parity harness (§3) | `parity_check.py`; absorb verify_parity.py | ~1 day |
| 5 | Consolidation (§2): one normalizer, one mapper, one assembler; delete dead code | `state_mapper.py` + feature_engineering exports | ~2 days |
| 6 | Manifests + load-time invariants (§5) | small writer at train time, checker at load | ~half day |
| 7 | Own-side revealed-moves semantics decision (§1) | pick canon; either mask live or retrain with full self-movesets | with next retrain |

Sequencing note: items 1–3 must land **before** DEEP-04's W0 baseline measurement (otherwise
W0 measures noise); item 4 gates any "ship" decision thereafter; items 5–6 ride the next
natural refactor/retrain. Log per CLAUDE.md: item-1 fixes and the §2 canon are PROJECT.md Key
Decisions; `state_mapper.py`/`parity_check.py` are PROJECT_CONTEXT.md pipeline entries; the
first clean parity run and the first fallback-rate number belong in experiments.md alongside
W0.
