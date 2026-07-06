# DEEP-07 — Search & Consumption Design: How the Bot Uses the Models

> Written 2026-07-05. Designs DEEP-02 §4's steps 2–3 (the augmentation ladder past behavioral
> cloning). Verified against the installed poke-env this session: `poke_env.calc` **ships a
> gen-9 damage calculator** (`calculate_damage(attacker_id, defender_id, move, battle,
> is_critical)`) working directly on live `Battle` objects, and `Pokemon` exposes
> `stats/base_stats/boosts/item/ability/tera_type/damage_multiplier`. The hard prerequisite
> stack: DEEP-03 F1 fixed (bot currently random-falls-back), DEEP-04 harness built (nothing
> here is believable without the winrate gauntlet), DEEP-08's parity discipline for every new
> live-side feature.

---

## 0. TL;DR

1. **The 1-ply expectimax is much cheaper than DEEP-02 assumed** because the damage calculator
   already exists in poke-env. The build is: opponent-action distribution q̂ (models we have) ×
   transition *sketch* (not a simulator) × leaf evaluation, over ≤ ~120 action pairs — trivially
   inside the Showdown timer even on CPU.
2. **Decision rule: robust best-response, not pure best-response.** Score each of our actions by
   `(1−w)·EV over q̂ + w·worst-case over q̂'s support`, w ≈ 0.2. The blend is the honest answer
   to "we exploit a prediction, not an equilibrium": it caps the damage of a confidently-wrong
   q̂ and blunts knife-edge reads that mixing opponents would punish.
3. **Damage fidelity knee is at "modal sets":** the built-in calc + Smogon modal spread/item for
   the opponent (DEEP-01's attribute table already planned to carry these) resolves the two
   binary facts that flip decisions — who moves first, and whether X OHKOs Y. Full
   uncertainty-aware set sampling is diminishing returns; approximate it with 2–3 item
   hypotheses only where they flip a conclusion (Scarf/Sash/Boots).
4. **Leaf evaluation: if the leaf is a win probability, risk management is free.** Start with a
   ~6-term hand heuristic; replace with the replay-trained V(s) as soon as DEEP-06's
   `replay_index` (winner + ratings) exists — V(s) needs no re-parse, just that join.
   EV-maximizing a *probability* automatically plays safe ahead and seeks variance behind;
   explicit risk knobs are only needed while the leaf is a linear material score.
5. **Search un-blocks Terastallization for free.** Our tera-variants become extra candidate
   actions evaluated by the same EV machinery — the bot finally teras (DEEP-02's gap) without
   ever training a tera model.
6. **Expect the first search bot to lose to the clone.** W2 (crude leaf) failing W0 A/B is the
   *predicted* outcome, not a refutation; the validation ladder (§6) is designed so each
   fidelity increment is separately accepted/rejected by the DEEP-04 harness.

---

## 1. Decision rule (prompt item 1)

### 1.1 Formalization

Per turn, from live state s:

- **Our action set A(s):** available moves (+ tera-variant of each while tera unused — poke-env
  exposes legality) + available switches. |A| ≤ ~13.
- **Opponent action set B(s):** their active's revealed moves ∪ top Smogon moves for that
  species (capped at 4 total, usage-weighted) + their alive bench switches. |B| ≤ ~9. This is
  the same candidate-set construction DEEP-02 §5 wants at training time — build it once, share.
- **Opponent distribution q̂(b|s):** composed from the existing three models until the unified
  scorer ships: `P(switch)` from the binary model splits mass between move-branch (×move-model
  distribution restricted to B's moves, renormalized) and switch-branch (×switch-target
  distribution — after DEEP-03 F3 is fixed; until then, usage-weighted uniform over their
  bench). Floor it: q̂ ← (1−ε)·q̂ + ε·uniform(B), ε = 0.1, so no opponent option is ever
  priced at zero. **Perspective note:** predicting the opponent means mapping the battle from
  *their* seat; that's a second feature-mapping path and therefore a new DEEP-08 parity
  surface — build it as a parameterized `map_battle(battle, seat)` from day one, not a copy.
- **Score:** for each a ∈ A: `EV(a) = Σ_b q̂(b)·L(T(s,a,b))`, `WC(a) = min_b L(T(s,a,b))`,
  `score(a) = (1−w)·EV(a) + w·WC(a)` with w = 0.2 initially (tuned in W5).

### 1.2 When best-response-to-prediction breaks, and why the blend is enough

- **Miscalibrated q̂:** a peaked-but-wrong q̂ makes pure BR chase phantom reads (hard-switching
  into a predicted move that never comes). The ε-floor + worst-case term price in the
  possibility we're wrong; DEEP-04's ECE metric tells us how big ε deserves to be.
- **Mixing opponents:** in one ladder game, a human can't estimate our policy well enough to
  systematically exploit pure BR; across *repeated* games (same opponent, tournament) they can.
  The maximin blend already concedes little EV on knife-edges; §4's ε-mixing between near-equal
  actions removes the deterministic tell. We explicitly do **not** attempt equilibrium
  computation (per-turn matrix-game solving over sketched payoffs is feasible but pays only
  against opponents who model *us* — out of scope until there's evidence of being exploited,
  which champion-A/B vs. a BR-aware mirror could measure someday).
- **Compounding sketch error:** the worst failure mode isn't game theory, it's T(s,a,b) being
  wrong (missed ability interaction, wrong speed read). This is why fidelity gates (§6) are
  winrate-based, not theory-based.

---

## 2. Damage model fidelity (prompt item 2)

Tiers, updated for the discovered calculator:

| Tier | What | Cost | What it gets right/wrong |
|---|---|---|---|
| T0 | DEEP-01 crude proxy (eff × BP × stat-ratio) | exists soon anyway (feature side) | ordering of options, roughly; wrong on every threshold |
| T1 | `poke_env.calc.calculate_damage` as-is | ~0 — call it | exact formula, our side exact (we know our team); opponent side uses poke-env's knowledge = revealed item/ability + default stats when unknown |
| **T2** | T1 + opponent overlays: Smogon modal spread + modal item/ability injected for unrevealed slots (DEEP-01 attribute table provides both) | small — a wrapper that temporarily fills unknowns before calling calc | **the knee.** Fixes the two decision-flipping binaries: speed order (Scarf, investment) and OHKO/2HKO thresholds (Specs/Band/bulk) |
| T3 | Set-uncertainty aware: evaluate under top-k (spread,item) hypotheses, probability-weighted | k× calc calls | diminishing returns *except* Scarf/Sash/Boots ambiguity; do it lazily — only branch on hypotheses when T2's conclusion flips between them |

Asymmetry convention (mirrors human practice): for **our** KOs use minimum roll (never count on
max damage); for **their** damage onto us use 85th-percentile roll (respect their upside).
`calculate_damage` returns roll information; the wrapper standardizes this.

Speed resolution — the other half of "damage": compute effective speeds with boosts, paralysis,
Tailwind, Trick Room, and T2's modal spread; carry a `scarf_possible` flag (item-class prob >
~15% and not disproven by observed move order) that T3-style branches only when the speed read
decides the line. Observed move order across the game is *evidence* — a mon that moved second
while "faster on paper" gets its scarf flag cleared / spread hypothesis shifted; this little
Bayesian ratchet is cheap (a per-battle dict) and high-value.

---

## 3. Leaf evaluation (prompt item 3)

### 3.1 The transition sketch T(s,a,b)

Deliberately **not a simulator**. Resolve order by effective speed + priority; apply expected
damage (per §2 conventions); mark faints; apply the deterministic slice of secondaries
(hazards set/cleared, screens, explicit stat drops, recoil/drain); apply switch consequences
(hazard chip on entry, boost reset); status/para/burn applied at face probability as expected
value. Ignore: multi-turn moves' tails, rare abilities' edge cases, end-of-turn ordering
subtleties. Every ignored mechanic is a known error source the winrate gate will price.

### 3.2 Leaf v0 — hand heuristic (ships with W2)

`L(s') = σ( c1·(Σ our HP% − Σ their HP%)/600 + c2·(our alive − their alive)/6 +
c3·hazard_differential + c4·status_differential + c5·boosted_threat_on_field +
c6·speed_control )` — six terms, σ squashes to (0,1) so it composes with §4. Initial c from
eyeballing, tuned once via small grid against the harness (this is W2/W3's known weakness —
don't over-invest; it's scaffolding for V(s)).

### 3.3 Leaf v1 — replay-trained V(s)

Binary "did this seat win" head over the DEEP-05 state trunk (shared encoder, small MLP), label
= winner joined from **DEEP-06's `replay_index`** — which means V(s) is trainable *now*,
without the parser re-parse (the index carries winner + ratings; weight by rating like the
policy per DEEP-02 §4.1). Same GroupShuffleSplit hygiene. Validation: AUC on held-out replays
(expect ~0.75–0.85 mid-game), monotone-sanity checks (V rises as opponent mons faint), and
agreement with leaf v0's sign on lopsided positions. **Blend for safety at first deployment:**
`L = 0.7·V + 0.3·heuristic`, retire the heuristic once W4 beats W3 cleanly.

One subtlety: V is trained on *reached* human states, but search evaluates *sketched* states
(post-expected-damage pseudo-states). Mitigate by evaluating V on the sketch's discrete part
(HP bins, faints, hazards — exactly the DEEP-01 feature space, which bins HP anyway) rather
than raw continuous deltas; the representation's coarseness is protective here.

### 3.4 Risk (prompt item 4, first half)

If L is a calibrated win probability, **maximizing EV already encodes correct risk attitude**:
behind, high-variance lines have higher E[P(win)] (a 30%-to-win coinflip beats a sure slow
loss the leaf scores at 15%); ahead, they don't. So: no explicit risk knob once V(s) lands.
While on leaf v0 (material-linear), add the crude version: when L < 0.35, break near-ties
toward the action whose *best-case* over q̂ is highest.

### 3.5 Mixing (prompt item 4, second half)

Among actions within Δ = 0.02 win-prob of the top score, sample proportionally to
`exp(score/τ)` with small τ — near-free EV-wise by construction, removes the deterministic
tell, and directly implements DEEP-02 §4's anti-exploitability note. Log the sampled-vs-argmax
choice per turn (DEEP-08 decision record) so we can quantify how often mixing mattered.

---

## 5. Budget (prompt item 5)

Arithmetic at the ceiling: |A| ≤ 13, |B| ≤ 9 → ≤ 117 pairs; per pair ~1–4 calc calls + sketch +
leaf. Python cost per pair is sub-millisecond except the leaf-V call — batch all sketched
states into **one** model forward (117 rows) per turn. Total per turn: q̂ inference (one
forward) + one batched V forward + ~300 calc calls ≈ **well under 2 s on this machine's CPU**;
against the ladder timer (150 s bank + per-turn grace) that's a >10× margin. Engineering
guards: hard 8 s deadline → fall back to clone argmax and **log the fallback with reason**
(DEEP-03 F1's lesson: fallbacks must be loud); cache per-battle constants (opponent modal sets,
type charts) at switch-in, not per turn.

**2-ply:** cost explodes to ~117 successor states × (q̂ + sketch grid) ≈ 10⁴ sketches + ~117
batched model calls ≈ tens of seconds in Python — borderline. Verdict: not before V(s) is
trusted (2-ply amplifies leaf noise), and then only *selectively*: expand a second ply for the
top-2 first-ply actions when they're within Δ, or on forced positions (|A| ≤ 3). Full 2-ply is
a PyTorch-vectorized-scorer project (DEEP-05 §4.4's framework note applies), not a today
project.

---

## 6. Validation plan (prompt item 6)

Each rung: 400 games vs `SimpleHeuristicsPlayer` (north-star) + 400-game champion A/B vs the
previous accepted rung, per DEEP-04 §3; ship gate = A/B ≥ 50% − CI; every rung is an
experiments.md row; per-decision logs record `argmax-clone vs search choice` disagreement %
(the cheapest diagnostic of what search is actually changing).

| Rung | Config | Expectation / decision |
|---|---|---|
| W0 | clone argmax, post-F1-fix, current models | the honest baseline nobody has measured yet |
| W1 | W0 + §3.5 mixing only | ≈ W0 in winrate (mixing is about exploitability, not strength) — cheap sanity of the harness's noise floor |
| W2 | 1-ply, T1 calc, leaf v0 | **may lose to W0** — accept the rung only for infrastructure (logs, latency, fallback discipline), not winrate |
| W3 | + T2 modal sets, speed ratchet | first rung expected to beat W0; if it doesn't, the sketch (not the concept) is the suspect — audit disagreement logs before abandoning |
| W4 | + V(s) leaf (0.7/0.3 blend → pure) | expected biggest jump; also unlocks the §3.4 risk-freebie |
| W5 | tune w, ε, τ, Δ on the harness; enable tera-variant actions | the polish rung; tera alone may be worth measurable winrate |
| W6 | selective 2-ply on near-ties | only if W4/W5 plateau vs SimpleHeuristics; else skip to unified scorer |

Sequencing vs. the rest of the program: W0–W1 need only the F1 fix + DEEP-04 harness (this
week's work); W2–W3 need nothing from the training pipeline at all (pure bot-side, usable with
the *current* v4 LightGBM models — fixing DEEP-03 F3 first improves q̂'s switch branch); W4
needs DEEP-06's index (for V's labels) and ideally the DEEP-05 trunk. So the search line can
proceed **in parallel** with the representation/architecture line and each rung's gain is
measured independently of model improvements — when a better q̂ (attribute model, then unified
scorer) lands, re-run the current rung's A/B with the swap as its own row: that number, "search
gain × model gain factorized," is the cleanest evidence of whether the two programs compound.

Per CLAUDE.md: W-rungs are experiments.md rows; adopting the search bot as the deployed
default is a PROJECT.md Key Decision; `predict_action.py`'s replacement (a new
`search_player.py` — don't grow the 1200-line file) updates PROJECT_CONTEXT.md's pipeline
diagram and the bot-model note in experiments.md.
