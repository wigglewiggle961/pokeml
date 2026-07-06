# DEEP-09 — Training Speed & Iteration Velocity: Make Experiments Cheap

> Written 2026-07-05. Everything below is anchored to **measurements taken this session on the
> actual machine** (profile script run against `data/100k.parquet` through the real
> `build_medium_X` and the real `build_embedding_model`): RTX 3060 Laptop 6 GB (idle — TF 2.19
> is CPU-only on Windows, per DEEP-05 §4.4), 16 GB RAM, oneDNN CPU TF. Boundary with DEEP-05:
> that doc owns what the model *is*; this one owns everything around it.

---

## 0. Measured baseline (prompt item 1 — the profile)

`data/100k.parquet` → 524,175 p1 move-rows after filtering; model = current architecture,
1.40 M params, 49 embed inputs / 220 numerical / 400+400 revealed dims / 277 classes:

| Stage | Measured | Notes |
|---|---|---|
| Parquet load | 4.4 s | fine |
| Filters (dropna/p1/move:) | 5.2 s | fine |
| **`build_medium_X`** | **234 s** | re-run identically on *every* invocation; output frame = 906 cols, **2.28 GB** |
| Model throughput, batch 256 | **2,058 samples/s** → ~4.2 min/epoch at 524 k rows | the configured default |
| Model throughput, batch 1024 | **16,376 samples/s** → ~32 s/epoch | **8× — the per-step Python/graph overhead dominates at bs=256** |

So a representative full run today ≈ 4 min build + (30–60 epochs × 4.2 min) ≈ **2.5–4.5 hours**,
of which the *irreducible* part after this doc's fixes is ≈ **15–35 minutes** — and a screening
run on the fast tier (§5) is ≈ **3 minutes**. That is the difference between ~5 hypotheses per
weekend burst and ~50.

**Timing log format:** every training script wraps stages in a `timings` dict and writes it
into the run's metadata JSON (`models/eval_<run>.json` per DEEP-04 §4.1):
`{parquet_load_s, feature_build_s (or "cache_hit"), prepare_inputs_s, epochs_run, s_per_epoch,
total_s, rows, batch}` — and the experiments.md row's `Duration` column (already added by
DEEP-04 §4.4) holds `total_s` humanized. No run without a duration, ever again.

---

## 1. Feature caching (prompt item 2 — the biggest structural win)

**Spec — `feature_engineering.get_or_build_features(df_source, feature_set, params) → (X, num_f, cat_f, meta)`:**

- **Key** = sha256 over: (a) dataset identity — the DEEP-06 corpus name when present
  (`C1-v1`), else sorted (path, size, mtime) of the input parquet(s); (b) `feature_set`;
  (c) the exact build-params dict (`min_feature_replay_count`, `smogon_top_n`,
  `disable_smogon_features`, blind flags…); (d) the **content hash of
  `feature_engineering.py`** (so any code edit auto-invalidates); (e) the pinned usage-file
  name (DEEP-06 §5 — `gen9ou-1695-YYYY-MM.json`).
- **Value**, under `data/cache/<key12>/`: `X.parquet` (2.3 GB frame round-trips in ~10–20 s —
  a 10–20× win over the 234 s rebuild), `features.json` (numerical/categorical lists, build
  params echoed, source row-count), and nothing model-specific.
- **What is deliberately *not* cached:** vocab encoders, scalers, split indices — they are
  train-split-dependent and cheap (seconds), and caching them would couple the cache to
  split/seed choices. Cache boundary = "model-agnostic feature frame."
- **Invalidation** is automatic (key changes); **eviction** is manual with a helper
  (`--cache-prune` keeps newest N=4; entries are ~1–3 GB each, disk is the cheap resource).
- **DEEP-01 plug-in:** the species-attribute table gets its *own* tiny cache entry keyed by
  (pokedex file hash, usage file, attribute-builder code hash); `attach_species_attributes`
  and `build_matchup_features` become build-params in the main key like everything else. The
  design requires no special-casing — new feature stages = new params = new key.
- LightGBM arms (DEEP-05 §3) read the same cached frame — the cache is trainer-agnostic,
  which is exactly what makes the ablation grid affordable.

---

## 2. Input pipeline (prompt item 3 — measured, and mostly "don't")

- **Current reality:** X frame 2.28 GB + `prepare_inputs`' float32 dict copies ≈ another
  ~1.5 GB + TF's own copy at fit time. At 524 k rows this fits 16 GB with room; extrapolating
  linearly, the ceiling for the current in-RAM approach is roughly **1.5–2 M rows**. The 9 M-row
  metamon corpus is ~39 GB of frame — flatly impossible in-RAM here (Run 5's "9M/16GB" battle
  scar is explained).
- **Recommendation: don't build the streaming pipeline yet.** DEEP-06's era/rating filters
  will shrink the eligible corpus substantially before scale-out is justified; engineer
  `tf.data`-from-shards only if the C1 corpus audit lands above ~2 M rows *and* the §4
  learning curves say more data still pays. If/when needed: shard the cached X by replay-group
  into ~500 k-row parquet shards, `tf.data` interleave + prefetch, per-shard scaler transform —
  a 1–2-day job that should not be paid speculatively.
- **Mixed precision / XLA on this hardware: no.** CPU TF gets nothing from fp16 (oneDNN fp32
  paths; no tensor cores in play), and XLA on Windows-CPU is marginal-to-flaky. These become
  relevant only after the GPU environment lands (DEEP-05 §4.4 / item 8 below) — on the 3060,
  mixed precision is then worth ~1.5–2× and is a two-line change in either framework.
- **What *is* worth it now:** dtype hygiene at the cache boundary (int8 for multi-hot, float32
  everywhere, categoricals as pandas `category`) — shrinks the frame and the copies for free.

---

## 3. Convergence economics (prompt item 4)

- **Batch 256 → 1024 is the headline: measured 8× step throughput.** Pair with learning rate
  ~2–3e-3 (linear-ish scaling from 1e-3) and keep `ReduceLROnPlateau`. Risk: batch-size changes
  generalization; per the re-validation rule (§6), one A/B against R7′ (DEEP-05's reference
  run) settles it. Note BN interacts with batch size — at 1024 the BN statistics are *better*,
  not worse; the risk is mild sharp-minima lore, and the dashboard will say.
  (Try 2048 too — the measured curve suggests gains may continue; VRAM isn't the constraint,
  RAM copies are.)
- **`epochs=100000` + `EarlyStopping(patience=10)`:** sane shape, mis-tuned tail. At 32 s
  epochs, patience 10 costs five wasted minutes per run — acceptable; drop to 6–8 with
  `ReduceLROnPlateau(patience=3)` for the fast tier where every minute is screening budget.
  Cosine/one-cycle: a *tuning-time* nicety, not a priority — plateau-based control is fine
  while epochs are cheap.
- **Optuna:** three compounding fixes — (a) trials on the **fast tier** (§5), full tier only
  for the top-3 finalists (search cost ÷ ~10); (b) the `MedianPruner` is already configured
  but starves at full-run cost — on 3-minute trials it actually gets to prune; (c) DEEP-04
  §2's `set_user_attr` logging + clean-CE objective land in the same edit. One-time validation
  that subset rankings transfer: run ~8 configs on both tiers once, check Spearman ρ of val
  clean-CE (expect > 0.8; if not, fast tier shrinks to 25% instead of 10% and re-check).

---

## 4. (merged into §3 — kept numbering consistent with the prompt's items)

---

## 5. The two-tier protocol (prompt item 5)

- **FAST tier (screening):** a **materialized, named, replay-subsampled 10% of the canonical
  corpus** (`C1-v1-fast10`: seeded GroupShuffle by replay_id, built once, cached once).
  Batch 1024, patience 6. Measured cost ≈ 3–5 min/run end-to-end on cache hits. Purpose:
  feature ablations (DEEP-01 stages), architecture arms (DEEP-05 grid), Optuna trials,
  smoke tests. *Same* fixed val/test replay-ids within the tier so fast-tier numbers are
  comparable to each other.
- **FULL tier (evidence):** the full named corpus, standard settings, DEEP-04 dashboard +
  baselines. Purpose: reference runs (R7′), grid winners, anything that will be cited in a
  decision or shipped to the bot.
- **The promotion rule:** a fast-tier improvement is *interesting* at Δclean-CE > 0.01
  (≈ the tier's observed seed noise — measure once by re-running one config with 3 seeds),
  and becomes *believed* only after full-tier confirmation. Nothing enters PROJECT.md /
  ROADMAP.md decisions, and no artifact ships, on fast-tier evidence alone.
- **Bookkeeping without confusion:** experiments.md rows carry a `Tier` value in the Strategy
  cell (`[F]`/`[FULL]`), fast-tier screening sweeps get **one summary row** (link to a
  sweep table in the run's JSON) rather than 15 rows of noise, and cross-tier numbers are
  never compared in the same sentence. The DEEP-04 dashboard applies to both tiers
  identically — only the corpus name differs.

---

## 6. Ranked speed changes (prompt item 6)

| # | Change | Speedup (measured/est.) | Cost | Result-changing risk → mitigation |
|---|---|---|---|---|
| 1 | **Batch 256→1024 (+LR scale)** | **8× per epoch (measured)** | one flag | real but small → one A/B vs R7′; becomes part of R7′ config if clean |
| 2 | **Feature cache (§1)** | 234 s + 2.3 GB churn → ~15 s per run; unblocks all grids | ~half day | zero (bit-identical frame) |
| 3 | **Fast tier `C1-v1-fast10` (§5)** | ~10× per screening run | ~1 h once cache + corpus exist | rankings may not transfer → one-time Spearman check |
| 4 | **Optuna on fast tier + pruning + user_attrs** | search ~10× cheaper; trials actually prunable | ~half day (merges with DEEP-04 §2 edits) | objective change is *intended* (clean CE) |
| 5 | Timing log + Duration column | n/a (observability) | ~1 h | none |
| 6 | Dtype hygiene at cache boundary | ~1.3–1.5× RAM headroom | ~1 h | none if dtypes chosen loudly |
| 7 | **GPU environment** (WSL2 for TF, or PyTorch-for-scorer per DEEP-05 §4.4) | est. 5–15× on step time; prerequisite for sequence models / big scorer | ~a day | framework/device numerics differ → re-validate vs R7′ |
| 8 | `tf.data` sharded pipeline | only matters > ~2 M rows | 1–2 days | deferred until DEEP-06's corpus audit demands it |
| — | ~~Mixed precision / XLA on CPU~~ | none here (verified CPU-only TF) | — | explicitly not doing |

**Compound effect:** items 1–3 alone take the standard experiment from ~3 hours to ~15–30 min
(full tier) and ~3 min (fast tier), for roughly one day of implementation — they should land
**before** the DEEP-01/05 experiment grids start, since those grids are exactly what the
speedups multiply. Re-validation rule, stated once: any change on this list that can alter
convergence (1, 4's objective, 7) gets one confirmation A/B against the frozen R7′ reference
before its numbers are trusted; changes that cannot (2, 5, 6) are asserted bit-identical /
metric-identical on one run and then trusted.

Per CLAUDE.md: the profile numbers in §0 and the two-tier protocol are the durable decisions —
log the protocol adoption in PROJECT.md Key Decisions, add `data/cache/` to
PROJECT_CONTEXT.md's layout table (gitignored), and give R7′-rerun-at-batch-1024 its own
experiments.md row when item 1 is validated.
