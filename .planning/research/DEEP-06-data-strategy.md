# DEEP-06 — Data Strategy & Meta Drift: Which Replays Deserve Training On?

> Written 2026-07-05. Builds on DEEP-02 §4.1/§5 (rating-weighted BC, parser additions), DEEP-03
> (F6 winner sidecar, the consolidated re-parse batch, dedup caveat), DEEP-04 (all experiments
> here are measured on its dashboard — skill margin and clean CE especially). Everything in §1
> was **measured on the actual repo contents this session**, not assumed.

---

## 0. TL;DR

1. **We already own the metadata; we just throw it away twice.** Raw `.log` files carry `|t:|`
   unix timestamps and per-player ratings in `|player|` lines; the metamon *raw* shards
   (`metamon_subset/`, cols: `id, format, players, log, uploadtime, formatid, rating`) carry
   upload time and rating as columns. Both `process_replays.py` and `process_metamon_data.py`
   discard all of it. One sidecar index fixes auditability without touching feature schemas.
2. **Measured composition of our own scrapes:** `replay_logs/` ≈ 30.3k logs (IDs 2172M–2335M ≈
   Aug 2024 → early 2025), `replay_logs_3/` ≈ 20.0k logs (IDs 2523M+ → Feb 2026). Sampled
   ratings: median ≈ 1390–1405, range ~1070–1920, **no rating filter was applied at download**
   (the bulk downloader's ID-iteration mode takes everything). We train on median-1400 play —
   the clone learns mid-ladder habits, exactly the ceiling DEEP-02 §4.1 predicted.
3. **Finding — the usage prior is low-ladder:** `data/gen9ou-0.json` is the **0-cutoff**
   (unweighted, all-ratings) Smogon file — 1.25M battles of everyone. Both the
   `*_active_usage_*` training features and the live bot's legality/prior filter are built from
   how *bad* players play. Switch to the **1695-cutoff** file (or 1760) and pin its month.
   This is a one-file swap with plausible free accuracy.
4. **Era matters and should be derived empirically, not from memory:** cut Gen 9 OU into eras
   at spikes in month-over-month Jensen–Shannon divergence of Smogon usage vectors (§2). Known
   hard watersheds to sanity-check against: Home (2023-05), Teal Mask (2023-09), Indigo Disk
   (2023-12), the 2024 tera/ban votes. Anything pre-Indigo-Disk is a different game and should
   be excluded or heavily down-weighted by default.
5. **Add a frozen temporal test set** (most recent ~2 months, high-rated) alongside the random
   GroupShuffleSplit test — the temporal one measures what deployment actually faces (§3).
6. **Canonical corpus "C1" defined in §6** — sources, era window, rating handling, dedup —
   so every experiments.md row can say `dataset: C1-v1` instead of "❓ unrecorded."

---

## 1. Corpus audit (prompt item 1)

### 1.1 What exists today (measured)

| Source | Contents | Era (est.) | Rating info |
|---|---|---|---|
| `replay_logs/` | ~30,271 `.log` (gen9ou) | IDs 2172M→2335M ≈ 2024-08 → early 2025 | in-file, unparsed; sampled median ≈1388 |
| `replay_logs_3/` | ~19,965 `.log` | IDs 2523M+ → 2026-02 | in-file, unparsed; sampled median ≈1405, max 1920 |
| `metamon_subset/` + `metamon_parquet_files/` | raw HF shards, `train-0000x-of-00041` | `uploadtime` column — auditable trivially | `rating` column — auditable trivially |
| `metamon_clean/` | processed shards, 184-col schema matching `process_replays.py` output | **stripped** | **stripped** |
| `data/*.parquet` (5000/10k/30k/100k, only_p1, test) | training-era artifacts | unknown — lineage unrecorded | none |
| `data/gen9ou-0.json` | Smogon usage, cutoff 0.0, 1,253,206 battles | month not stored in file (mtime 2025-04) | n/a — includes all ratings |

Notes: a few sampled logs have degenerate `|t:|` values (epoch 0) — the audit script must
fall back to replay-ID ordering (IDs are globally sequential) when timestamps are missing.
The named parquets' unknown lineage is the same liability DEEP-03 flagged; C1 (§6) supersedes
them rather than reconstructing their history.

### 1.2 The audit artifact: `data/replay_index.parquet`

One row per replay, built by a standalone `build_replay_index.py` that reads **only log
headers** (first ~15 lines) plus the metamon raw-shard columns — no full parsing, so it runs
in minutes over 50k logs:

`replay_id (canonical numeric), source (scrape1|scrape3|metamon), upload_ts, upload_month,
p1_rating, p2_rating, rated (bool), tier, teamsize_ok, n_lines, winner` — winner needs the
tail of the file too (`|win|`), which the same pass grabs. This artifact *is* DEEP-03 F6's
winner sidecar and DEEP-02 §5.2's rating store, delivered without waiting for the big re-parse:
training scripts join it on `replay_id` for sample weights, filters, and split definitions.

**Deduplication:** canonicalize `replay_id` to its numeric part and inner-join across sources —
the metamon corpus and our scrapes cover overlapping ladder periods, and the same battle
appearing twice under two id formats would silently double-weight it and, worse, could cross a
split boundary (the one hole in DEEP-03 §2's split guarantee). The index makes this a two-line
check; report the overlap count in the audit output.

**Parser additions** (fold into the DEEP-03 §7 consolidated re-parse, already carrying ratings/
forced-flags/volatiles): emit `upload_ts` and per-seat rating onto every row so future datasets
are self-describing even without the index.

### 1.3 First deliverable

Run the index over everything and produce `.planning/research/corpus-audit.md`: histograms of
month × source, rating × source, replays/month, dedup overlap. Every later decision in this
doc gets its thresholds from that table. (Half a day, no ML.)

---

## 2. Meta drift: how stale is too stale? (prompt item 2)

### 2.1 Deriving era boundaries empirically

Don't hand-pick dates from memory (mine ends 2026-01 and ban votes since are invisible to me).
Smogon publishes monthly usage files; the drift signal is directly computable:

1. Download monthly `gen9ou-1695.json` for every month since 2023-01 (~40 small files, one URL
   pattern) into `data/usage_history/` (gitignored, but listed in the corpus audit).
2. For consecutive months, compute Jensen–Shannon divergence between (a) species usage vectors
   and (b) per-top-30-species move distributions.
3. Era boundaries = local spikes. Expect spikes at: Home (2023-05), Teal Mask (2023-09),
   **Indigo Disk (2023-12) — almost certainly the biggest**, the 2024 tera-preview/ban votes,
   and any 2025–26 events I can't know about. The plot answers instead of my recall.

This costs an afternoon and doubles as freshness tooling (§5): the same JS metric between "the
month the model trained on" and "current month" is the retrain trigger.

### 2.2 The drift-cost experiment

Once the index exists (all arms measured on the DEEP-04 dashboard, same model config = the
current champion, ~500k-row samples per arm to control for volume):

| Arm | Train | Eval |
|---|---|---|
| D1 | era E_latest−1 only | frozen temporal test (§3) |
| D2 | era E_latest only | same |
| D3 | all eras, uniform | same |
| D4 | all eras, recency-weighted (half-life ≈ one era) | same |

Predictions to falsify: D2 > D1 by a wide margin on **skill margin** (priors drift hardest —
species/move frequencies), narrower on menu-restricted top-1 (board-reading may transfer
across eras: type math doesn't drift, tier lists do). If D4 ≈ D2 with more data, use
recency-weighting and keep old data; if D2 dominates outright, hard-cut the window. This also
directly informs whether DEEP-01's attribute representation (which shouldn't care about tier
composition) drifts less than the species-token model — a bonus generalization argument if true.

### 2.3 Default policy until D-experiments run

Train on **Indigo Disk onward (≈2024-01+)**; treat older data as excluded. Rationale: both DLC
waves rewrote the tier's cast; a clone trained on pre-DLC replays learns lines into Pokémon
that no longer exist in the meta. The metamon corpus likely contains substantial 2023 material
(the audit will say exactly how much of its 9M rows survives the cut — the Run-5 scaling result
should be reinterpreted after seeing that number).

---

## 3. Time-based evaluation (prompt item 3)

Keep both, with distinct jobs:

- **Random GroupShuffleSplit test (existing):** measures in-distribution model quality;
  continuity with every recorded run; stays the per-run dashboard basis (DEEP-04 §4).
- **Frozen temporal test T1 (new):** all replays from the most recent ~2 complete months in the
  index, restricted to max(p1,p2) rating ≥ 1600, **never sampled into any training set**, frozen
  by name (`T1-2026-05_06`). Every milestone model reports the dashboard on T1 too. The
  random-vs-T1 *gap* is the measured drift penalty — the number deployment actually cares about.

Hygiene notes: temporal split is also grouped by replay (trivially — whole months), so no
GroupShuffleSplit conflict; T1 rows must be excluded *before* the random split runs (one filter
on the index join); when T1 is eventually rotated (new burst, newer months), the old T1 may
enter training and the new one gets a new name — never silently reuse the label. Report T1
results in their own experiments.md column to keep the two test populations from ever being
compared as if interchangeable.

---

## 4. Quality vs. quantity (prompt item 4)

### 4.1 The learning-curve grid

One-time experiment, current champion config, all cells on the same fixed random test set
(and T1), sampling **by replay** (never by row) from the deduped index:

- **Volume axis:** 3%, 10%, 30%, 100% of eligible replays (era-filtered per §2.3).
- **Quality axis:** (Q0) all ratings; (Q1) hard floor max-seat ≥ 1400; (Q2) hard floor ≥ 1600;
  (Q3) all ratings with soft weights w = σ((rating − 1400)/150) per DEEP-02 §4.1.

Read-outs: clean CE and skill margin vs. log(rows). Decision rules: if curves are still rising
at 100%, volume is binding → prioritize scraping/metamon inclusion; if Q2@30% ≥ Q0@100%,
quality dominates → tighten the floor; expected winner (hypothesis to falsify): **Q3 soft
weighting ≥ any hard filter at equal volume**, because hard floors discard the (still
informative) negatives of mid-ladder play while weighting keeps their state-transition
coverage but discounts their decisions.

### 4.2 The 9M-row question

Run 5 showed metamon-scale data converged better but didn't break memorization. Re-frame via
the audit: how many of those 9M rows survive (a) dedup, (b) the era cut, (c) any rating floor?
The honest hypothesis is that a large minority is 2023-era and its contribution is prior-noise
for a 2026 bot. The learning-curve grid at Q3 answers whether the survivors are worth the
16 GB-RAM handling cost (DEEP-09's input-pipeline work is the enabler if yes).

---

## 5. Freshness operations (prompt item 5)

- **Usage-stats pinning:** replace `gen9ou-0.json` with the high-cutoff file and encode
  provenance in the filename: `data/gen9ou-1695-YYYY-MM.json`. The loader takes the path from
  config (both `feature_engineering.py` and `predict_action.py` already parameterize it);
  every experiments.md row's dataset cell includes the usage-file name. **Never overwrite a
  pinned file in place** — models are only reproducible with the exact prior they trained on.
  The 0-cutoff → 1695-cutoff swap itself is an experiment row (expected: better bot filter,
  slightly better usage features; cheap to verify).
- **Scrape cadence (burst-shaped):** at the *start* of each development burst, run the
  time-based downloader for the trailing gap (`bulk_download_replays.py` already supports
  `before`-timestamp pagination), landing in `replay_logs_<date>/`; refresh the index; that's
  it. No standing infrastructure, no cron — matches how this project actually gets worked on.
- **Retrain triggers** (checked at burst start, in order of strength): (1) a DLC/ban/tera-vote
  event since the last training month — mandatory retrain; (2) JS divergence between the
  model's pinned usage month and the current month exceeding the typical intra-era month-to-
  month level from §2.1's plot — recommended; (3) T1 skill-margin decay when re-evaluated
  against a *newer* temporal slice — the direct measurement, when data exists. Log the check's
  outcome in STATE.md each burst per CLAUDE.md.

---

## 6. The canonical corpus: **C1** (prompt item 6)

> **C1-v1** := deduped union of {replay_logs, replay_logs_3, metamon(raw, re-processed with
> metadata)} where: tier = gen9ou, rated, upload month ≥ 2024-01 (Indigo Disk era-start,
> pending §2.1's empirical boundary), teamsize 6, parse-clean; minus T1 months.
> Training weights: DEEP-02 §4.1 soft rating weight (Q3), recency half-life per §2.2's D4
> outcome (uniform until measured). Usage prior: `gen9ou-1695-<pinned-month>.json`.
> Eval: fixed random GroupShuffleSplit (seed 42) + frozen T1.

Mechanics: a checked-in spec file `.planning/corpus/C1-v1.md` lists the filter predicate, the
index snapshot hash, replay counts per source, and the pinned usage file — the parquet itself
stays gitignored per repo policy, but the *definition* is versioned text, which is what
reproducibility actually needs. Changes create `C1-v2`, never mutate v1. Every experiments.md
dataset cell from now on holds a corpus name (or an explicit legacy filename), closing the
"❓ unrecorded" hole permanently.

**Sequencing** (fits one burst, mostly independent of the model work in flight):
1. `build_replay_index.py` + corpus-audit.md (½ day) — unblocks everything.
2. Usage-file swap to 1695-cutoff, pinned (1 h + one comparison run).
3. `usage_history/` download + JS-divergence era plot (½ day).
4. C1-v1 spec + materialization + T1 freeze (½ day).
5. Drift grid (§2.2) and learning-curve grid (§4.1) as background runs across the burst,
   each an experiments.md row.

Per CLAUDE.md: adopting C1 is a Key Decision (PROJECT.md), the corpus spec directory is new
(PROJECT_CONTEXT.md repo-layout table), and the first audit + era plot results belong in
STATE.md's burst summary.
