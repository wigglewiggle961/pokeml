# CLAUDE.md — Working guide for PokéML

Read this first, then [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the full architecture and
[.planning/](.planning/) for progress. This project is developed in bursts with long gaps, so the
written trail below is the source of truth — **keep it fresh** (see "Decision logging" rule).

## What this is
Gen 9 OU Pokémon Showdown battle AI. Pipeline:
`replay .log → process_replays.py → parquet → train_*.py → predict_action.py (live bot via poke-env)`.
Core challenge: the move model tends to memorize opponent *species identity* instead of game state.
Best model so far = the embedding model (`train_action_predictor_embedding.py`, Run 7).

## Repository layout — where things go
Run all scripts **from the repo root**; paths are root-relative.

| Put this here | Folder | Tracked in git? |
|---|---|---|
| Datasets (`*.parquet`, `*.csv`), `gen9ou-0.json` | `data/` | No (gitignored) |
| Trained artifacts (`*.txt`, `*.keras`, `*.joblib`, metadata) | `models/` | No (gitignored) |
| Exploratory notebooks | `notebooks/` | Yes |
| Superseded / scratch scripts | `archive/` | No (gitignored) |
| Raw replay logs, metamon working dirs | `replay_logs*/`, `metamon_*/` | No (gitignored) |
| Live pipeline / data-collection / EDA scripts | repo root `*.py` | Yes |

**When saving new files:** weights, checkpoints, scalers, encoders, and metadata → `models/`.
Datasets → `data/`. Never save model/data files to the repo root. Never `git add` data or model
binaries — they are gitignored on purpose; their provenance is tracked in text (see below).

**Artifact naming convention:** `<task>_<framework>_<version>_<feature_set>[_<target>]`
(e.g. `action_lgbm_model_v4_medium_move_only.txt`, `action_tf_embedding_v1_medium.keras`).
A model's companion files share its stem: `_feature_info_`, `_scaler_`, `_label_encoder_`.

## Decision logging — keep the trail fresh (important)
Whenever a **key decision, experiment result, or architectural change** happens, update the relevant
doc(s) in the same session — do not rely on memory of the burst:

- **Training run / experiment** → add a row to the table in `.planning/experiments.md`
  (strategy, command, metrics, status) **and** a row to its "Artifacts & Reproducibility" table
  (artifact filename + **exact dataset used** — this is the only record of which data made which model).
- **Key design decision** → add/update the "Key Decisions" table in `.planning/PROJECT.md`
  (decision, rationale, outcome).
- **Phase/status change** → update `.planning/STATE.md` and `.planning/ROADMAP.md`.
- **Anything that makes PROJECT_CONTEXT.md stale** (new folder, changed pipeline, new model wired into
  the bot, changed config) → update `PROJECT_CONTEXT.md` so it always reflects reality.
- Durable, non-obvious facts about the user or project → save to Claude memory as well.

If a decision touches more than one of the above, update all that apply.

## Guardrails
- `train_action_predictor.py` is the **stable** LightGBM/plain-TF pipeline — treat as frozen; do new
  TF experimentation in `train_action_predictor_embedding.py`.
- All preprocessing/normalization is shared via `feature_engineering.py` — keep training and inference
  (`predict_action.py`) 100% consistent (species names lowercase, no spaces/hyphens).
- The live bot currently loads the **v4 LightGBM** action model, not the better embedding model —
  wiring the embedding model into the bot is deferred work.
