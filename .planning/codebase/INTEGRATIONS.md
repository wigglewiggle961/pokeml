# Integrations

## 1. Pokémon Showdown — poke-env

### Library
- **Package**: `poke-env`
- **Key imports**: `Player`, `AccountConfiguration`, `ShowdownServerConfiguration`, `Battle`, `Pokemon`, `DefaultBattleOrder`, `BattleOrder`
- **Purpose**: Async battle client connecting to Pokémon Showdown's WebSocket server

### Integration Points in `predict_action.py`
| Method / Class | Description |
|----------------|-------------|
| `PredictionPlayer(Player)` | Extends poke-env's `Player` base class |
| `choose_move(battle)` | Main override — maps battle state → model prediction → `BattleOrder` |
| `_handle_battle_message(split_messages)` | Override to track last-used moves per player turn |
| `_update_last_moves(split_messages)` | Parses `move` events from battle message bundles |
| `map_battle_to_dataframe_row(battle)` | Converts live `Battle` object to flat dict for feature alignment |

### Battle Object Fields Used
| Field | Purpose |
|-------|---------|
| `battle.team` / `battle.opponent_team` | Dict of `Pokemon` objects by slot |
| `battle.active_pokemon` / `battle.opponent_active_pokemon` | Currently active Pokémon |
| `battle.turn` | Current turn number |
| `pkmn.species` | Pokémon species ID |
| `pkmn.current_hp_fraction` | HP as 0.0–1.0 fraction |
| `pkmn.status` | Status condition enum |
| `pkmn.fainted` | Boolean faint state |
| `pkmn.boosts` | Dict of stat stage modifiers |
| `pkmn.is_terastallized` / `pkmn.tera_type` | Tera state |
| `pkmn.moves` | Dict of revealed `Move` objects |

### Connection Configuration
- **Username**: `SHOWDOWN_USER` env var (default: `www31`)
- **Password**: `SHOWDOWN_PASS` env var
- **Format**: `gen9ou`
- **Server**: Default main Showdown server via `ShowdownServerConfiguration`
- **Team**: Hard-coded Smogon-legal 6-Pokémon team (Great Tusk, Kingambit, Gholdengo, Gliscor, Heatran, Dragapult)

---

## 2. Smogon Usage Statistics — `gen9ou-0.json`

### Source
- **File**: `gen9ou-0.json` (locally stored, ~14MB)
- **Origin**: Smogon competitive statistics export for Gen 9 OU
- **Structure**: `{ "info": { "metagame": "gen9ou", ... }, "data": { "PokemonName": { "Moves": { "MoveName": count }, "Raw count": N, "usage": 0.xxx } } }`

### What is Extracted
| Data | Usage |
|------|-------|
| Top-100 usage Pokémon (by `usage` field) | Filters switch target classifier to top-100 classes |
| Valid move sets per species | Filters predicted moves to Smogon-viable options at inference time |
| Move usage percentages per species | Injected as numerical features: `p1_active_usage_{movename}` |

### Sanitization Convention
Both training (`train_action_predictor.py`) and inference (`predict_action.py`) must apply **identical** sanitization:
```python
# Current v4 (training)
re.sub(r'[^a-z0-9]', '', name.lower())  # sanitize_name()

# Current v4 (inference — load_smogon_moves)
move_key.lower().replace(' ', '').replace('-', '').replace('_', '').replace(':', '').replace('%', 'perc')
```
> ⚠️ **Known Issue**: Minor parity drift between training and inference sanitization was flagged in `fix-overfitting.md`. The `sanitize_name()` function (regex-based) should be the canonical implementation.

### Species Normalization for Smogon Lookups
```python
name.lower().replace(' ', '').replace('-', '')  # Smogon key normalization
```

---

## 3. Replay Data Sources

### Raw Replay Logs
| Directory | Description |
|-----------|-------------|
| `replay_logs/` | First batch of raw `.log` replay files |
| `replay_logs_2/` | Second batch |
| `replay_logs_3/` | Third batch |
| `replay_20k/` | 20k replay batch |

### Download Scripts
- `download_replays.py` — Single-threaded downloader for Showdown replays
- `bulk_download_replays.py` — Parallel bulk downloader

### External Dataset: Metamon
- **Source**: External Pokémon battle dataset
- **Storage**: `metamon_parquet_files/`, `metamon_processed/`
- **Extraction**: `extract_metamon_logs.py` → `process_metamon_data.py`
- **Status**: Referenced in README as a potential future data source

---

## 4. Processed Training Data
| File | Rows (approx) | Description |
|------|--------------|-------------|
| `30k.parquet` | ~30k replays | Primary training dataset |
| `30k_augmented.parquet` | ~112k+ | Perspective-augmented (via `augment_perspectives.py`) |
| `10k.parquet` | ~10k | Smaller dataset for quick experiments |
| `10k.csv` | ~30k rows | Legacy CSV format (~385MB) |

---

## 5. EDA & Notebooks (Non-Production)
| File | Purpose |
|------|---------|
| `catboost.ipynb` | CatBoost model exploration |
| `tensorflow.ipynb` | TF training experiments |
| `eda.py`, `eda2.py`, `eda3.py` | Exploratory data analysis scripts |
| `demonstration.py` | Battle demonstration / manual testing |
