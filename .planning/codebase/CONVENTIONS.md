# Code Conventions

## 1. Species Name Normalization

All Pokémon species names throughout the codebase follow this convention:

```python
# Standard: lowercase, no hyphens, no spaces
name.lower().replace('-', '').replace(' ', '')

# Examples:
"Great Tusk"  → "greattusk"
"Iron Valiant" → "ironvaliant"
"Gholdengo"   → "gholdengo"
"Urshifu-Rapid-Strike" → "urshifurapidstrike"
```

This normalization is applied in:
- `process_replays.py` → `normalize_species_name()` function
- `predict_action.py` → `pkmn.species.title().lower().replace(" ","")` at inference
- `load_smogon_moves()` → `pokemon_name.lower().replace(' ', '').replace('-','')`

> ⚠️ **Discrepancy**: In `predict_action.py` line 727, `pkmn.species.title().lower()` is applied — the `.title()` is redundant and potentially a bug carry-over. The standard should be plain `.lower()`.

### Forme Suffix Stripping
`process_replays.py` defines an extensive `FORME_SUFFIXES` list for matching battle-only formes back to their base species (e.g., `mimikyubusted` → `mimikyu`). Lookups use a `get_base_species()` function that tries progressively stripped candidates.

---

## 2. Move Name Sanitization

There are **two sanitization styles** in the codebase that must be kept in sync:

### Canonical (v4+): `sanitize_name()` in `train_action_predictor.py`
```python
def sanitize_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())
```
Strips all non-alphanumeric characters. Used for Smogon key matching in training.

### Legacy: inline in `predict_action.py` / `train_switch_predictor.py`
```python
move.lower().replace(' ', '').replace('-', '').replace('_', '').replace(':', '').replace('%', 'perc')
```
Chained replaces — functionally similar but not identical (e.g., `%` becomes `perc` vs stripped).

> ⚠️ **Known Issue** (documented in `fix-overfitting.md`): Parity drift between these two approaches was causing <90% Smogon move match rates. `verify_parity.py` exists to measure this gap. The regex version should be the single canonical implementation.

---

## 3. Model Artifact Versioning

Model files follow this naming convention:
```
{scope}_{framework}_model_{version}_{feature_set}_{mode}.{ext}
```

| Segment | Values | Example |
|---------|--------|---------|
| `scope` | `action`, `switch_predictor`, `switch_target_predictor` | `action` |
| `framework` | `lgbm`, `tf` | `lgbm` |
| `version` | `v4` (action), `v2` (switch) | `v4` |
| `feature_set` | `medium`, `simplified`, `full` | `medium` |
| `mode` | `move_only`, `all_actions` (action only) | `move_only` |

Companion files always use the same stem:
- `*_feature_info_*.joblib` — dict with `feature_names_in_order`, `numerical_features`, `categorical_features`, `category_map`
- `*_scaler_*.joblib` — fitted `StandardScaler`
- `*_label_encoder_*.joblib` — fitted `LabelEncoder` (where applicable)

---

## 4. Feature Set Naming

Three named feature sets are used across training scripts:

| Set | Description |
|-----|-------------|
| `simplified` | Active Pokémon species + HP + status + revealed moves + hazards/screens + field |
| `medium` | Simplified + bench team slots + Smogon usage stats injection |
| `full` | All available columns, including all 6 slots per player with all sub-fields |

The `simplified` set is used for the switch predictors (binary + target). The `medium` set is the primary production feature set for the action predictor.

---

## 5. Player Perspective Convention

All `medium` and `simplified` feature sets are **filtered to `player_to_move == 'p1'`** before training. This means:

- The model always "sees" the world from P1's perspective
- `p1_*` features describe the bot's own team
- `p2_*` features describe the opponent's team
- The bot at inference time always sets `player_to_move = 'p1'`

---

## 6. HP Representation

| Stage | Format |
|-------|--------|
| Raw parse | `int` (0–100, via `parse_hp()`) |
| Training (v4+) | Categorical bins via `bin_hp()`: `Fainted`, `Sash`, `Critical`, `Low`, `Middle`, `High`, `Full` |
| Inference | HP fraction from poke-env → `round(pkmn.current_hp_fraction * 100)` |

---

## 7. Status Conditions

Status values are stored as lowercase strings:
- `none` — no status
- `brn`, `par`, `slp`, `frz`, `psn`, `tox` — standard conditions
- `fnt` — fainted (used internally during parsing)

---

## 8. Feature Category Maps

All categorical features passed to LightGBM must have a `category_map` dict stored in the `*_feature_info_*.joblib` file. This maps each column name to a `pd.CategoricalDtype` with all known categories, ensuring inference can handle unseen values by mapping them to `'Unknown'`.

---

## 9. Revealed Moves Encoding

Revealed moves are stored during parsing as comma-separated strings:
```
"Stealth Rock,Rapid Spin,Headlong Rush"
```

During training, these are multi-hot encoded:
- Column name: `{player}_active_revealed_move_{sanitized_move_name}`
- Value: `1` if move in revealed set, `0` otherwise
- Only active Pokémon's moves are encoded for the `medium`/`simplified` sets

During inference (`prepare_input_data_medium()`), this same encoding logic is re-applied from `p1_active_revealed_moves_str` and `p2_active_revealed_moves_str` string columns.

---

## 10. Import Style

All scripts use direct function/class imports rather than module-level references:
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from poke_env.player import Player, DefaultBattleOrder
import lightgbm as lgb
```

No relative imports are used (all scripts are top-level, not organized into packages).
