# Phase 4 Research: TensorFlow Embedding Model — Refactored Architecture

**Gathered:** 2026-04-14
**Status:** Research Complete

---

## 1. Multi-Input Keras Architecture (Functional API)

For tabular data mixing categorical, numerical, and multi-hot features, the **Keras Functional API** is the standard approach. It allows separate "branches" per feature group that are concatenated before deep dense layers.

### Proposed Branch Structure

```
species_input ──► Embedding(vocab, 50) ──► Flatten ──┐
p2_species_input ► Embedding(vocab, 50) ──► Flatten ──┤
numerical_input ─────────────────────────────────────►│──► Concatenate ──► Dense Stack ──► Softmax(num_classes)
revealed_moves_in ► Dense(50, no_bias) ──────────────►│
smogon_usage_in ──────────────────────────────────────┘
```

- **Categorical branches:** Separate `Input` + `Embedding` layers for high-cardinality features (species, last move, tera type)
- **Numerical branch:** Single `Input` for all scaled numerical features (boosts, hazards, turn number)
- **Multi-hot branch:** Revealed moves — handled via a `Dense(embedding_dim, use_bias=False)` layer (sum-pooling equivalent)
- **Smogon usage branch:** Float32 vector — can be passed directly to the concatenation or through a small `Dense` projection

---

## 2. Embedding Dimension Sizing

| Rule | Formula | Result for 900 species | Result for 700 moves |
|------|---------|----------------------|---------------------|
| Rule of thumb | `min(50, vocab // 2)` | 50 | 50 |
| Power-law | `vocab ** 0.25` | ~5.5 → round to 32 | ~5.1 → round to 32 |
| FastAI convention | `min(600, round(1.6 * vocab**0.56))` | ~50 | ~46 |

**Recommendation:** Use **dim=32** for moves and **dim=48** for species. This balances expressiveness vs. parameter count. These are tunable hyperparameters.

---

## 3. Handling Multi-Hot "Revealed Moves"

Two viable approaches:

1. **Dense projection (recommended):** Apply a `Dense(embedding_dim, use_bias=False)` to the multi-hot binary vector. Mathematically equivalent to summing the embeddings of all revealed moves. Simple, fast, and compatible with the existing multi-hot encoding pipeline.

2. **Ragged Embedding lookup:** Convert the revealed moves list to a ragged integer sequence, lookup move embeddings, and apply sum/mean pooling. More accurate but requires changing the input representation from binary vectors to integer sequences.

**Go with approach 1 for Phase 4** — it reuses the existing multi-hot pipeline (no changes to `feature_engineering.py` output) and is a well-understood pattern.

---

## 4. Architecture & Hyperparameters for ~400 classes / millions of rows

```
Dense Stack after concatenation:
  512 → BatchNorm → Swish → Dropout(0.25)
  256 → BatchNorm → Swish → Dropout(0.25)
  128 → BatchNorm → Swish → Dropout(0.2)
  num_classes → Softmax
```

- **Activation:** `swish` often outperforms `relu` in deep tabular networks (smoother gradient)
- **BatchNormalization:** Critical after each Dense layer
- **Dropout:** 0.2–0.3 range; reduce as layers narrow
- **Label smoothing:** 0.05–0.1 for large class spaces (already parameterized in existing code)
- **Optimizer:** Adam with `lr=0.001`, reduce on plateau or cosine decay

---

## 5. Shared `feature_engineering.py` Module

Extract the following from `train_action_predictor.py` into `feature_engineering.py`:

| Function | Description |
|---|---|
| `sanitize_name(name)` | Normalize species/move names — remove non-alphanumeric, lowercase |
| `bin_hp(hp_val)` | Bin HP% into categorical labels (Fainted/Sash/Critical/Low/Middle/High/Full) |
| `get_smogon_usages_df(filepath, top_n)` | Load and prune Smogon usage JSON |
| `find_active_species(row, player_prefix)` | Find active Pokémon in slot-based df row |
| `build_medium_features(df, ...)` | Full medium feature set builder (vectorized) — returns `(X, numerical_features, categorical_features)` |

**Important:** `train_action_predictor.py` will NOT be modified. `feature_engineering.py` is new code that `train_action_predictor_embedding.py` will import. In a future cleanup phase, `train_action_predictor.py` can optionally be refactored to import from it too.

---

## 6. Model Artifact Naming

To prevent clashing with existing stable artifacts (`action_lgbm_model_v4_*`, `action_tf_model_v4_*`):

| Artifact | Path |
|---|---|
| Keras model | `action_tf_embedding_v1_{feature_set}.keras` |
| Label encoder | `action_label_encoder_v1_{feature_set}.joblib` |
| TF preprocessor | `action_tf_embedding_preprocessor_v1_{feature_set}.joblib` |
| Metadata JSON | `action_tf_embedding_metadata_v1_{feature_set}.json` |

The `v1` prefix clearly distinguishes embedding model artifacts from the `v4` plain-TF artifacts.

---

## 7. Known TF/Keras Gotchas

- **Dictionary inputs are most robust:** Use `model({'species_in': x, 'numerical_in': y, ...})` matched to named `Input` layers. Avoids ordering bugs.
- **Sparse inputs:** For binary vectors <1000 features, convert to dense before `BatchNormalization` — BN is not compatible with `SparseTensor`.
- **Embedding input type:** `Embedding` layers expect `int32` inputs, not float. The integer-encoded species/move IDs need a separate `LabelEncoder` (or `StringLookup` layer) from the one encoding the target variable.
- **`mask_zero=True`:** Use on `Embedding` layers if padding is needed (relevant for ragged move sequences but not for the dense single-value species embedding).
- **GPU:** Embedding layers are efficiently handled on GPU by default. No special flags needed vs the existing pipeline.

---

## Summary of Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Keras Functional API | Required for multi-input architecture |
| Species embedding dim | 48 | Balances expressiveness vs. parameter count |
| Move embedding dim | 32 | Moves are less diverse than species |
| Multi-hot revealed moves | Dense(32, no_bias) | Reuses existing pipeline, sum-pool equivalent |
| Post-concat stack | 512→256→128→classes | Higher capacity than current 256→128→64 |
| Activation | swish | Smoother gradients than relu |
| Shared module | `feature_engineering.py` | New file; old script untouched |
| Artifact prefix | `action_tf_embedding_v1_*` | No clash with existing v4 artifacts |
