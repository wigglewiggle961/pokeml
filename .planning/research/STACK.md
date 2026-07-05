# Stack Research (PokéML)

## Standard Stack Recommendation (2025)
For handling sparse categorical replay data in Pokémon Showdown:

| Component | Choice | Rationale | Confidence |
|-----------|--------|-----------|------------|
| **Core Model** | LightGBM | Superior at handling sparse binary features and categorical species data compared to standard MLPs. | 95% |
| **Preprocessing** | Pandas + NumPy | Standard for tabular data manipulation; `get_dummies` for binary revealed moves. | 100% |
| **Optimization** | Optuna | Automated hyperparameter tuning (Learning Rate, Depth, `min_child_samples`). | 90% |
| **Inference Backend** | Python (Custom) | Seamlessly integrates with `poke-env` and standard joblib model loaders. | 100% |

## Libraries & Versions
- `lightgbm >= 4.0.0`
- `pandas >= 2.1.0`
- `numpy >= 1.24.0`
- `joblib >= 1.3.0`

## What NOT to use
- **Naive One-Hot Encoding**: Avoid for species if using high-cardinality names; use Stat-embeddings instead.
- **Large Neural Networks**: Can be harder to tune for tabular Pokémon data without extensive embedding architecture.
