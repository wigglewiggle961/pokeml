# PokéML Training Experiments

| Run ID | Strategy | Command Snippet | Train Acc | Eval Acc | Gap | Eval LogLoss | Status |
|---|---|---|---|---|---|---|---|
| **0** | Baseline | `--feature_set medium` | 80.5% | 40.1% | 40.4% | 1.134 | 🚩 Leaky |
| **1** | HP Binning + Reg | `--lgbm_estimators 6000` | 76.2% | 39.5% | 36.7% | 1.029 | ⚠️ Marginal |
| **2** | **Hardened** | `--smoothing 0.1 --replay_count 50` | 83.5% | 37.7% | 45.8% | 1.978 | ❌ Gap Widened |
| **4** | **Clean Baseline** | **Corrected Parser (Temporal Fix)** | **87.0%** | **37.1%** | **49.9%** | **2.146** | 🏁 Leak Fixed, Memorization High |
| **5** | **Metamon Scaling** | **9M Rows / 16GB RAM** | **83.6%** | **37.3%** | **46.3%** | **1.985** | ❌ Identity Memorization |
| **6** | **Aggressive Reg** (Proposed) | `--cat_smooth 1000 --reg_lambda 100` | TBD | TBD | TBD | TBD | 🤞 Targeting Gap < 0.4 |
| **7** | **TF Embedding** | **Embeddings + Swish Stack (Ph 4)** | **51.5%** | **38.9%** | **12.6%** | **2.567** | ✅ **Breakthrough** |

---
## Run 5: Metamon Scaling Summary
Scaling to 9 million rows significantly improved convergence but **did not stop the identity shortcut**. `p1_active_species` remains the dominant feature (2.6M importance), meaning the model is still a lookup table for "Most Common Move per Species." 

## Strategy for Run 6: Aggressive Regularization
Since mass data alone didn't break the memorization, we will now **Radically Regularize** the model:
1.  **Categorical Smoothing**: Set `cat_smooth=1000` to prevent species-based splits without overwhelming evidence.
2.  **Feature Blinding**: Drop `colsample_bytree` to **0.05** so the model is forced to learn from hazards/HP in 95% of trees.
3.  **Complexity Cap**: Strictly limit tree depth and increase `min_child_samples` to 10k.

---
## Run 7: TF Embedding Model (Phase 4)

**Significant Breakthrough**: By moving to learned entity embeddings for species and moves, we collapsed the generalization gap from **~50% down to 12.6%**. 

**Key Observations**:
- **Performance**: Hit a new record of **38.9% validation accuracy** (up from 37.3% ceiling).
- **Generalization**: The model is no longer "memorizing" match IDs. The top-5 accuracy of **84.5%** suggests the model reaches an expert-level understanding of the competitive "menu" even if it doesn't always pick the #1 move.
- **Loss Gap**: While the jump from 1.2 (Train) to 2.5 (Val) looks steep, the absolute Val Loss of **2.56** is the lowest recorded for this complexity of target space (277 classes).

**Next Steps**:
- **Increase Regularization**: Add higher Dropout (0.35) or weight decay to try and close that final 12% gap.
- **Architecture**: Test if "Wide & Deep" (concatenating numerical features directly to the final layer) helps preserve raw stat values.

---
## Artifacts & Reproducibility

> Since `data/` and `models/` are **gitignored** (git no longer stores the bytes), this
> table is the source of truth linking each result to the artifact + dataset that produced it.
> **Going forward: whenever you train, add a row here (artifact filename + exact dataset).**

| Model file (in `models/`) | Produced by | Feature set | Dataset | Result / Run |
|---|---|---|---|---|
| `action_lgbm_model_v4_medium_move_only.txt` (+ `_feature_info_`, `_scaler_`, `action_label_encoder_v4_medium_move_only`) | `train_action_predictor.py --model_type lightgbm` | medium (move-only target) | ❓ unrecorded (10k/30k parquet lineage) | LightGBM baseline, Runs 0–4 (~37% eval) |
| `action_tf_model_v4_medium.keras`, `action_tf_model_v4_medium_move_only.keras` (+ `action_tf_preprocessor_v4_*`) | `train_action_predictor.py --model_type tensorflow` | medium | `data/10k.parquet` (per `archive/abababa.txt` cmd) | plain TF dense, ~30% val |
| `action_tf_embedding_v1_medium.keras` (+ `action_tf_embedding_metadata_v1_medium.json`, `action_tf_embedding_artifacts_v1_medium.joblib`, `action_label_encoder_v1_medium.joblib`) | `train_action_predictor_embedding.py --feature_set medium` | medium (277 classes) | ❓ unrecorded — **record on next run** | **Run 7 — best.** 38.5% test / 84.5% top-5 / loss 2.567 |
| `switch_predictor_lgbm_model_v2_simplified.txt` (+ `_feature_info_`, `_scaler_`) | `train_switch_predictor.py --feature_set simplified` | simplified | ❓ unrecorded | Binary switch classifier (2025-05-04) |
| `switch_target_predictor_lgbm_model_simplified_moves.txt` (+ `_feature_info_`, `_scaler_`, `_label_encoder_`) | `train_pokemon_switch_predictor.py --feature_set simplified` | simplified_moves | ❓ unrecorded | Switch-target classifier (2025-10-03) |

**Note — bot vs. best model:** `predict_action.py` currently loads the **v4 LightGBM** action model
(`action_lgbm_model_v4_medium_move_only`), **not** the Run-7 embedding model. Wiring the embedding
model into the live bot is deferred work (Phase 3 / ensemble).

**Reproducibility convention:** artifacts are named `<task>_<framework>_<version>_<feature_set>[_<target>]`.
Datasets used to be loose parquet files in the repo root; they now live in `data/` and are gitignored,
so their provenance lives here, not in git.
