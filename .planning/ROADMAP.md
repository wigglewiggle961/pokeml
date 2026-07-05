# Roadmap

## Milestone 1: Generalization & Parity

### Phase 1: Training Optimization
Focus on hardening the model against overfitting using semantic features and frequency-based pruning.
**Goal**: Narrow the Train/Eval gap.
- **Depends on**: None
- **Acceptance Criteria**: LogLoss < 1.8, Semantic mapping implemented.

### Phase 2: Sanitization & Parity Cleanup
Standardize the cleaning and normalization logic across the entire codebase to ensure inference matches training.
**Goal**: 100% parity.
- **Depends on**: Phase 1
- **Acceptance Criteria**: `verify_parity.py` passes.

### Phase 3: Bot Integration
Update the live battle bot to use the new features and model.
**Goal**: Reliable live inference.
- **Depends on**: Phase 2
- **Acceptance Criteria**: `predict_action.py` runs a full game without mapping errors.

### Phase 4: TensorFlow Embedding Model — Refactored Architecture [COMPLETED]
Build a dedicated `train_action_predictor_embedding.py` that replaces dense one-hot encoding with learned entity embeddings (species, moves, types) and a multi-input neural network architecture. Extract shared utilities into `feature_engineering.py`.
**Goal ACHIEVED**: Hit **38.9% test accuracy**, shattering the 28% ceiling.
- **Depends on**: Phase 1
- **Result**: Validated generalization (12% gap) and shared module success.

### Phase 5: Hyperparameter Tuning
Automate the search for optimal hyperparameters (embedding dimensions, classification head depth/width, learning rates, batch sizes) to further improve model performance.
**Goal**: Optimize `train_action_predictor_embedding.py`.
- **Depends on**: Phase 4
- **Acceptance Criteria**: Validation Accuracy > 40%, Top-5 Accuracy > 75%.

## Milestone 2: Scaling & Advanced Features (Backlog)

### Phase 999.1: Ensemble Prediction
Combine the Action Predictor and Switch Predictor into a single decision-making unit.
### Phase 999.2: Usage-Stats Weighting
Adjust model outputs based on Smogon usage rates for better "expected value" decisions.
