# Requirements

## REQ-01: Generalization Hardening
The training pipeline must be modified to prevent match-specific memorization (fingerprinting).

### Acceptance Criteria
- [ ] **Semantic Mapping**: Species features are converted to Types and Stats before training.
- [ ] **Frequency Pruning**: Input columns for revealed moves are pruned unless they appear in at least 50 unique replays.
- [ ] **LogLoss Goal**: Evaluation LogLoss for the Action Predictor drops below 1.8.
- [ ] **Accuracy Stability**: Accuracy gap between Train and Eval is reduced to < 25%.

## REQ-02: Inference Parity
The bot's real-time prediction logic must match the training pre-processing exactly.

### Acceptance Criteria
- [ ] **Normalization**: Both pipelines use the 100% identical Regex-based `sanitize_name` function.
- [ ] **Parity Script**: `verify_parity.py` passes with zero discrepancies between training and inference logs.

## REQ-03: Bot Integration (Inference)
The trained model must be deployable to `predict_action.py` for live battle tests.

### Acceptance Criteria
- [ ] **Feature Mapping**: `predict_action.py` correctly maps live board state to the new semantic (Stats/Types) feature format.
- [ ] **Action Selection**: The bot correctly suggests moves based on the latest model weights.

## Non-Functional Requirements
- **Performance**: Feature engineering during training should complete in < 5 minutes for a 30k sample.
- **Maintainability**: Centralize all normalization logic in a shared module or consistent helper blocks.
