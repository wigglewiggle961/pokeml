# Architecture Research (PokéML)

## Data Flow
1. **Raw Replay (.json/.log)**: Collected from Showdown.
2. **Parser (`process_replays.py`)**: Flattens state into turns.
3. **Feature Engineer (`train_*.py`)**: Maps species to Stats/Types and prunes moves.
4. **Training**: LightGBM/TF models generated.
5. **Inference (`predict_action.py`)**: Bot queries model based on live socket state.

## Component Boundaries
- **Encoding Layer**: Should be a standalone function shared between `process_replays` and `predict_action`.
- **Stat Reference**: A JSON/Dict containing all 1000+ Pokémon attributes.
- **Ensemble Logic**: Post-processing model probabilities to favor high-utility moves (e.g., Stealth Rock on Turn 1).

## Suggested Build Order
1. **Shared Sanitization Module**: Fix the drift between regex and string replace.
2. **Stat Mapping Utility**: Build the Species -> Attributes lookup.
3. **Training Update**: Retrain with new features.
4. **Inference Update**: Deploy to the bot.
