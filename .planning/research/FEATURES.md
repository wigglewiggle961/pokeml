# Features Research (PokéML)

## Table Stakes (Must-Have)
- **Active Board State**: High-fidelity tracking of current HP, status, and turns.
- **Type Effectiveness**: Model must account for 18x18 type matchups.
- **Movepools**: Prediction must be constrained to legal moves based on Smogon usage or direct observation.

## Differentiators (Competitive Advantage)
- **Bench-Aware Strategy**: Using known bench members to predict switches vs. moves.
- **Parity-First Inference**: Ensuring zero drift between training and real-time bots.

## Anti-Features (Will Not Build)
- **Match-ID Memorization**: Explicitly preventing the model from recognizing specific players or replay IDs.
- **Real-time Chat Analysis**: Parsing text logs for strategy (too noisy/unreliable).
