# Summary of CARD Model Debugging and Enhancement

This document summarizes the debugging process for the "stationary attention" issue in the CARD model and the ongoing efforts to align the `vibes.py` implementation with the CARD research paper.

## Initial Problem: "Stationary Attention"

The initial problem was that the attention mechanism in the CARD model was "stationary," meaning it was not dynamically adjusting its focus on past data points. This was observed through consistently identical debug outputs for the attention statistics (Min: 0.0000, Max: 1.0000, Mean: 0.0909) and poor model performance (low Direction Accuracy).

## Debugging Process and Key Observations

A series of hypotheses were tested to diagnose and resolve the stationary attention issue. These included:

1.  **Positional Embedding Dominance:** Scaled input embeddings, corrected several bugs in `DataEmbedding`, and even temporarily disabled positional encoding.
2.  **Causal Masking:** Implemented a strict causal mask to align with the paper's description.
3.  **Training Epochs:** Increased training epochs to allow the model more time to learn.
4.  **Attention Temperature:** Adjusted the `attention_temperature` hyperparameter.
5.  **Token Embedding Padding:** Changed the `padding_mode` in `TokenEmbedding` from `circular` to `reflect`.
6.  **Numerical Stability (`nan_to_num`):** Removed `nan_to_num` calls from `MultiHeadAttention` to check for masked numerical issues.

**Key Observation:** Despite these significant changes, the debug output for the attention statistics remained *absolutely identical* for every sample in the batch. This strongly suggests that the issue is either a very subtle bug or a misleading debug report.

## User Insight and Current Understanding

The user provided a crucial insight: "the attention heads very strongly aggregate to certain values." This suggests that the attention is not uniformly stationary, but rather highly concentrated on a few specific values or positions. This is consistent with the `Min: 0.0000, Max: 1.0000` debug output.

## Discrepancies with the CARD Paper

A detailed comparison of the `vibes.py` implementation with the CARD research paper revealed several significant discrepancies. The `vibes.py` implementation is a simplified Transformer encoder, not a full implementation of the CARD model. Key missing components include:

*   **`T0` Token:** An extra token for static covariates/longer history summary.
*   **EMA in Attention:** Exponential Moving Average (EMA) on `Q` and `K` projections.
*   **Attention in Hidden Dimensions (`Ac:i2`):** An additional attention structure for local information.
*   **CARD Attention Over Channels:** The mechanism for attention over channels.
*   **Token Blend Module:** A module for merging adjacent tokens to capture multi-scale knowledge.
*   **Signal Decay-Based Loss Function:** The paper's proposed loss function.

## Recent and Ongoing Enhancements

To address these discrepancies and move closer to the CARD paper's implementation, the following changes have been made:

1.  **Exponential Moving Average (EMA):** The EMA has been implemented in the `MultiHeadAttention` mechanism.
2.  **`T0` Token:** The `T0` token has been implemented in the `DataEmbedding` class.
3.  **Updated Attention Visualization:** The `card_attention_viz.py` script has been updated to include Dynamic Time Warping (DTW) scores and the sum of attention scores for each patch, providing a more sophisticated analysis of the attention patterns.

## Next Steps

The plan is to continue implementing the remaining missing components from the CARD paper to create a more faithful and potentially more performant model. The remaining components to be implemented are:

*   Attention in hidden dimensions (`Ac:i2`).
*   CARD Attention Over Channels.
*   Token Blend Module.
*   Signal decay-based loss function.

These changes will be implemented one by one, with evaluations at each step to observe their impact on the attention mechanism and model performance.
