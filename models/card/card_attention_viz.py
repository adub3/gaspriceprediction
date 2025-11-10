"""
CARD Attention Pattern Visualization - Updated for Enhanced Model
Save as: models/card/card_attention_viz.py

Extracts and visualizes attention patterns from CARD transformer layers
Compatible with the enhanced vibes.py with trend decomposition
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import sys
import os
from dtaidistance import dtw

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from enhanced vibes.py (Config instead of CARDConfig)
from models.card.vibes import (
    Config,  # Changed from CARDConfig
    Model
)


def extract_attention_weights(
    model: Model,
    x_enc: torch.Tensor,
    x_mark_enc: Optional[torch.Tensor] = None,
    layer_idx: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract attention weights from a specific transformer layer
    
    Args:
        model: CARD Model instance
        x_enc: Encoder input (batch, seq_len, channels)
        x_mark_enc: Time features (optional)
        layer_idx: Which encoder layer to extract from (-1 = last layer)
    
    Returns:
        attention_weights: (batch, n_heads, n_patches, n_patches)
        token_embeddings: (batch, n_patches, d_model)
    """
    model.eval()
    device = next(model.parameters()).device
    
    x_enc = x_enc.to(device)
    if x_mark_enc is not None:
        x_mark_enc = x_mark_enc.to(device)
    
    # Create dummy time features if not provided
    if x_mark_enc is None:
        x_mark_enc = torch.zeros(
            (x_enc.size(0), x_enc.size(1), 4),
            device=device
        )
    
    with torch.no_grad():
        # Normalize input
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = torch.clamp(stdev, min=1e-5)
        x_enc_norm = x_enc_norm / stdev
        
        # Check for NaN
        if torch.isnan(x_enc_norm).any():
            x_enc_norm = torch.nan_to_num(x_enc_norm, nan=0.0)
        
        # Apply patching
        patches = model.patching(x_enc_norm)
        
        # Embed patches
        embedded = model.enc_embedding(patches)
        
        # Pass through encoder layers
        x = embedded
        attention_weights = None
        
        for i, layer in enumerate(model.encoder.layers):
            # Apply layer
            x, attn = layer(x, mask=None)
            
            # Store attention from requested layer
            if i == layer_idx or (layer_idx == -1 and i == len(model.encoder.layers) - 1):
                attention_weights = attn
        
        token_embeddings = x
    
    return attention_weights, token_embeddings


def visualize_attention_patterns(
    attention_weights: torch.Tensor,
    input_sequence: torch.Tensor,
    output_sequence: torch.Tensor,
    n_patches: int,
    patch_len: int,
    stride: int,
    save_path: str = 'attention_viz.png',
    sample_idx: int = 0,
    channel_idx: int = 0,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Visualize attention patterns with detailed analysis
    
    Args:
        attention_weights: (batch, n_heads, n_patches, n_patches)
        input_sequence: (batch, seq_len, channels)
        output_sequence: (batch, pred_len, channels)
        n_patches: Number of patches
        patch_len: Length of each patch
        stride: Stride between patches
        save_path: Where to save the visualization
        sample_idx: Which sample in batch to visualize
        channel_idx: Which channel to visualize
        figsize: Figure size
    """
    # Extract single sample and channel
    attn = attention_weights[sample_idx].cpu().numpy()  # (n_heads, n_patches, n_patches)
    input_seq = input_sequence[sample_idx, :, channel_idx].cpu().numpy()
    output_seq = output_sequence[sample_idx, :, channel_idx].cpu().numpy()
    
    n_heads = attn.shape[0]
    seq_len = len(input_seq)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, n_heads, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle(
        f'CARD Attention Pattern Analysis - Sample {sample_idx}, Channel {channel_idx}',
        fontsize=16,
        fontweight='bold'
    )
    
    # 1. Attention heatmaps for each head
    for head_idx in range(n_heads):
        ax = fig.add_subplot(gs[0, head_idx])
        
        # Plot attention matrix
        sns.heatmap(
            attn[head_idx],
            cmap='viridis',
            cbar=True,
            square=True,
            ax=ax,
            vmin=0,
        )
        ax.set_title(f'Head {head_idx + 1}', fontsize=12)
        ax.set_xlabel('Key Patch Index')
        ax.set_ylabel('Query Patch Index')

    # 2. DTW vs. Sum of Attention Scores
    ax = fig.add_subplot(gs[1, :])

    # Compute DTW scores between each patch and the forecast sequence
    dtw_scores = []
    for i in range(n_patches):
        patch_start = i * stride
        patch_end = min(patch_start + patch_len, seq_len)
        patch = input_seq[patch_start:patch_end]
        distance = dtw.distance(patch, output_seq)
        dtw_scores.append(distance)
    dtw_scores = np.array(dtw_scores)

    # Sum of attention scores for each patch (sum over query dimension)
    sum_attention_scores = attn.mean(axis=0).sum(axis=0)

    # Normalize scores for plotting
    dtw_scores_norm = (dtw_scores - dtw_scores.min()) / (dtw_scores.max() - dtw_scores.min() + 1e-8)
    sum_attention_scores_norm = (sum_attention_scores - sum_attention_scores.min()) / (sum_attention_scores.max() - sum_attention_scores.min() + 1e-8)

    patch_indices = np.arange(n_patches)
    width = 0.35

    ax.bar(patch_indices - width/2, dtw_scores_norm, width, label='DTW Scores (normalized)', alpha=0.6, color='steelblue')
    ax.bar(patch_indices + width/2, sum_attention_scores_norm, width, label='Sum of Attention Scores (normalized)', alpha=0.6, color='indianred')

    ax.set_xlabel('Patch Index')
    ax.set_ylabel('Normalized Score')
    ax.set_title('DTW vs. Sum of Attention Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Input time series with attention overlay
    ax = fig.add_subplot(gs[2, :])
    seq_len = len(input_seq)
    pred_len = len(output_seq)

    # Plot input (history)
    ax.plot(range(seq_len), input_seq, 'b-', linewidth=1.5, label='History', alpha=0.7)

    # Plot output (future)
    ax.plot(
        range(seq_len, seq_len + pred_len),
        output_seq,
        'r-',
        linewidth=1.5,
        label='Future',
        alpha=0.7
    )

    # Mark patch boundaries and overlay attention
    for i in range(n_patches):
        patch_start = i * stride
        ax.axvline(patch_start, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.text(patch_start + patch_len / 2, ax.get_ylim()[0], f'{sum_attention_scores_norm[i]:.2f}', ha='center', va='bottom', fontsize=8, color='indianred')

    ax.axvline(seq_len, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Forecast start')
    ax.set_title('Time Series with Sum of Attention Scores')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()


def analyze_attention_statistics(
    attention_weights: torch.Tensor,
    n_patches: int
) -> dict:
    """
    Compute statistical properties of attention patterns
    
    Args:
        attention_weights: (batch, n_heads, n_patches, n_patches)
        n_patches: Number of patches
    
    Returns:
        dict with attention statistics
    """
    attn = attention_weights.cpu().numpy()
    batch_size, n_heads, _, _ = attn.shape
    
    # Average across batch and heads
    attn_avg = attn.mean(axis=(0, 1))
    
    # Entropy of attention distribution (higher = more uniform)
    entropy = -np.sum(attn_avg * np.log(attn_avg + 1e-10), axis=-1)
    
    # Sparsity (percentage of attention weights < 0.01)
    sparsity = (attn_avg < 0.01).mean()
    
    # Max attention per query
    max_attention = attn_avg.max(axis=-1)
    
    # Attention concentration (how much attention is on top-k tokens)
    top_k = 5
    top_k_attention = np.sort(attn_avg, axis=-1)[:, -top_k:].sum(axis=-1)
    
    stats = {
        'mean_entropy': float(entropy.mean()),
        'sparsity': float(sparsity),
        'mean_max_attention': float(max_attention.mean()),
        'mean_top_k_concentration': float(top_k_attention.mean()),
        'attention_range': (float(attn_avg.min()), float(attn_avg.max()))
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    print("CARD Attention Visualization Module")
    print("=" * 50)
    print("\nThis module provides attention analysis for CARD models.")
    print("\nUsage:")
    print("  from models.card.card_attention_viz import extract_attention_weights, visualize_attention_patterns")
    print("\nOr use with cardeval.py:")
    print("  python -m tests.cardeval --example attention --csv data.csv --checkpoint model.pt")