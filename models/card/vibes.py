"""
CARD Model - Complete Implementation for Gas Price Forecasting
models/card/vibes.py - Enhanced with all locality and trend fixes

This file contains everything: model, training, evaluation, metrics, and visualization.
"""

import os
import math
from datetime import datetime
from typing import Dict, Tuple, Optional
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from einops import rearrange


# ============================================================================
# Configuration
# ============================================================================
class CARDConfig:
    """Enhanced configuration for gas price forecasting"""
    def __init__(
        self,
        # Task settings
        task_name: str = 'long_term_forecast',
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        
        # Model architecture - OPTIMIZED FOR GAS PRICES
        enc_in: int = 1,
        dec_in: int = 1,
        c_out: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        
        # CARD-specific params - FIXED FOR LOCALITY
        patch_len: int = 16,
        stride: int = 4,  # OVERLAPPING patches for locality
        merge_size: int = 2,
        dp_rank: int = 16,
        alpha: float = 0.2,
        momentum: float = 0.1,
        use_statistic: bool = True,
        
        # Forecasting strategy
        autoregressive: bool = True,
        ar_steps: int = 24,
        
        # Teacher forcing schedule
        teacher_forcing_start: float = 0.5,
        teacher_forcing_end: float = 0.0,
        
        # Normalization
        global_norm: bool = True,
        normalize_deltas: bool = True,
        
        # EMA control
        use_ema: bool = False,
        ema_first_layer_only: bool = True,
        
        # Multi-scale output
        use_hybrid_head: bool = True,
        
        # Loss configuration
        loss_huber_weight: float = 0.7,
        loss_slope_weight: float = 0.3,
        huber_delta: float = 1.0,
        
        # For classification
        num_class: int = 10
    ):
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.patch_len = patch_len
        self.stride = stride
        self.merge_size = merge_size
        self.dp_rank = dp_rank
        self.alpha = alpha
        self.momentum = momentum
        self.use_statistic = use_statistic
        self.num_class = num_class
        
        self.autoregressive = autoregressive
        self.ar_steps = ar_steps
        self.teacher_forcing_start = teacher_forcing_start
        self.teacher_forcing_end = teacher_forcing_end
        
        self.global_norm = global_norm
        self.normalize_deltas = normalize_deltas
        
        self.use_ema = use_ema
        self.ema_first_layer_only = ema_first_layer_only
        
        self.use_hybrid_head = use_hybrid_head
        
        self.loss_huber_weight = loss_huber_weight
        self.loss_slope_weight = loss_slope_weight
        self.huber_delta = huber_delta
        
        # Will be set by model
        self.total_token_number = None
        self.internal_channels = enc_in * 3


# ============================================================================
# Global Normalizer
# ============================================================================
class GlobalNormalizer:
    """Maintains global statistics for proper price scale preservation"""
    def __init__(self, normalize_deltas: bool = True):
        self.mean = None
        self.std = None
        self.normalize_deltas = normalize_deltas
        
    def fit(self, data: np.ndarray):
        """Fit on training data"""
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std = np.maximum(self.std, 1e-4)
        
    def normalize(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize using global statistics"""
        if self.mean is None:
            raise ValueError("Must call fit() before normalize()")
        
        mean = torch.FloatTensor(self.mean).to(data.device)
        std = torch.FloatTensor(self.std).to(data.device)
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
        
        normalized = (data - mean) / std
        return normalized, mean, std
    
    def denormalize(self, data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize back to original scale"""
        return data * std + mean


# ============================================================================
# Helper Functions
# ============================================================================
def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Build causal attention mask"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def add_derivative_channels(x: torch.Tensor, normalize_deltas: bool = True, 
                           mean: Optional[torch.Tensor] = None, 
                           std: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Add first and second derivative channels"""
    # First derivative
    dx = torch.zeros_like(x)
    dx[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    
    # Second derivative
    ddx = torch.zeros_like(x)
    ddx[:, :, 1:] = dx[:, :, 1:] - dx[:, :, :-1]
    
    # Normalize derivatives with same stats as base if requested
    if normalize_deltas and mean is not None and std is not None:
        dx = (dx - mean) / std
        ddx = (ddx - mean) / std
    
    # Concatenate along channel dimension
    x_aug = torch.cat([x, dx, ddx], dim=1)
    return x_aug


# ============================================================================
# Model Components
# ============================================================================

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Attenion(nn.Module):
    """Modified attention with optional EMA and causal masking"""
    def __init__(self, config, over_hidden=False, layer_idx=0):
        super().__init__()
        
        self.over_hidden = over_hidden
        self.n_heads = config.n_heads
        self.c_in = config.internal_channels
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        
        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear(config.d_model, config.d_model)
        
        self.norm_post1 = nn.LayerNorm(config.d_model)
        self.norm_post2 = nn.LayerNorm(config.d_model)
        self.norm_attn  = nn.LayerNorm(config.d_model)
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        
        self.ff_1 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True)
        )
        
        self.ff_2 = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=True)
        )     
        self.merge_size = config.merge_size
        
        # EMA control
        self.use_ema = config.use_ema and (not config.ema_first_layer_only or layer_idx == 0)
        
        if self.use_ema:
            ema_size = max(config.internal_channels, config.total_token_number or 100, config.dp_rank)
            ema_matrix = torch.zeros((ema_size, ema_size))
            alpha = config.alpha
            ema_matrix[0][0] = 1
            total_tokens = config.total_token_number or 100
            for i in range(1, min(total_tokens, ema_size)):
                for j in range(i):
                    ema_matrix[i][j] = ema_matrix[i-1][j] * (1 - alpha)
                ema_matrix[i][i] = alpha
            self.register_buffer('ema_matrix', ema_matrix)
    
    def ema(self, src):
        if not self.use_ema:
            return src
        return torch.einsum('bnhad,ga->bnhgd', src, 
                           self.ema_matrix[:src.shape[-2], :src.shape[-2]])
    
    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = torch.clamp(src_dp, -30, 30)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('bnhef,bnhec->bnhcf', src, src_dp)
        return src_dp
    
    def forward(self, src, causal_mask=None):
        B, nvars, H, C = src.shape
        
        qkv = self.qkv(src).reshape(B, nvars, H, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if not self.over_hidden: 
            attn_score_along_token = torch.einsum(
                'bnhed,bnhfd->bnhef', 
                self.ema(q), 
                self.ema(k)
            ) / math.sqrt(self.head_dim)
            
            if causal_mask is not None:
                attn_score_along_token = attn_score_along_token + causal_mask
            
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)
        else:
            v_dp = self.dynamic_projection(v, self.dp_v)
            k_dp = self.dynamic_projection(k, self.dp_k)
            attn_score_along_token = torch.einsum(
                'bnhed,bnhfd->bnhef',
                self.ema(q),
                self.ema(k_dp)
            ) / math.sqrt(self.head_dim)
            
            if causal_mask is not None:
                attn_score_along_token = attn_score_along_token + causal_mask
            
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1))
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)
        
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q, k) / math.sqrt(q.shape[-2])
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1))
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)

        merge_size = self.merge_size
        output1 = rearrange(
            output_along_token.reshape(B * nvars, -1, self.head_dim),
            'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d', 
            hl1=self.n_heads // merge_size, 
            hl2=output_along_token.shape[-2],
            hl3=merge_size
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)
        
        output2 = rearrange(
            output_along_hidden.reshape(B * nvars, -1, self.head_dim),
            'bn (hl1 hl2 hl3) d -> bn hl2 (hl3 hl1) d', 
            hl1=self.n_heads // merge_size, 
            hl2=output_along_token.shape[-2],
            hl3=merge_size
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)
        
        output1 = self.norm_post1(output1).reshape(B, nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2).reshape(B, nvars, -1, self.n_heads * self.head_dim)
        
        src2 = self.ff_1(output1) + self.ff_2(output2)
        src = src + src2
        src = src.reshape(B * nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.n_heads * self.head_dim)
        return src


class HybridOutputHead(nn.Module):
    """Multi-scale output: local (per-patch) + global (pooled) predictions"""
    def __init__(self, config, n_patches: int):
        super().__init__()
        self.pred_len = config.pred_len
        self.stride = config.stride
        self.patch_len = config.patch_len
        self.seq_len = config.seq_len
        self.n_patches = n_patches
        self.d_model = config.d_model
        
        # Calculate actual reconstruction length from patching
        self.recon_len = (n_patches - 1) * self.stride + self.patch_len
        
        # Local head: per-patch prediction of stride steps (sliding window)
        self.local_proj = nn.Linear(config.d_model, config.stride)
        
        # Global head: pooled representation to full sequence
        self.global_proj = nn.Linear(config.d_model, config.pred_len)
        
        # Learnable mixing weights
        self.w_local = nn.Parameter(torch.tensor(0.0))
        self.w_global = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, token_embeddings):
        batch, channels, n_tokens, d_model = token_embeddings.shape
        
        # Remove CLS/stat token (first token)
        patch_tokens = token_embeddings[:, :, 1:, :]
        actual_patches = patch_tokens.shape[2]
        
        # LOCAL HEAD - predict stride values per patch (sliding)
        local_preds = self.local_proj(patch_tokens)  # [B, C, n_patches, stride]
        
        # Overlap-add reconstruction
        local_full = torch.zeros(batch, channels, self.recon_len, device=local_preds.device)
        counts = torch.zeros(batch, channels, self.recon_len, device=local_preds.device)
        
        for i in range(actual_patches):
            start = i * self.stride
            end = start + self.stride
            if end <= local_full.shape[2]:
                local_full[:, :, start:end] += local_preds[:, :, i, :]
                counts[:, :, start:end] += 1.0
        
        # Average overlapping regions
        local_full = local_full / (counts + 1e-8)
        
        # Take the last pred_len steps (this is the prediction horizon)
        if local_full.shape[2] >= self.pred_len:
            local_full = local_full[:, :, -self.pred_len:]
        else:
            # If reconstruction is shorter than pred_len, pad with last value
            pad_len = self.pred_len - local_full.shape[2]
            local_full = torch.nn.functional.pad(local_full, (0, pad_len), mode='replicate')
        
        # GLOBAL HEAD
        pooled = torch.mean(patch_tokens, dim=2)
        global_full = self.global_proj(pooled)
        
        # Mix with learnable weights
        w_l = torch.sigmoid(self.w_local)
        w_g = torch.sigmoid(self.w_global)
        total = w_l + w_g + 1e-8
        
        output = (w_l * local_full + w_g * global_full) / total
        return output


class CARDformer(nn.Module):
    """Enhanced CARDformer with all fixes"""
    def __init__(self, config, normalizer: Optional[GlobalNormalizer] = None):
        super().__init__()
        
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.task_name = config.task_name
        self.global_norm = config.global_norm
        self.normalizer = normalizer
        self.normalize_deltas = config.normalize_deltas
        self.use_hybrid_head = config.use_hybrid_head
        
        # Multi-scale conv front-end
        self.edge_conv = nn.Conv1d(
            in_channels=config.enc_in,
            out_channels=config.enc_in,
            kernel_size=3,
            padding=1,
            groups=config.enc_in
        )
        
        # Calculate patch count
        patch_num = int((config.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num, config.d_model) * 0.02)
        self.model_token_number = 0
        
        self.total_token_number = (self.patch_num + self.model_token_number + 1)
        config.total_token_number = self.total_token_number
        
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)  
        self.input_dropout = nn.Dropout(config.dropout) 
        
        self.use_statistic = config.use_statistic
        self.W_statistic = nn.Linear(2, config.d_model) 
        self.cls = nn.Parameter(torch.randn(1, config.d_model) * 0.02)
        
        # Output head
        if config.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.use_hybrid_head:
                self.output_head = HybridOutputHead(config, patch_num)
            else:
                self.W_out = nn.Linear((patch_num + 1) * config.d_model, config.pred_len)
        elif config.task_name == 'imputation' or config.task_name == 'anomaly_detection':
            self.W_out = nn.Linear((patch_num + 1) * config.d_model, config.seq_len) 
        elif config.task_name == 'classification':
            self.W_out = nn.Linear(config.d_model * config.internal_channels, config.num_class)
        
        # Attention layers
        self.Attentions_over_token = nn.ModuleList([
            Attenion(config, over_hidden=False, layer_idx=i) for i in range(config.e_layers)
        ])
        
        # Conditional over-channel attention
        self.bypass_over_channel = (config.enc_in <= 2)
        if not self.bypass_over_channel:
            self.Attentions_over_channel = nn.ModuleList([
                Attenion(config, over_hidden=True, layer_idx=i) for i in range(config.e_layers)
            ])
        
        self.Attentions_mlp = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model) for _ in range(config.e_layers)
        ])
        self.Attentions_dropout = nn.ModuleList([
            nn.Dropout(config.dropout) for _ in range(config.e_layers)
        ])
        self.Attentions_norm = nn.ModuleList([
            nn.LayerNorm(config.d_model) for _ in range(config.e_layers)
        ])
        
        # Build causal mask
        self.register_buffer('causal_mask', build_causal_mask(self.total_token_number, torch.device('cpu')))

    def forward(self, z, return_norm_params=False):
        b, c_base, s = z.shape
        device = z.device
        
        if self.causal_mask.device != device:
            self.causal_mask = self.causal_mask.to(device)
        
        # Global normalization
        if self.global_norm and self.normalizer is not None:
            z, z_mean, z_std = self.normalizer.normalize(z)
        else:
            z_mean = torch.mean(z, dim=(-1), keepdims=True)
            z_std = torch.std(z, dim=(-1), keepdims=True)
            z_std = torch.clamp(z_std, min=1e-4)
            z = (z - z_mean) / z_std
        
        # Multi-scale conv
        z_edges = self.edge_conv(z)
        z = z + 0.1 * z_edges
        
        # Add derivative channels
        z_aug = add_derivative_channels(z, self.normalize_deltas, z_mean, z_std)
        
        # Patching
        zcube = z_aug.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed
        
        # Add statistics token
        if self.use_statistic:
            z_mean_squeeze = z_mean.squeeze(-1)
            z_std_squeeze = z_std.squeeze(-1)
            
            if z_mean_squeeze.shape[0] == 1 and b > 1:
                z_mean_squeeze = z_mean_squeeze.expand(b, -1)
                z_std_squeeze = z_std_squeeze.expand(b, -1)
            
            z_mean_squeeze = z_mean_squeeze.repeat(1, 3)
            z_std_squeeze = z_std_squeeze.repeat(1, 3)
            
            z_stat = torch.stack([z_mean_squeeze, z_std_squeeze], dim=-1)
            c_aug = z_stat.shape[1]
            
            if c_aug > 1:
                z_stat_reshaped = z_stat.reshape(b * c_aug, 2)
                z_stat_mean = z_stat_reshaped.mean(dim=0, keepdim=True)
                z_stat_std = z_stat_reshaped.std(dim=0, keepdim=True) + 1e-4
                z_stat_reshaped = (z_stat_reshaped - z_stat_mean) / z_stat_std
                z_stat = z_stat_reshaped.reshape(b, c_aug, 2)
            
            z_stat = self.W_statistic(z_stat)
            z_stat = z_stat.unsqueeze(-2)
            z_embed = torch.cat((z_stat, z_embed), dim=-2)
        else:
            cls_token = self.cls.repeat(b, z_embed.shape[1], 1, 1)
            z_embed = torch.cat((cls_token, z_embed), dim=-2)
        
        # Transformer layers
        inputs = z_embed
        causal_mask = self.causal_mask[:inputs.shape[-2], :inputs.shape[-2]]
        
        for layer_idx, (a_tok, mlp, drop, norm) in enumerate(zip(
            self.Attentions_over_token,
            self.Attentions_mlp,
            self.Attentions_dropout,
            self.Attentions_norm
        )):
            # Over-channel attention
            if not self.bypass_over_channel:
                a_ch = self.Attentions_over_channel[layer_idx]
                output_1 = a_ch(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            else:
                output_1 = inputs
            
            # Over-token attention with causal mask
            output_2 = a_tok(output_1, causal_mask=causal_mask)
            
            outputs = drop(mlp(output_1 + output_2)) + inputs
            outputs = norm(outputs)
            inputs = outputs
        
        # Output projection
        if self.task_name != 'classification':
            if self.use_hybrid_head:
                outputs_base = outputs.reshape(b, c_base, 3, -1, self.d_model).sum(dim=2)
                z_out = self.output_head(outputs_base)
            else:
                outputs_flat = outputs.reshape(b, c_base, 3, -1).sum(dim=2)
                z_out = self.W_out(outputs_flat)
            
            z = z_out * z_std + z_mean
        else:
            z = self.W_out(torch.mean(outputs[:, :, :, :], dim=-2).reshape(b, -1))
        
        if return_norm_params:
            return z, z_mean, z_std
        return z


class Model(nn.Module):
    """Wrapper with autoregressive decoding and teacher forcing"""
    def __init__(self, config, normalizer: Optional[GlobalNormalizer] = None):
        super().__init__()    
        self.model = CARDformer(config, normalizer)
        self.task_name = config.task_name
        self.config = config
        self.autoregressive = config.autoregressive
        self.ar_steps = config.ar_steps if config.autoregressive else config.pred_len
        
        self.tf_start = config.teacher_forcing_start
        self.tf_end = config.teacher_forcing_end
        self.current_epoch = 0
        self.max_epochs = 100
    
    def set_epoch(self, epoch: int, max_epochs: int):
        self.current_epoch = epoch
        self.max_epochs = max_epochs
    
    def get_teacher_forcing_prob(self) -> float:
        if self.max_epochs <= 1:
            return self.tf_end
        progress = self.current_epoch / (self.max_epochs - 1)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.tf_end + (self.tf_start - self.tf_end) * cos_decay
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if not self.autoregressive or self.task_name == 'classification':
            x = x_enc.permute(0, 2, 1)
            x = self.model(x)
            
            if self.task_name != 'classification':
                x = x.permute(0, 2, 1)
                return x[:, -self.config.pred_len:, :]
            return x
        
        else:
            batch_size, seq_len, n_channels = x_enc.shape
            predictions = []
            current_input = x_enc
            
            n_iterations = math.ceil(self.config.pred_len / self.ar_steps)
            tf_prob = self.get_teacher_forcing_prob() if self.training else 0.0
            
            for i in range(n_iterations):
                x = current_input.permute(0, 2, 1)
                pred = self.model(x)
                pred = pred.permute(0, 2, 1)
                
                steps_to_predict = min(self.ar_steps, self.config.pred_len - len(predictions) * self.ar_steps)
                pred_window = pred[:, -steps_to_predict:, :]
                predictions.append(pred_window)
                
                if i < n_iterations - 1:
                    if self.training and torch.rand(1).item() < tf_prob:
                        gt_start = self.config.label_len + i * self.ar_steps
                        gt_end = gt_start + steps_to_predict
                        next_chunk = x_dec[:, gt_start:gt_end, :]
                    else:
                        next_chunk = pred_window
                    
                    current_input = torch.cat([
                        current_input[:, steps_to_predict:, :],
                        next_chunk
                    ], dim=1)
            
            final_pred = torch.cat(predictions, dim=1)
            return final_pred[:, :self.config.pred_len, :]


# ============================================================================
# Loss Function
# ============================================================================
def trend_aware_loss(pred, target, config, horizon_weight='uniform'):
    """
    Multi-component loss with Huber + slope awareness
    
    Args:
        pred: [B, T, C] predictions
        target: [B, T, C] ground truth
        config: CARDConfig with loss weights
        horizon_weight: 'uniform', 'linear', or 'sqrt' for time weighting
    """
    B, T, C = pred.shape
    
    # Time-based weights
    if horizon_weight == 'linear':
        weights = torch.linspace(1.0, 2.0, T, device=pred.device)
    elif horizon_weight == 'sqrt':
        weights = torch.sqrt(torch.linspace(1.0, 4.0, T, device=pred.device))
    else:
        weights = torch.ones(T, device=pred.device)
    
    weights = weights.view(1, T, 1)
    
    # 1. Huber loss (magnitude)
    huber = F.smooth_l1_loss(pred, target, reduction='none', beta=config.huber_delta)
    huber_weighted = (huber * weights).mean()
    
    # 2. Slope loss (first differences)
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    target_diff = target[:, 1:, :] - target[:, :-1, :]
    slope_loss = F.mse_loss(pred_diff, target_diff, reduction='none')
    slope_weighted = (slope_loss * weights[:, 1:, :]).mean()
    
    # Combine
    w_huber = config.loss_huber_weight
    w_slope = config.loss_slope_weight
    
    total_loss = w_huber * huber_weighted + w_slope * slope_weighted
    
    return total_loss, huber_weighted, slope_weighted


# ============================================================================
# Dataset
# ============================================================================
class TimeSeriesDataset(Dataset):
    def __init__(
        self, 
        data: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        freq: str = 'h',
        normalizer: Optional[GlobalNormalizer] = None
    ):
        self.data = data
        self.dates = dates if dates is not None else pd.date_range(
            start='2020-01-01', periods=len(data), freq=freq
        )
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.normalizer = normalizer
        
        self.n_samples = len(data) - seq_len - pred_len + 1
    
    def __len__(self):
        return self.n_samples
    
    def _get_time_features(self, dates):
        features = []
        for date in dates:
            features.append([
                date.hour / 23.0,
                date.day / 30.0,
                date.weekday() / 6.0,
                date.month / 11.0
            ])
        return np.array(features, dtype=np.float32)
    
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        x_enc = self.data[s_begin:s_end]
        x_dec = self.data[r_begin:r_end]
        y = self.data[r_begin:r_end]
        
        x_mark_enc = self._get_time_features(self.dates[s_begin:s_end])
        x_mark_dec = self._get_time_features(self.dates[r_begin:r_end])
        
        return (
            torch.FloatTensor(x_enc),
            torch.FloatTensor(x_mark_enc),
            torch.FloatTensor(x_dec),
            torch.FloatTensor(x_mark_dec),
            torch.FloatTensor(y)
        )


# ============================================================================
# Metrics
# ============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculate comprehensive forecast metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    # sMAPE (symmetric MAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_true - y_pred) / (denominator + 1e-8)) * 100
    
    # MASE (Mean Absolute Scaled Error)
    if y_train is not None and len(y_train) > 1:
        naive_errors = np.abs(y_train[1:] - y_train[:-1])
        naive_mae = np.mean(naive_errors)
        if naive_mae > 0:
            mase = mae / naive_mae
        else:
            mase = float('inf')
    else:
        mase = float('nan')
    
    # Direction accuracy
    if y_true.shape[1] > 1:
        true_diff = y_true[:, 1:] - y_true[:, :-1]
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        direction_correct = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
    else:
        direction_correct = float('nan')
    
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "MASE": float(mase),
        "DirectionAccuracy": float(direction_correct)
    }


def calculate_per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                  bins: list = None) -> Dict[str, Dict[str, float]]:
    """Calculate metrics per horizon bin"""
    pred_len = y_true.shape[1]
    
    if bins is None:
        bins = [(0, 12), (12, 24), (24, 48), (48, pred_len)]
    
    results = {}
    
    for start, end in bins:
        end = min(end, pred_len)
        if start >= end:
            continue
        
        y_true_bin = y_true[:, start:end, :]
        y_pred_bin = y_pred[:, start:end, :]
        
        # Direction accuracy
        if end - start > 1:
            true_diff = y_true_bin[:, 1:] - y_true_bin[:, :-1]
            pred_diff = y_pred_bin[:, 1:] - y_pred_bin[:, :-1]
            dir_acc = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
        else:
            dir_acc = float('nan')
        
        mae = np.mean(np.abs(y_true_bin - y_pred_bin))
        
        bin_name = f"h{start+1}-{end}"
        results[bin_name] = {
            "DirectionAccuracy": float(dir_acc),
            "MAE": float(mae)
        }
    
    return results


# ============================================================================
# Visualization
# ============================================================================
def plot_predictions_with_derivatives(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel_idx: int = 0,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    title: str = "CARD Predictions"
):
    """Plot predictions with derivative overlay"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    indices = np.linspace(0, len(y_true) - 1, n_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        row = i // 2
        col = i % 2
        
        # Main plot
        ax_main = fig.add_subplot(gs[row*2, col])
        true_seq = y_true[idx, :, channel_idx]
        pred_seq = y_pred[idx, :, channel_idx]
        x_axis = np.arange(len(true_seq))
        
        ax_main.plot(x_axis, true_seq, label='Ground Truth', 
                    color='#2E86AB', linewidth=2.5, alpha=0.8)
        ax_main.plot(x_axis, pred_seq, label='Prediction', 
                    color='#A23B72', linewidth=2.5, alpha=0.8)
        
        error = np.abs(true_seq - pred_seq)
        ax_main.fill_between(x_axis, pred_seq - error, pred_seq + error, 
                            alpha=0.2, color='#A23B72')
        
        mae = np.mean(error)
        ax_main.set_title(f'Sample {idx} (MAE: {mae:.4f})', 
                         fontsize=11, fontweight='bold')
        ax_main.legend(loc='upper right', framealpha=0.9)
        ax_main.set_ylabel('Value', fontsize=10)
        ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Derivative plot
        ax_deriv = fig.add_subplot(gs[row*2+1, col])
        true_diff = np.diff(true_seq)
        pred_diff = np.diff(pred_seq)
        x_axis_diff = np.arange(len(true_diff))
        
        ax_deriv.plot(x_axis_diff, true_diff, label='True Δ', 
                     color='#2E86AB', linewidth=2, alpha=0.8)
        ax_deriv.plot(x_axis_diff, pred_diff, label='Pred Δ', 
                     color='#A23B72', linewidth=2, alpha=0.8)
        ax_deriv.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Highlight large jumps
        jump_threshold = np.std(true_diff)
        large_jumps = np.abs(true_diff) > jump_threshold
        if large_jumps.any():
            ax_deriv.scatter(x_axis_diff[large_jumps], true_diff[large_jumps], 
                           color='red', s=50, alpha=0.6, label='Large Jumps', zorder=5)
        
        ax_deriv.set_title('First Differences (Jumps)', fontsize=10)
        ax_deriv.legend(loc='upper right', framealpha=0.9, fontsize=8)
        ax_deriv.set_xlabel('Time Step', fontsize=10)
        ax_deriv.set_ylabel('Δ Value', fontsize=10)
        ax_deriv.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Training
# ============================================================================
def train_card_model(
    model: Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: CARDConfig,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_path: Optional[str] = None,
    patience: int = 10,
    loss_type: str = 'uniform'
) -> Model:
    """Train CARD model with teacher forcing schedule"""
    model = model.to(device)
    model.max_epochs = epochs
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n===== TRAINING CARD MODEL (ENHANCED) =====")
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"LR: {lr}, Epochs: {epochs}, Patience: {patience}")
    print(f"Loss: {loss_type}, AR: {config.autoregressive}")
    if config.autoregressive:
        print(f"AR steps: {config.ar_steps}, TF: {config.teacher_forcing_start}→{config.teacher_forcing_end}")
    print(f"EMA: {config.use_ema}, Hybrid Head: {config.use_hybrid_head}")
    print(f"Bypass over-channel: {config.enc_in <= 2}")
    
    for epoch in range(epochs):
        model.set_epoch(epoch, epochs)
        tf_prob = model.get_teacher_forcing_prob()
        
        # Training
        model.train()
        train_loss = 0
        train_huber = 0
        train_slope = 0
        n_train_batches = 0
        train_nan_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            target = y[:, -config.pred_len:, :]
            
            if epoch == 0 and batch_idx == 0:
                print(f"\nFirst batch:")
                print(f"  Input: {x_enc.shape} [{x_enc.min():.4f}, {x_enc.max():.4f}]")
                print(f"  Pred: {pred.shape} [{pred.min():.4f}, {pred.max():.4f}] std={pred.std():.4f}")
                print(f"  Target: {target.shape} [{target.min():.4f}, {target.max():.4f}]")
            
            loss, huber_loss, slope_loss = trend_aware_loss(
                pred, target, config, horizon_weight=loss_type
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                train_nan_batches += 1
                if train_nan_batches <= 3:
                    print(f"WARNING: NaN loss at epoch {epoch}, batch {batch_idx}")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_huber += huber_loss.item()
            train_slope += slope_loss.item()
            n_train_batches += 1
        
        if n_train_batches == 0:
            print(f"ERROR: All batches NaN in epoch {epoch}!")
            break
            
        avg_train_loss = train_loss / n_train_batches
        avg_train_huber = train_huber / n_train_batches
        avg_train_slope = train_slope / n_train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
                x_enc = x_enc.to(device)
                x_mark_enc = x_mark_enc.to(device)
                x_dec = x_dec.to(device)
                x_mark_dec = x_mark_dec.to(device)
                y = y.to(device)
                
                pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                target = y[:, -config.pred_len:, :]
                loss, _, _ = trend_aware_loss(pred, target, config, horizon_weight=loss_type)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    n_val_batches += 1
        
        if n_val_batches == 0:
            print(f"ERROR: All validation batches NaN!")
            break
            
        avg_val_loss = val_loss / n_val_batches
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {avg_train_loss:.6f} (H:{avg_train_huber:.4f} S:{avg_train_slope:.4f}) | "
              f"Val: {avg_val_loss:.6f} | TF: {tf_prob:.3f}", end="")
        
        if train_nan_batches > 0:
            print(f" | NaN: {train_nan_batches}")
        else:
            print()
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config
                }, save_path)
                print(f"  → Best saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best
    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model (epoch {checkpoint['epoch']+1})")
    
    return model


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_card_model(
    model: Model,
    test_loader: DataLoader,
    config: CARDConfig,
    outdir: str,
    device: str = 'cuda',
    train_data: Optional[np.ndarray] = None
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate CARD model with comprehensive metrics"""
    model.eval()
    model = model.to(device)
    
    y_true_list = []
    y_pred_list = []
    
    print("\n===== EVALUATING =====")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
            x_enc = x_enc.to(device)
            x_mark_enc = x_mark_enc.to(device)
            x_dec = x_dec.to(device)
            x_mark_dec = x_mark_dec.to(device)
            y = y.to(device)
            
            pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            y_true = y[:, -config.pred_len:, :].cpu().numpy()
            y_pred_list.append(pred.cpu().numpy())
            y_true_list.append(y_true)
            
            if batch_idx == 0:
                print(f"Sample pred: [{pred.min():.4f}, {pred.max():.4f}]")
                print(f"Sample true: [{y_true.min():.4f}, {y_true.max():.4f}]")
    
    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)
    
    # Metrics
    metrics = calculate_metrics(y_true_all, y_pred_all, train_data)
    per_horizon_metrics = calculate_per_horizon_metrics(y_true_all, y_pred_all)
    
    # Save
    os.makedirs(outdir, exist_ok=True)
    
    all_metrics = {
        **metrics,
        "per_horizon": per_horizon_metrics
    }
    
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Plots
    n_channels = y_true_all.shape[2]
    for ch in range(n_channels):
        plot_predictions_with_derivatives(
            y_true_all, y_pred_all, channel_idx=ch, n_samples=4,
            save_path=os.path.join(outdir, f'predictions_ch{ch}.png'),
            title=f"CARD Predictions - Channel {ch}"
        )
    
    np.save(os.path.join(outdir, 'y_true.npy'), y_true_all)
    np.save(os.path.join(outdir, 'y_pred.npy'), y_pred_all)
    
    print(f"\n===== RESULTS =====")
    print(f"Samples: {len(y_true_all)}")
    for key, value in metrics.items():
        if isinstance(value, dict):
            continue
        print(f"{key}: {value:.6f}")
    
    print(f"\nPer-Horizon:")
    for horizon, h_metrics in per_horizon_metrics.items():
        print(f"  {horizon}: DA={h_metrics['DirectionAccuracy']:.2f}% MAE={h_metrics['MAE']:.4f}")
    
    print(f"\nSaved to: {outdir}")
    
    return metrics, y_true_all, y_pred_all


# ============================================================================
# Data Loading
# ============================================================================
def load_and_prepare_data_for_card(
    csv_path: str,
    date_col: str = "date",
    value_cols: Optional[list] = None,
    seq_len: int = 96,
    pred_len: int = 96,
    label_len: int = 48,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, CARDConfig, GlobalNormalizer, np.ndarray]:
    """Load and prepare data for CARD training"""
    print(f"Loading from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill()
    
    if df.isnull().any().any():
        print("WARNING: NaN values present!")
        df = df.fillna(0)
    
    # Dates
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        dates = pd.DatetimeIndex(df[date_col])
    else:
        dates = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
    
    # Columns
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if date_col in value_cols:
            value_cols.remove(date_col)
    
    data = df[value_cols].values
    n_channels = len(value_cols)
    
    print(f"\n===== DATA STATS =====")
    print(f"Shape: {data.shape}")
    print(f"Channels: {n_channels} {value_cols}")
    print(f"Range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"Mean/Std: {data.mean():.4f} / {data.std():.4f}")
    
    # Split
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    train_dates = dates[:train_size]
    val_dates = dates[train_size:train_size + val_size]
    test_dates = dates[train_size + val_size:]
    
    print(f"Splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Normalizer
    normalizer = GlobalNormalizer(normalize_deltas=True)
    normalizer.fit(train_data)
    print(f"\nNormalizer: mean={normalizer.mean.flatten()}, std={normalizer.std.flatten()}")
    
    # Datasets
    train_dataset = TimeSeriesDataset(train_data, train_dates, seq_len, label_len, pred_len, normalizer=normalizer)
    val_dataset = TimeSeriesDataset(val_data, val_dates, seq_len, label_len, pred_len, normalizer=normalizer)
    test_dataset = TimeSeriesDataset(test_data, test_dates, seq_len, label_len, pred_len, normalizer=normalizer)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Config with optimal defaults for gas prices
    config = CARDConfig(
        task_name='long_term_forecast',
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        enc_in=n_channels,
        dec_in=n_channels,
        c_out=n_channels,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.1,
        patch_len=16,
        stride=4,  # Overlapping patches
        alpha=0.2,
        use_ema=False,  # Disabled for price data
        ema_first_layer_only=True,
        autoregressive=True,
        ar_steps=24,
        teacher_forcing_start=0.5,
        teacher_forcing_end=0.0,
        global_norm=True,
        normalize_deltas=True,
        use_hybrid_head=True,
        loss_huber_weight=0.7,
        loss_slope_weight=0.3,
        huber_delta=1.0
    )
    
    return train_loader, val_loader, test_loader, config, normalizer, train_data