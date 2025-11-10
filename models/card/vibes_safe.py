"""
Patched Enhanced CARD Model (NaN-safe evaluation + robust CSV loading)

This is a drop-in replacement for models/card/vibes.py with minimal changes:
- Robust CSV loading: coerce numerics and drop rows with NaNs in value columns.
- Safer evaluation: np.nan_to_num applied to predictions and ground truth before concatenation;
  compute_metrics also sanitizes inputs to prevent NaN metrics.
- Optional clipping in evaluation to keep extreme values in a reasonable numeric range.

Everything else remains the same as the user's provided implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math
import os

# ---------------------------------------------------------------------------
# Utilities from original file (unchanged unless noted)
# ---------------------------------------------------------------------------

def extract_trend_component(x, kernel_size=16):
    batch, seq_len, channels = x.shape
    kernel_size = min(kernel_size, seq_len)
    if kernel_size < 1:
        kernel_size = 1
    x_t = x.transpose(1, 2)
    if kernel_size >= seq_len or kernel_size == 1:
        trend = x_t.mean(dim=2, keepdim=True).expand_as(x_t)
    else:
        padding = kernel_size // 2
        trend = F.avg_pool1d(
            F.pad(x_t, (padding, padding), mode='reflect'),
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )
        if trend.size(2) > seq_len:
            trend = trend[:, :, :seq_len]
        elif trend.size(2) < seq_len:
            trend = F.pad(trend, (0, seq_len - trend.size(2)), mode='replicate')
    return trend.transpose(1, 2)


def project_trend_linearly(trend_history, pred_len):
    batch, seq_len, channels = trend_history.shape
    device = trend_history.device
    dtype = trend_history.dtype
    trend_history = torch.clamp(trend_history, -1e6, 1e6)
    t_hist = torch.arange(seq_len, dtype=dtype, device=device).view(1, -1, 1)
    t_mean = t_hist.mean()
    t_centered = t_hist - t_mean
    x_mean = trend_history.mean(dim=1, keepdim=True)
    x_centered = trend_history - x_mean
    cov = (t_centered * x_centered).sum(dim=1, keepdim=True)
    var_t = (t_centered ** 2).sum() + 1e-8
    slope = torch.clamp(cov / var_t, -10.0, 10.0)
    intercept = x_mean
    t_future = torch.arange(seq_len, seq_len + pred_len, dtype=dtype, device=device).view(1, -1, 1) - t_mean
    trend_future = intercept + slope * t_future
    trend_future = torch.clamp(trend_future, -1e6, 1e6)
    return trend_future


def create_reduced_causal_mask(seq_len, lookahead=4, device='cpu'):
    mask = torch.ones(seq_len, seq_len, device=device)
    for i in range(seq_len):
        if i + lookahead + 1 < seq_len:
            mask[i, i + lookahead + 1:] = 0
    return mask


def compute_trend_aware_loss(pred, true, config):
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    if torch.isnan(true).any() or torch.isinf(true).any():
        true = torch.nan_to_num(true, nan=0.0, posinf=1e6, neginf=-1e6)
    mse_loss = F.mse_loss(pred, true)
    trend_loss = torch.tensor(0.0, device=pred.device)
    if config.use_trend_decomposition and pred.size(1) >= config.trend_smoothing_kernel:
        try:
            pred_trend = extract_trend_component(pred, config.trend_smoothing_kernel)
            true_trend = extract_trend_component(true, config.trend_smoothing_kernel)
            trend_loss = F.mse_loss(pred_trend, true_trend)
        except Exception:
            pass
    direction_loss = torch.tensor(0.0, device=pred.device)
    if pred.size(1) > 1:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        direction_loss = (torch.sign(pred_diff) != torch.sign(true_diff)).float().mean()
    momentum_loss = torch.tensor(0.0, device=pred.device)
    if pred.size(1) > 2:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        pred_diff2 = pred_diff[:, 1:] - pred_diff[:, :-1]
        true_diff2 = true_diff[:, 1:] - true_diff[:, :-1]
        momentum_loss = F.mse_loss(pred_diff2, true_diff2)
    total_loss = mse_loss
    if not torch.isnan(trend_loss) and not torch.isinf(trend_loss):
        total_loss = total_loss + config.loss_trend_mse_weight * trend_loss
    if not torch.isnan(direction_loss) and not torch.isinf(direction_loss):
        total_loss = total_loss + config.loss_direction_weight * direction_loss
    if not torch.isnan(momentum_loss) and not torch.isinf(momentum_loss):
        total_loss = total_loss + config.loss_momentum_weight * momentum_loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = mse_loss
    return total_loss, {
        'mse': float(mse_loss) if not torch.isnan(mse_loss) else 0.0,
        'trend': float(trend_loss) if isinstance(trend_loss, torch.Tensor) and not torch.isnan(trend_loss) else 0.0,
        'direction': float(direction_loss) if not torch.isnan(direction_loss) else 0.0,
        'momentum': float(momentum_loss) if isinstance(momentum_loss, torch.Tensor) and not torch.isnan(momentum_loss) else 0.0
    }


@dataclass
class Config:
    seq_len: int = 96
    pred_len: int = 96
    label_len: int = 48
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    use_ema: bool = False
    use_hybrid_head: bool = True
    normalize_deltas: bool = True
    loss_huber_weight: float = 0.5
    loss_slope_weight: float = 0.3
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    autoregressive: bool = False
    ar_steps: int = 4
    use_trend_decomposition: bool = True
    trend_smoothing_kernel: int = 16
    trend_projection_strength: float = 0.7
    reduce_causal_bias: bool = True
    causal_lookahead: int = 4
    attention_temperature: float = 1.0
    use_trend_loss: bool = True
    loss_trend_mse_weight: float = 2.0
    loss_direction_weight: float = 1.5
    loss_momentum_weight: float = 0.5


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_len, stride):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        n_patches = patches.shape[1]
        patches = patches.reshape(batch_size, n_patches, self.patch_len * n_vars)
        return patches


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        Q = self.W_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            if mask.dim() == 2:
                if mask.size(0) == q_len and mask.size(1) == k_len:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                else:
                    mask = None
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x, mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        attn_out, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
        return x


class Model(nn.Module):
    def __init__(self, config, normalizer=None):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.patching = Patching(config.patch_len, config.stride)
        self.enc_embedding = DataEmbedding(config.patch_len * config.enc_in, config.d_model, config.dropout)
        self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.dropout)
        self.encoder = TransformerEncoder([
            TransformerEncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.e_layers)
        ])
        self.decoder = TransformerDecoder([
            TransformerDecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.d_layers)
        ])
        self.projection = nn.Linear(config.d_model, config.c_out)
        self.register_buffer('enc_mask', None)
        self.register_buffer('dec_mask', None)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        batch_size = x_enc.size(0)
        device = x_enc.device
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_normalized = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_normalized, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev = torch.clamp(stdev, min=1e-5)
        x_enc_normalized = x_enc_normalized / stdev
        if torch.isnan(x_enc_normalized).any() or torch.isinf(x_enc_normalized).any():
            x_enc_normalized = torch.nan_to_num(x_enc_normalized, nan=0.0, posinf=10.0, neginf=-10.0)
        if self.config.use_trend_decomposition:
            trend_enc = extract_trend_component(x_enc_normalized, self.config.trend_smoothing_kernel)
            residual_enc = x_enc_normalized - trend_enc
            trend_patches = self.patching(trend_enc)
            residual_patches = self.patching(residual_enc)
            trend_emb = self.enc_embedding(trend_patches)
            residual_emb = self.enc_embedding(residual_patches)
        else:
            residual_patches = self.patching(x_enc_normalized)
            residual_emb = self.enc_embedding(residual_patches)
            trend_emb = None
        n_patches = residual_emb.size(1)
        if self.config.reduce_causal_bias and self.training:
            enc_self_attn_mask = create_reduced_causal_mask(n_patches, self.config.causal_lookahead, device).unsqueeze(0).unsqueeze(0)
        else:
            enc_self_attn_mask = None
        residual_enc_out, attns = self.encoder(residual_emb, enc_self_attn_mask)
        if self.config.use_trend_decomposition and trend_emb is not None:
            trend_enc_out, _ = self.encoder(trend_emb, mask=None)
        else:
            trend_enc_out = None
        if x_dec is None:
            dec_inp = torch.zeros((batch_size, self.config.label_len + self.config.pred_len, self.config.dec_in), device=device)
            if self.config.label_len > 0:
                dec_inp[:, :self.config.label_len, :] = x_enc_normalized[:, -self.config.label_len:, :self.config.dec_in]
        else:
            dec_inp = x_dec
        dec_emb = self.dec_embedding(dec_inp)
        dec_out = self.decoder(dec_emb, residual_enc_out, None, None)
        if self.config.use_trend_decomposition and trend_emb is not None:
            trend_future = project_trend_linearly(trend_enc, self.config.pred_len)
            dec_pred = dec_out[:, -self.config.pred_len:, :]
            residual_pred = self.projection(dec_pred)
            residual_pred_denorm = residual_pred * stdev[:, -1:, :] + means[:, -1:, :]
            trend_future_denorm = trend_future * stdev[:, -1:, :] + means[:, -1:, :]
            if torch.isnan(residual_pred_denorm).any() or torch.isinf(residual_pred_denorm).any():
                residual_pred_denorm = torch.nan_to_num(residual_pred_denorm, nan=0.0, posinf=1e6, neginf=-1e6)
            if torch.isnan(trend_future_denorm).any() or torch.isinf(trend_future_denorm).any():
                trend_future_denorm = torch.nan_to_num(trend_future_denorm, nan=0.0, posinf=1e6, neginf=-1e6)
            alpha = self.config.trend_projection_strength
            output = alpha * trend_future_denorm + (1 - alpha) * residual_pred_denorm
        else:
            dec_pred = dec_out[:, -self.config.pred_len:, :]
            output = self.projection(dec_pred)
            output = output * stdev[:, -1:, :] + means[:, -1:, :]
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        return output


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, label_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        x_enc = self.data[s_begin:s_end]
        x_dec = self.data[r_begin:r_end]
        y = self.data[s_end:r_end]
        x_mark_enc = np.zeros((self.seq_len, 4))
        x_mark_dec = np.zeros((self.label_len + self.pred_len, 4))
        return (
            torch.FloatTensor(x_enc),
            torch.FloatTensor(x_mark_enc),
            torch.FloatTensor(x_dec),
            torch.FloatTensor(x_mark_dec),
            torch.FloatTensor(y)
        )


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, data):
        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return data * self.std + self.mean


def load_and_prepare_data_for_card(
    csv_path,
    date_col='date',
    value_cols=None,
    seq_len=96,
    pred_len=96,
    label_len=48,
    train_ratio=0.7,
    val_ratio=0.1,
    batch_size=32
):
    # Robust CSV load: coerce numerics and drop NaNs in value columns
    df = pd.read_csv(csv_path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # Determine value columns if not specified
    if value_cols is None:
        value_cols = [col for col in df.columns if col != date_col]
    # Coerce to numeric
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Drop any rows with NaNs in value columns
    df = df.dropna(subset=value_cols)
    # Sort by date if present
    if date_col in df.columns:
        df = df.sort_values(date_col)
    data = df[value_cols].values.astype(np.float32)

    n = len(data)
    if n < (seq_len + pred_len + 1):
        raise ValueError(f"Not enough rows after cleaning: {n} < required {seq_len + pred_len + 1}")
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    normalizer = StandardScaler()
    normalizer.fit(train_data)

    train_data_norm = normalizer.transform(train_data)
    val_data_norm = normalizer.transform(val_data)
    test_data_norm = normalizer.transform(test_data)

    train_dataset = TimeSeriesDataset(train_data_norm, seq_len, pred_len, label_len)
    val_dataset = TimeSeriesDataset(val_data_norm, seq_len, pred_len, label_len)
    test_dataset = TimeSeriesDataset(test_data_norm, seq_len, pred_len, label_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    config = Config(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=label_len,
        enc_in=len(value_cols),
        dec_in=len(value_cols),
        c_out=len(value_cols),
        use_trend_decomposition=True,
        trend_smoothing_kernel=max(16, seq_len // 6),
        trend_projection_strength=0.7,
        reduce_causal_bias=True,
        causal_lookahead=4,
        use_trend_loss=True,
        loss_trend_mse_weight=2.0,
        loss_direction_weight=1.5,
        loss_momentum_weight=0.5
    )
    return train_loader, val_loader, test_loader, config, normalizer, train_data


def train_card_model(
    model,
    train_loader,
    val_loader,
    config,
    epochs=50,
    lr=1e-3,
    device='cpu',
    save_path='checkpoints/card_model.pt',
    patience=15,
    loss_type='uniform'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0
    print(f"\nTraining on {device}")
    print(f"Trend decomposition: {config.use_trend_decomposition}")
    print(f"Trend-aware loss: {config.use_trend_loss}")
    print(f"Reduced causal bias: {config.reduce_causal_bias}")
    nan_count = 0
    max_nan_tolerance = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_losses_dict = {'mse': 0, 'trend': 0, 'direction': 0, 'momentum': 0}
        for i, batch in enumerate(train_loader):
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if torch.isnan(outputs).any():
                nan_count += 1
                if nan_count >= max_nan_tolerance:
                    config.use_trend_decomposition = False
                    config.use_trend_loss = False
                    nan_count = 0
                outputs = torch.nan_to_num(outputs, nan=0.0)
            if config.use_trend_loss:
                loss, loss_components = compute_trend_aware_loss(outputs[:, -config.pred_len:, :], y[:, -config.pred_len:, :], config)
                for key in train_losses_dict:
                    train_losses_dict[key] += loss_components[key]
            else:
                loss = F.mse_loss(outputs[:, -config.pred_len:, :], y[:, -config.pred_len:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            if (i + 1) % 20 == 0:
                avg_loss = train_loss / (i + 1)
                if config.use_trend_loss:
                    print(f"  Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {avg_loss:.6f} | Trend: {train_losses_dict['trend']/(i+1):.4f} | Dir: {train_losses_dict['direction']/(i+1):.4f}")
                else:
                    print(f"  Epoch {epoch+1}/{epochs} | Batch {i+1}/{len(train_loader)} | Loss: {avg_loss:.6f}")
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        val_losses_dict = {'mse': 0, 'trend': 0, 'direction': 0, 'momentum': 0}
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y = [b.to(device) for b in batch]
                outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                if config.use_trend_loss:
                    loss, loss_components = compute_trend_aware_loss(outputs[:, -config.pred_len:, :], y[:, -config.pred_len:, :], config)
                    for key in val_losses_dict:
                        val_losses_dict[key] += loss_components[key]
                else:
                    loss = F.mse_loss(outputs[:, -config.pred_len:, :], y[:, -config.pred_len:, :])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        if config.use_trend_loss:
            print(f"  Val Components: MSE={val_losses_dict['mse']/len(val_loader):.4f}, Trend={val_losses_dict['trend']/len(val_loader):.4f}, Direction={val_losses_dict['direction']/len(val_loader):.4f}, Momentum={val_losses_dict['momentum']/len(val_loader):.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': avg_val_loss, 'config': config}, save_path)
            print(f"  ✓ New best model saved (val_loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        print("-" * 70)
    try:
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ Training completed. Best model from epoch {checkpoint['epoch']+1} loaded.")
    except Exception as e:
        print(f"\n⚠ Could not load best checkpoint: {e}")
        print("Using final model state instead.")
    return model


def evaluate_card_model(model, test_loader, config, outdir='results/', device='cpu', train_data=None):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_trues = []
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in test_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec, y = [b.to(device) for b in batch]
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            pred = outputs[:, -config.pred_len:, :].detach().cpu().numpy()
            true = y[:, -config.pred_len:, :].detach().cpu().numpy()
            # Sanitize to avoid NaN metrics
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            true = np.nan_to_num(true, nan=0.0, posinf=1e6, neginf=-1e6)
            # Optional: clip to a reasonable range to prevent overflow in metrics
            pred = np.clip(pred, -1e8, 1e8)
            true = np.clip(true, -1e8, 1e8)
            all_preds.append(pred)
            all_trues.append(true)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    metrics = compute_metrics(y_true, y_pred, train_data)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'predictions.npy'), y_pred)
    np.save(os.path.join(outdir, 'ground_truth.npy'), y_true)
    metrics_serializable: Dict[str, float] = {}
    for key, value in metrics.items():
        metrics_serializable[key] = value if isinstance(value, dict) else float(value)
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        import json as _json
        _json.dump(metrics_serializable, f, indent=2)
    print(f"\n✓ Results saved to {outdir}")
    return metrics, y_true, y_pred


def compute_metrics(y_true, y_pred, train_data=None):
    # Sanitize arrays
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e6, neginf=-1e6)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    y_true = np.clip(y_true, -1e8, 1e8)
    y_pred = np.clip(y_pred, -1e8, 1e8)

    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    mse = float(np.mean((y_true_flat - y_pred_flat) ** 2))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    rmse = float(np.sqrt(mse))
    smape = float(200 * np.mean(np.abs(y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + np.abs(y_pred_flat) + 1e-8)))

    if train_data is not None:
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=1e6, neginf=-1e6)
        naive_error = np.mean(np.abs(np.diff(train_data, axis=0)))
        mase = float(mae / (naive_error + 1e-8))
    else:
        mase = 0.0

    y_true_diff = y_true[:, 1:, :] - y_true[:, :-1, :]
    y_pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    direction_correct = (np.sign(y_true_diff) == np.sign(y_pred_diff)).astype(float)
    direction_accuracy = float(100 * np.mean(direction_correct))

    per_horizon = {}
    horizons = [1, 4, 8, 16, 32, 64, 96] if y_true.shape[1] >= 96 else [1, 4, 8, 16]
    for h in horizons:
        if h <= y_true.shape[1]:
            h_idx = h - 1
            y_true_h = y_true[:, h_idx, :]
            y_pred_h = y_pred[:, h_idx, :]
            per_horizon[f'horizon_{h}'] = {
                'MAE': float(np.mean(np.abs(y_true_h - y_pred_h))),
                'RMSE': float(np.sqrt(np.mean((y_true_h - y_pred_h) ** 2))),
                'DirectionAccuracy': float(100 * np.mean(
                    (np.sign(y_true_h - y_true[:, max(0, h_idx-1), :]) == 
                     np.sign(y_pred_h - y_pred[:, max(0, h_idx-1), :])).astype(float)
                ))
            }

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'sMAPE': smape,
        'MASE': mase,
        'DirectionAccuracy': direction_accuracy,
        'per_horizon': per_horizon
    }


def quick_train_and_evaluate(
    csv_path,
    date_col='date',
    value_cols=None,
    seq_len=96,
    pred_len=96,
    epochs=50,
    batch_size=32,
    device='auto',
    save_dir='results_card/'
):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("="*70)
    print("CARD MODEL - QUICK TRAIN & EVALUATE")
    print("="*70)
    print(f"CSV: {csv_path}")
    print(f"Device: {device}")
    print(f"Sequence length: {seq_len}")
    print(f"Prediction length: {pred_len}")
    print(f"Epochs: {epochs}")
    print("\nTrend Capture Features:")
    print("  ✓ Trend decomposition enabled")
    print("  ✓ Linear trend projection")
    print("  ✓ Reduced causal masking")
    print("  ✓ Trend-aware loss function")
    print("="*70)
    print("\nLoading data...")
    train_loader, val_loader, test_loader, config, normalizer, train_data = load_and_prepare_data_for_card(
        csv_path=csv_path,
        date_col=date_col,
        value_cols=value_cols,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
    print("\nCreating model...")
    model = Model(config, normalizer=normalizer)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {n_params:,} parameters")
    print("\nStarting training...")
    os.makedirs('checkpoints', exist_ok=True)
    trained_model = train_card_model(
        model,
        train_loader,
        val_loader,
        config,
        epochs=epochs,
        device=device,
        save_path='checkpoints/card_trend_model.pt'
    )
    print("\nEvaluating on test set...")
    os.makedirs(save_dir, exist_ok=True)
    metrics, y_true, y_pred = evaluate_card_model(
        trained_model,
        test_loader,
        config,
        outdir=save_dir,
        device=device,
        train_data=train_data
    )
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"MSE:  {metrics['MSE']:.6f}")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"sMAPE: {metrics['sMAPE']:.2f}%")
    print(f"MASE: {metrics['MASE']:.4f}")
    print(f"Direction Accuracy: {metrics['DirectionAccuracy']:.2f}%")
    if 'per_horizon' in metrics:
        print("\nPer-Horizon Metrics:")
        for horizon, h_metrics in metrics['per_horizon'].items():
            print(f"  {horizon}: DA={h_metrics['DirectionAccuracy']:.2f}% MAE={h_metrics['MAE']:.4f}")
    print("="*70)
    return metrics


def get_config_preset(preset='default', seq_len=96, pred_len=96, enc_in=7):
    base_config = Config(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dec_in=enc_in, c_out=enc_in)
    if preset == 'default':
        base_config.use_trend_decomposition = True
        base_config.trend_smoothing_kernel = max(16, seq_len // 6)
        base_config.trend_projection_strength = 0.7
        base_config.reduce_causal_bias = True
        base_config.causal_lookahead = 4
        base_config.loss_trend_mse_weight = 2.0
        base_config.loss_direction_weight = 1.5
    elif preset == 'aggressive_trend':
        base_config.use_trend_decomposition = True
        base_config.trend_smoothing_kernel = max(32, seq_len // 3)
        base_config.trend_projection_strength = 0.85
        base_config.reduce_causal_bias = True
        base_config.causal_lookahead = 8
        base_config.loss_trend_mse_weight = 3.0
        base_config.loss_direction_weight = 2.0
        base_config.loss_momentum_weight = 1.0
    elif preset == 'conservative':
        base_config.use_trend_decomposition = False
        base_config.reduce_causal_bias = False
        base_config.use_trend_loss = False
    elif preset == 'fast':
        base_config.d_model = 256
        base_config.n_heads = 4
        base_config.e_layers = 1
        base_config.d_layers = 1
        base_config.d_ff = 1024
        base_config.use_trend_decomposition = True
        base_config.trend_smoothing_kernel = 16
    return base_config


if __name__ == "__main__":
    print(__doc__)
    print("\nPatched module is ready. Import Model, load_and_prepare_data_for_card, and evaluate_card_model as usual.")
