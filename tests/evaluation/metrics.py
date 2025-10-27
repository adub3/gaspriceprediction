# tests/evaluation/metrics.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from sklearn.metrics import (
    roc_curve, roc_auc_score, brier_score_loss, log_loss as sk_logloss
)

# --- keep your original simple metrics so old code still works ---
def brier_score(p_hat: np.ndarray, y: np.ndarray) -> float:
    return float(brier_score_loss(y, p_hat))

def logloss(p_hat: np.ndarray, y: np.ndarray) -> float:
    return float(sk_logloss(y, np.clip(p_hat, 1e-15, 1 - 1e-15)))

def rocauc(p_hat: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, p_hat))

# ======================================================
#                 STAT HELPERS
# ======================================================
def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2*n)) / denom
    half = z * np.sqrt((p*(1-p) + (z**2)/(4*n)) / n) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi

def make_bin_edges(p_hat: np.ndarray, method: str, n_bins: int) -> np.ndarray:
    """Create explicit bin edges in [0,1] with tie-safe quantiles."""
    p = np.asarray(p_hat)
    if method == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins+1)))
        # fallback if ties collapsed too much
        if len(edges) < max(6, n_bins//2):
            edges = np.linspace(0, 1, n_bins+1)
    elif method == "uniform":
        edges = np.linspace(0, 1, n_bins+1)
    else:
        raise ValueError("bin_method must be 'quantile' or 'uniform'")
    edges[0], edges[-1] = 0.0, 1.0
    return edges

def calibration_table_from_edges(
    p_hat: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    min_bin_n: int = 50
) -> pd.DataFrame:
    """Return calibration table with Wilson CIs; drop bins with n < min_bin_n."""
    p = np.asarray(p_hat)
    yy = np.asarray(y).astype(int)

    idx = np.digitize(p, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges)-2)

    rows = []
    for b in range(len(edges)-1):
        m = (idx == b)
        n = int(m.sum())
        if n < min_bin_n:
            continue
        ph_mean = float(p[m].mean()) if n else 0.0
        k = int(yy[m].sum())
        freq, lo, hi = wilson_interval(k, n)
        se = np.sqrt(freq*(1-freq)/n) if 0 < freq < 1 else 0.0
        rows.append(dict(p_hat_mean=ph_mean, freq=freq, n=n,
                         se=se, lower=lo, upper=hi))
    return pd.DataFrame(rows)

def ece_from_table(tbl: pd.DataFrame) -> float:
    if len(tbl) == 0:
        return float("nan")
    w = tbl["n"].to_numpy()
    return float(np.average(np.abs(tbl["freq"] - tbl["p_hat_mean"]), weights=w))

def compute_ece(
    p_hat: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    min_bin_n: int = 50
) -> Tuple[float, pd.DataFrame]:
    tbl = calibration_table_from_edges(p_hat, y, edges, min_bin_n=min_bin_n)
    return ece_from_table(tbl), tbl

def bootstrap_metric(
    p_hat: np.ndarray,
    y: np.ndarray,
    func,
    n_boot: int = 500,
    seed: int = 0
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Generic bootstrap (i.i.d.). Use a block bootstrap if temporal dependence matters."""
    rng = np.random.default_rng(seed)
    n = len(y); vals = []
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        vals.append(func(p_hat[ii], y[ii]))
    vals = np.asarray(vals)
    mean = float(np.nanmean(vals))
    lo = float(np.nanpercentile(vals, 2.5))
    hi = float(np.nanpercentile(vals, 97.5))
    return mean, (lo, hi), vals

def bootstrap_ece(
    p_hat: np.ndarray, y: np.ndarray, edges: np.ndarray,
    min_bin_n: int = 50, n_boot: int = 500, seed: int = 0
) -> Tuple[float, Tuple[float, float], np.ndarray]:
    def _ece_func(ph, yy):
        tbl = calibration_table_from_edges(ph, yy, edges, min_bin_n=min_bin_n)
        return ece_from_table(tbl)
    return bootstrap_metric(p_hat, y, _ece_func, n_boot=n_boot, seed=seed)

def brier_decomposition(p_hat: np.ndarray, y: np.ndarray, edges: np.ndarray) -> Dict[str, float]:
    """Murphy decomposition: BS = reliability - resolution + uncertainty."""
    yy = np.asarray(y).astype(int)
    pp = np.asarray(p_hat)
    BS = brier_score_loss(yy, pp)

    # Use min_bin_n=1 to include all bins for decomposition
    tbl = calibration_table_from_edges(pp, yy, edges, min_bin_n=1)
    if len(tbl) == 0:
        ybar = yy.mean()
        return dict(brier=BS, reliability=np.nan, resolution=np.nan, uncertainty=ybar*(1-ybar))

    N = len(yy)
    ybar = yy.mean()
    reliability = float(np.sum((tbl["n"]/N) * (tbl["freq"] - tbl["p_hat_mean"])**2))
    resolution  = float(np.sum((tbl["n"]/N) * (tbl["freq"] - ybar)**2))
    uncertainty = float(ybar*(1 - ybar))
    return dict(brier=BS, reliability=reliability, resolution=resolution, uncertainty=uncertainty)

def brier_skill_score(p_hat: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Skill vs climatology (always predict base rate)."""
    bs_model = brier_score_loss(y, p_hat)
    p_clim = np.full_like(p_hat, fill_value=np.mean(y), dtype=float)
    bs_clim = brier_score_loss(y, p_clim)
    bss = 1.0 - bs_model / bs_clim
    return bss, bs_model, bs_clim

def safe_logloss(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Older sklearn versions don't support eps kwarg; clip explicitly."""
    return float(sk_logloss(y_true, np.clip(p_hat, 1e-15, 1 - 1e-15)))

# ======================================================
#                 PLOTTING HELPERS
# ======================================================
def plot_calibration(calib_tbl: pd.DataFrame, ece: float, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="tab:blue")
    if len(calib_tbl):
        x = calib_tbl["p_hat_mean"].to_numpy()
        y = calib_tbl["freq"].to_numpy()
        lo = calib_tbl["lower"].to_numpy()
        hi = calib_tbl["upper"].to_numpy()
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=2, color="tab:orange")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (ECE = {ece:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_sharpness_hist(p_hat: np.ndarray, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_hat, bins=30, edgecolor="black")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title("Sharpness (distribution of predictions)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_roc_curve(y: np.ndarray, p_hat: np.ndarray, auc: float, out_png: str) -> None:
    fpr, tpr, _ = roc_curve(y, p_hat)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.plot(fpr, tpr, linewidth=2, color="tab:orange")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {auc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_brier_decomp(bdec: Dict[str, float], out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    parts = ["reliability", "resolution", "uncertainty"]
    vals = [bdec.get("reliability", np.nan),
            bdec.get("resolution", np.nan),
            bdec.get("uncertainty", np.nan)]
    ax.bar(parts, vals, color="tab:blue")
    ax.axhline(bdec.get("brier", np.nan), linestyle="--", linewidth=1, color="tab:orange")
    ax.set_ylabel("Contribution")
    ax.set_title("Brier Decomposition (Murphy)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_ece_bar(ece: float, ci: Optional[Tuple[float, float]], out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 5))
    ax.bar([0], [ece], width=0.5, color="tab:blue")
    if ci is not None:
        yerr = [[ece - ci[0]], [ci[1] - ece]]
        ax.errorbar([0], [ece], yerr=yerr, fmt="o", capsize=4, color="tab:orange")
    ax.set_xticks([0]); ax.set_xticklabels(["ECE"])
    ymax = max(0.01, (ci[1] if ci else ece) * 1.25)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE (95% CI)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_ece_sensitivity(
    p_hat: np.ndarray, y: np.ndarray, out_png: str, min_bin_n: int = 50,
    bin_counts=(5, 10, 15, 20, 30, 40, 50), methods=("quantile", "uniform")
) -> None:
    rows = []
    for m in methods:
        xs, ys = [], []
        for k in bin_counts:
            edges = make_bin_edges(p_hat, method=m, n_bins=k)
            ece, _ = compute_ece(p_hat, y, edges, min_bin_n=min_bin_n)
            xs.append(k); ys.append(ece)
        rows.append((m, xs, ys))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, xs, ys in rows:
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("# bins"); ax.set_ylabel("ECE")
    ax.set_title("ECE Sensitivity to Binning")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_rolling_ece(
    dates: pd.Index, p_hat: np.ndarray, y: np.ndarray,
    out_png: str, window: int = 250, step: int = 50,
    bin_method: str = "quantile", n_bins: int = 20, min_bin_n: int = 50
) -> None:
    dates = pd.Index(dates)
    xs, ys = [], []
    for start in range(0, len(y) - window + 1, step):
        sl = slice(start, start + window)
        ph = p_hat[sl]; yy = y[sl]
        edges = make_bin_edges(ph, method=bin_method, n_bins=n_bins)
        ece, _ = compute_ece(ph, yy, edges, min_bin_n=min_bin_n)
        xs.append(dates[start + window - 1]); ys.append(ece)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(xs, ys, marker="o", linewidth=1)
    ax.set_title(f"Rolling ECE (window={window}, step={step})")
    ax.set_ylabel("ECE"); ax.set_xlabel("Date")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def plot_bss(bss: float, out_png: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.bar([0], [bss], width=0.5, color="tab:blue")
    ax.set_xticks([0]); ax.set_xticklabels(["BSS"])
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Brier Skill Score (vs. climatology)")
    ax.set_title("Skill Score")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

# ======================================================
#                 ONE-CALL WRAPPER
# ======================================================
def evaluate_and_plot(
    p_hat: np.ndarray,
    y: np.ndarray,
    dates: pd.Index,
    outdir: str,
    *,
    bin_method: str = "quantile",
    n_bins: int = 20,
    min_bin_n: int = 50,
    bootstrap: int = 500
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute metrics, save plots/tables into outdir, and return (summary, calib_tbl)."""

    os.makedirs(outdir, exist_ok=True)

    # core metrics
    brier = brier_score(p_hat, y)
    ll = logloss(p_hat, y)
    auc = rocauc(p_hat, y)

    # calibration + ECE
    edges = make_bin_edges(p_hat, method=bin_method, n_bins=n_bins)
    ece, calib_tbl = compute_ece(p_hat, y, edges, min_bin_n=min_bin_n)

    # extra diagnostics
    bdec = brier_decomposition(p_hat, y, edges)
    bss, bs_model, bs_clim = brier_skill_score(p_hat, y)

    # bootstraps (optional)
    ece_mean = ece_ci = brier_mean = brier_ci = ll_mean = ll_ci = auc_mean = auc_ci = None
    if bootstrap and bootstrap > 0:
        ece_mean, ece_ci, _ = bootstrap_ece(p_hat, y, edges, min_bin_n=min_bin_n, n_boot=bootstrap, seed=0)
        brier_mean, brier_ci, _ = bootstrap_metric(p_hat, y, lambda ph, yy: brier_score_loss(yy, ph), n_boot=bootstrap, seed=1)
        ll_mean, ll_ci, _ = bootstrap_metric(p_hat, y, lambda ph, yy: sk_logloss(yy, np.clip(ph, 1e-15, 1-1e-15)), n_boot=bootstrap, seed=2)
        auc_mean, auc_ci, _ = bootstrap_metric(p_hat, y, lambda ph, yy: roc_auc_score(yy, ph), n_boot=bootstrap, seed=3)

    # save tables
    calib_tbl.to_csv(os.path.join(outdir, "calibration_bins.csv"), index=False)

    # plots
    plot_calibration(calib_tbl, ece, out_png=os.path.join(outdir, "calibration.png"))
    plot_sharpness_hist(p_hat, out_png=os.path.join(outdir, "sharpness_hist.png"))
    plot_roc_curve(y, p_hat, auc=auc, out_png=os.path.join(outdir, "roc_curve.png"))
    plot_brier_decomp(bdec, out_png=os.path.join(outdir, "brier_decomposition.png"))
    plot_ece_bar(ece, ece_ci, out_png=os.path.join(outdir, "ece_bar.png"))
    plot_ece_sensitivity(
        p_hat, y, out_png=os.path.join(outdir, "ece_sensitivity.png"),
        min_bin_n=min_bin_n, bin_counts=(5, 10, 15, 20, 30, 40, 50),
        methods=("quantile", "uniform")
    )
    # rolling ECE (best-effort; skip if dates not datelike)
    try:
        plot_rolling_ece(
            dates, p_hat, y,
            out_png=os.path.join(outdir, "ece_rolling.png"),
            window=min(250, max(50, len(y)//4)),
            step=max(25, len(y)//40),
            bin_method=bin_method, n_bins=n_bins, min_bin_n=min_bin_n
        )
    except Exception:
        pass

    # summary dict
    summary: Dict[str, float] = dict(
        Brier=float(brier), LogLoss=float(ll), ROC_AUC=float(auc),
        ECE=float(ece),
        Brier_reliability=float(bdec.get("reliability", np.nan)),
        Brier_resolution=float(bdec.get("resolution", np.nan)),
        Brier_uncertainty=float(bdec.get("uncertainty", np.nan)),
        Brier_total=float(bdec.get("brier", np.nan)),
        BSS=float(bss),
        Brier_model=float(bs_model),
        Brier_climatology=float(bs_clim),
    )
    # add bootstrap CI if available
    if ece_ci:
        summary.update(
            ECE_CI_low=float(ece_ci[0]), ECE_CI_high=float(ece_ci[1])
        )
    if brier_ci:
        summary.update(
            Brier_CI_low=float(brier_ci[0]), Brier_CI_high=float(brier_ci[1])
        )
    if ll_ci:
        summary.update(
            LogLoss_CI_low=float(ll_ci[0]), LogLoss_CI_high=float(ll_ci[1])
        )
    if auc_ci:
        summary.update(
            ROC_AUC_CI_low=float(auc_ci[0]), ROC_AUC_CI_high=float(auc_ci[1])
        )

    # write metrics.txt
    with open(os.path.join(outdir, "metrics.txt"), "w") as f:
        f.write(f"Brier Score: {brier:.6f}\n")
        f.write(f"LogLoss:     {ll:.6f}\n")
        f.write(f"ROC-AUC:     {auc:.6f}\n")
        f.write(f"ECE:         {ece:.6f}\n")
        if ece_ci:
            f.write("\nBootstrap 95% CIs:\n")
            f.write(f"ECE:     mean={ece_mean:.6f}, CI=({ece_ci[0]:.6f}, {ece_ci[1]:.6f})\n")
        if brier_ci:
            f.write(f"Brier:   mean={brier_mean:.6f}, CI=({brier_ci[0]:.6f}, {brier_ci[1]:.6f})\n")
        if ll_ci:
            f.write(f"LogLoss: mean={ll_mean:.6f}, CI=({ll_ci[0]:.6f}, {ll_ci[1]:.6f})\n")
        if auc_ci:
            f.write(f"AUC:     mean={auc_mean:.6f}, CI=({auc_ci[0]:.6f}, {auc_ci[1]:.6f})\n")
        f.write("\nBrier decomposition (Murphy):\n")
        for k in ("reliability", "resolution", "uncertainty", "brier"):
            f.write(f"{k}: {bdec.get(k, np.nan):.6f}\n")
        f.write("\nSkill vs climatology:\n")
        f.write(f"Brier(model):       {bs_model:.6f}\n")
        f.write(f"Brier(climatology): {bs_clim:.6f}\n")
        f.write(f"Brier Skill Score:  {bss:.6f}\n")

    return summary, calib_tbl
