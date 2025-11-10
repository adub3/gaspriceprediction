import numpy as np
import pandas as pd
from math import erf, sqrt

# ==============================================================
# Your model (unchanged)
# ==============================================================
class LinearRWEventModel:
    """
    Linear Regression + Random Walk Event Model
    -------------------------------------------
    • Fits a simple linear trend: price_t = a + b*t + error_t
    • Uses residual std (sigma) as random-walk uncertainty
    • Predicts P(next_price > threshold) under N(mu_next, sigma^2)
    """

    def __init__(self, threshold: float = 3.00):
        self.threshold = threshold
        self.intercept_ = None
        self.slope_ = None
        self.sigma_ = None
        self.n_train_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRWEventModel":
        if X.shape[0] < 3:
            raise ValueError("Need at least 3 rows to fit regression.")
        t = np.arange(len(X), dtype=float)
        y_data = X.iloc[:, 0].astype(float).to_numpy()
        slope, intercept = np.polyfit(t, y_data, 1)
        y_hat = intercept + slope * t
        resid = y_data - y_hat
        self.intercept_ = float(intercept)
        self.slope_ = float(slope)
        self.sigma_ = float(np.std(resid, ddof=1))
        self.n_train_ = len(X)
        return self

    def predict_proba(self, X_future: pd.DataFrame) -> pd.Series:
        if any(v is None for v in [self.intercept_, self.slope_, self.sigma_]):
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        n_future = len(X_future)
        t_future = self.n_train_ + np.arange(n_future, dtype=float)
        mu_next = self.intercept_ + self.slope_ * t_future
        sigma = self.sigma_
        thr = self.threshold
        if sigma <= 0:
            probs = np.where(mu_next > thr, 1.0, 0.0)
        else:
            z = (mu_next - thr) / sigma
            probs = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        return pd.Series(probs, index=X_future.index, name="p_event")


# ==============================================================
# Metrics + Bootstrap Utilities
# ==============================================================
def _clip_probs(p, eps=1e-12):
    return np.clip(p, eps, 1.0 - eps)

def log_loss(y, p):
    p = _clip_probs(p)
    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))

def brier_score(y, p):
    return float(np.mean((p - y)**2))

def ece(y, p, n_bins=15):
    # Expected Calibration Error (weighted L1 over bins)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p, bins) - 1
    ece_val = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        nb = np.sum(mask)
        if nb == 0:
            continue
        p_hat = np.mean(p[mask])
        y_hat = np.mean(y[mask])
        ece_val += (nb / n) * abs(p_hat - y_hat)
    return float(ece_val)

def _roc_auc_numpy(y_true, y_score):
    # Pure NumPy ROC-AUC (ties handled by ranking)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Rank by score, average ranks for ties
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p)+1, dtype=float)
    # tie-adjusted: average ranks for equal scores
    # Find ties
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j+1 < len(sorted_p) and sorted_p[j+1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = (ranks[order][i:j+1].mean())
            ranks[order][i:j+1] = avg
        i = j + 1
    sum_ranks_pos = np.sum(ranks[y == 1])
    auc = (sum_ranks_pos - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg)
    return float(auc)

def auc(y, p):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return _roc_auc_numpy(y, p)

def murphy_brier_decomposition(y, p, n_bins=15):
    """
    Returns reliability, resolution, uncertainty, brier as in Murphy (1973),
    using probability bins (consistent with calibration/ECE computation).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pi_bar = np.mean(y)  # climatology
    # Bin by predicted probability
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p, bins) - 1
    reliability = 0.0
    resolution = 0.0
    n = len(y)
    for b in range(n_bins):
        mask = idx == b
        nb = np.sum(mask)
        if nb == 0:
            continue
        p_b = np.mean(p[mask])
        y_b = np.mean(y[mask])
        reliability += (nb / n) * (p_b - y_b)**2
        resolution  += (nb / n) * (y_b - pi_bar)**2
    uncertainty = pi_bar * (1.0 - pi_bar)
    brier = brier_score(y, p)
    return {
        "reliability": float(reliability),
        "resolution":  float(resolution),
        "uncertainty": float(uncertainty),
        "brier":       float(brier),
    }

def brier_skill(y, p):
    b = brier_score(y, p)
    pi_bar = float(np.mean(y))
    b_clim = pi_bar * (1.0 - pi_bar)  # climatology brier (uncertainty)
    if b_clim == 0:
        return np.nan, b, b_clim
    bss = 1.0 - b / b_clim
    return float(bss), float(b), float(b_clim)

def _bootstrap_stat(y, p, fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        samp = rng.integers(0, n, size=n)
        vals[i] = fn(y[idx[samp]], p[idx[samp]])
    return vals

def bootstrap_cis(y, p, n_boot=2000, seed=42):
    stats = {}
    # base metrics
    stats["ECE"]   = _bootstrap_stat(y, p, lambda yy,pp: ece(yy, pp, n_bins=15), n_boot, seed)
    stats["Brier"] = _bootstrap_stat(y, p, brier_score, n_boot, seed)
    stats["LogLoss"] = _bootstrap_stat(y, p, log_loss, n_boot, seed)
    stats["AUC"]   = _bootstrap_stat(y, p, auc, n_boot, seed)
    out = {}
    for k, arr in stats.items():
        mean = float(np.nanmean(arr))
        lo, hi = float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))
        out[k] = {"mean": mean, "ci": (lo, hi)}
    return out

def evaluate_metrics(y_true: pd.Series, p_pred: pd.Series, n_bins: int = 15, n_boot: int = 2000, seed: int = 42):
    # align & clean
    df = pd.concat([y_true.rename("y"), p_pred.rename("p")], axis=1).dropna()
    y = df["y"].astype(int).to_numpy()
    p = df["p"].astype(float).to_numpy()

    # core metrics
    out = {}
    out["Brier"]  = brier_score(y, p)
    out["LogLoss"] = log_loss(y, p)
    out["AUC"]    = auc(y, p)
    out["ECE"]    = ece(y, p, n_bins=n_bins)

    # Murphy decomposition
    decomp = murphy_brier_decomposition(y, p, n_bins=n_bins)
    out.update({f"decomp_{k}": v for k, v in decomp.items()})

    # Skill vs climatology
    bss, b_model, b_clim = brier_skill(y, p)
    out["BrierSkill"] = bss
    out["Brier_model"] = b_model
    out["Brier_climatology"] = b_clim

    # Bootstrap CIs
    out["bootstrap"] = bootstrap_cis(y, p, n_boot=n_boot, seed=seed)
    return out

def print_metrics_report(report: dict):
    print(f"Brier Score: {report['Brier']:.6f}")
    print(f"LogLoss:     {report['LogLoss']:.6f}")
    print(f"ROC-AUC:     {report['AUC']:.6f}")
    print(f"ECE:         {report['ECE']:.6f}\n")

    bs = report["bootstrap"]
    print("Bootstrap 95% CIs:")
    for k in ["ECE", "Brier", "LogLoss", "AUC"]:
        m = bs[k]["mean"]; lo, hi = bs[k]["ci"]
        print(f"{k}:     mean={m:.6f}, CI=({lo:.6f}, {hi:.6f})")
    print()

    print("Brier decomposition (Murphy):")
    print(f"reliability: {report['decomp_reliability']:.6f}")
    print(f"resolution:  {report['decomp_resolution']:.6f}")
    print(f"uncertainty: {report['decomp_uncertainty']:.6f}")
    print(f"brier:       {report['decomp_brier']:.6f}\n")

    print("Skill vs climatology:")
    print(f"Brier(model):       {report['Brier_model']:.6f}")
    print(f"Brier(climatology): {report['Brier_climatology']:.6f}")
    if np.isnan(report['BrierSkill']):
        print("Brier Skill Score:  NA (degenerate class distribution)")
    else:
        print(f"Brier Skill Score:  {report['BrierSkill']:.6f}")

def latex_table_row(name: str, rpt: dict) -> str:
    """
    Produce a LaTeX table row for quick copy/paste:
      Name & Brier & LogLoss & AUC & ECE & BSS \\
    """
    bss = rpt['BrierSkill']
    bss_pct = ("NA" if np.isnan(bss) else f"{100*bss:.2f}\\%")
    return (f"{name} & {rpt['Brier']:.4f} & {rpt['LogLoss']:.4f} & "
            f"{rpt['AUC']:.4f} & {rpt['ECE']:.4f} & {bss_pct} \\\\")

# ==============================================================
# Example usage
# ==============================================================
# X_train, y_train, X_test, y_test prepared elsewhere
# model = LinearRWEventModel(threshold=3.0).fit(X_train, y_train)
# p_test = model.predict_proba(X_test)
# rpt = evaluate_metrics(y_test, p_test, n_bins=15, n_boot=2000, seed=42)
# print_metrics_report(rpt)
# print(latex_table_row("LinearRW", rpt))
