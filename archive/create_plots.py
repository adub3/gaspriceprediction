
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tests.evaluation.metrics import calibration_table


def plot_time_series(output_path: str):
    """Plot the full time series of gas prices with the $3 threshold."""
    df = pd.read_csv("/Users/kabirgrewal/kalshiproject/gaspriceprediction/data/GASREGW.csv", parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "GASREGW": "price"})
    df['price'] = df['price'].interpolate()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df['date'], df['price'], label="Weekly Gas Price", color="#2c3e50", linewidth=1.5)
    ax.axhline(y=3.0, color="#c0392b", linestyle="--", linewidth=2, label="$3.00 Threshold")

    ax.fill_between(df['date'], df['price'], 3.0, where=df['price'] > 3.0, 
                    color="#e74c3c", alpha=0.3, interpolate=True)

    ax.set_title("Weekly U.S. Gas Price (1990-Present)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Price (USD per Gallon)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='major', linestyle='--', linewidth='0.5')
    fig.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Saved time series plot to {output_path}")

def plot_reliability_diagram(output_path: str):
    """Plot the reliability diagram for all three models."""
    lm_results = pd.read_csv("linear_rw_eval_results.csv")
    gb_results = pd.read_csv("gb_eval_results.csv")
    ag_results = pd.read_csv("arima_garch_eval_results.csv")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    models = {
        "Linear Model": lm_results,
        "Gradient Boosting": gb_results,
        "ARIMA-GARCH": ag_results
    }

    for name, results in models.items():
        calib_table, _ = calibration_table(results['p_hat'], results['y'], bins=10)
        ax.plot(calib_table['p_mean'], calib_table['y_freq'], marker='o', linestyle='-', label=name)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")

    ax.set_title("Reliability Diagram (Calibration)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Forecast Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='major', linestyle='--', linewidth='0.5')
    fig.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Saved reliability diagram to {output_path}")

def plot_roc_curves(output_path: str):
    """Plot the ROC curves for all three models."""
    lm_results = pd.read_csv("linear_rw_eval_results.csv")
    gb_results = pd.read_csv("gb_eval_results.csv")
    ag_results = pd.read_csv("arima_garch_eval_results.csv")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    models = {
        "Linear Model": lm_results,
        "Gradient Boosting": gb_results,
        "ARIMA-GARCH": ag_results
    }

    for name, results in models.items():
        common_index = results.index
        y_true = results.loc[common_index, 'y']
        y_pred = results.loc[common_index, 'p_hat']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")

    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=16, fontweight='bold')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='major', linestyle='--', linewidth='0.5')
    fig.tight_layout()

    plt.savefig(output_path, dpi=300)
    print(f"Saved ROC curve plot to {output_path}")


def main():
    """Generate and save all plots."""
    plot_time_series("gas_price_time_series.png")
    plot_reliability_diagram("reliability_diagram.png")
    plot_roc_curves("roc_curves.png")

if __name__ == "__main__":
    main()
