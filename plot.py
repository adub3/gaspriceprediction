from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def plot_price_vs_pihat():
    # === User inputs ===
    price_csv = r"data\GASREGW.csv"          # Original price data
    pihat_csv = r"linear_rw_eval_results.csv" # Model predictions
    horizon = 3                              # shift amount (t+3 aligned to t)
    out_png = "price_vs_pihat_H3.png"        # output filename

    # === Load price data ===
    dfp = pd.read_csv(price_csv)
    dfp["observation_date"] = pd.to_datetime(dfp["observation_date"], errors="coerce")
    dfp = (
        dfp.dropna(subset=["observation_date", "GASREGW"])
           .sort_values("observation_date")
           .set_index("observation_date")
           .rename(columns={"GASREGW": "price"})
    )
    dfp["price"] = pd.to_numeric(dfp["price"], errors="coerce")
    dfp = dfp.dropna(subset=["price"])

    # === Load p_hat data ===
    dfh = pd.read_csv(pihat_csv)
    dfh["date"] = pd.to_datetime(dfh["date"], errors="coerce")
    dfh = (
        dfh.dropna(subset=["date", "p_hat"])
           .sort_values("date")
           .set_index("date")
    )
    dfh["p_hat"] = pd.to_numeric(dfh["p_hat"], errors="coerce")
    dfh = dfh.dropna(subset=["p_hat"])

    # === Align and shift price ===
    dfp["price_shifted"] = dfp["price"].shift(-horizon) - 3
    plot_df = dfh.join(dfp[["price_shifted"]], how="inner").dropna(subset=["price_shifted"])

    if plot_df.empty:
        raise ValueError("No overlapping timestamps between price and p_hat after alignment.")

    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(plot_df.index, plot_df["price_shifted"], color="tab:blue",
             linewidth=1.6, label=f"Price shifted by -{horizon}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Future Price (shifted)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(plot_df.index, plot_df["p_hat"], color="tab:red",
             linewidth=1.2, alpha=0.8, label="Predicted Probability (p_hat)")
    ax2.set_ylabel("Predicted Probability")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Future Price (shift -{horizon}) vs Predicted Probability", y=0.95)
    fig.tight_layout()

    plt.savefig(out_png, dpi=160)
    print(f"Saved plot to {out_png}")
    plt.show()

    return plot_df


if __name__ == "__main__":
    plot_df = plot_price_vs_pihat()
