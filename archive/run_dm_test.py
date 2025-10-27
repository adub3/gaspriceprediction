from __future__ import annotations

import pandas as pd
from tests.evaluation.dm_test import diebold_mariano


def main() -> None:
    """Run the Diebold-Mariano test to compare model forecasts."""
    # ----------------------------
    # 1. Load model results
    # ----------------------------
    try:
        gb_results = pd.read_csv("gb_eval_results.csv", index_col="date", parse_dates=True)
        ag_results = pd.read_csv("arima_garch_eval_results.csv", index_col="date", parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run both evaluations first.")
        return

    # ----------------------------
    # 2. Align data and calculate loss
    # ----------------------------
    # Ensure both dataframes are aligned on the same dates
    common_index = gb_results.index.intersection(ag_results.index)
    gb_results = gb_results.loc[common_index]
    ag_results = ag_results.loc[common_index]

    # Calculate the Brier score loss for each model at each time step
    # loss = (p_hat - y)^2
    loss_gb = (gb_results['p_hat'] - gb_results['y']) ** 2
    loss_ag = (ag_results['p_hat'] - ag_results['y']) ** 2

    # ----------------------------
    # 3. Run the Diebold-Mariano test
    # ----------------------------
    # The loss differential is d = loss_gb - loss_ag.
    # A positive mean(d) means that the Gradient Boosting model has a higher average loss.
    # We test the alternative hypothesis that ARIMA-GARCH is better (has less loss).
    stat, p_value = diebold_mariano(loss_gb, loss_ag, alternative="greater")

    # ----------------------------
    # 4. Print and interpret results
    # ----------------------------
    print("\n===== DIEBOLD-MARIANO TEST =====")
    print(f"Comparing Gradient Boosting (Model A) vs. ARIMA-GARCH (Model B)")
    print(f"Null Hypothesis (H0): Both models have the same forecast accuracy.")
    print(f"Alternative Hypothesis (Ha): Model A has greater loss than Model B (ARIMA-GARCH is better).")
    print("-" * 30)
    print(f"DM Statistic: {stat:.4f}")
    print(f"p-value:      {p_value:.4f}")
    print("-" * 30)

    if p_value < 0.05:
        print("Conclusion: We REJECT the null hypothesis.")
        print("The ARIMA-GARCH model is statistically significantly more accurate than the Gradient Boosting model.")
    else:
        print("Conclusion: We FAIL to reject the null hypothesis.")
        print("We cannot conclude a statistically significant difference in the models' accuracy.")


if __name__ == "__main__":
    main()
