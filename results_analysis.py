# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


# Function to save figures
def save_fig(fig_name, fig_dir="output/imgs", dpi=300):
    os.makedirs(fig_dir, exist_ok=True)
    file_path = os.path.join(fig_dir, fig_name)
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")


arima_results = pd.read_csv("output/values/values_arima2025-11-28.csv")
xgb_results = pd.read_csv("output/values/values_xgb2025-12-06.csv")

# %%

# arima_results
# xgb_results

# %%
x_axis = pd.date_range(start="2024-01-01", end="2024-12-31")
x_axis = x_axis[x_axis != "2024-02-29"]
x_axis
# x_axis = x_axis.loc[~((x_axis.index.month == 2) & (x_axis.index.day == 29))]

# %%

arima_results["timestamp"] = x_axis
arima_results.set_index("timestamp", inplace=True)
xgb_results["timestamp"] = x_axis
xgb_results.set_index("timestamp", inplace=True)

# %%

# Real vs predictions
plt.plot(arima_results["ytrue"], linewidth=1.5)
plt.plot(arima_results["yhat"], linewidth=1.5)
plt.plot(xgb_results["yhat"], linewidth=1.5)
plt.title("Arima and XGBoost predictions vs Real")
plt.xlabel("Time")
plt.ylabel("Nº Occurrences")
plt.grid(True)
plt.legend(["Real", "Arima", "XGBoost"])
save_fig("Real_pred_comp.pdf")
# plt.show()

# %%

#  Boxsplots by months

results = {
    "ARIMA-SW3Y": arima_results,
    "XGB-SW3Y": xgb_results,
}

dfs = []

for model_name, res in results.items():
    abs_errors = np.abs(res["ytrue"] - res["yhat"])

    df_model = pd.DataFrame({"abs": abs_errors, "model": model_name, "date": x_axis})

    dfs.append(df_model)

df = pd.concat(dfs, ignore_index=True)
df["month"] = df["date"].dt.month_name(locale="en_US")

df

# %%

palette = sns.color_palette("colorblind", n_colors=5)
meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "#4c4c4c",
    "markersize": "5",
}

plt.figure(figsize=(11.69, 8.27))
sns.boxplot(
    x="month",
    y="abs",
    hue="model",
    data=df,
    palette=palette,
    showmeans=True,
    meanprops=meanprops,
    linewidth=2.0,
)
plt.title("Month-by-month model comparison")
plt.xlabel("Months")
plt.ylabel("ABS")
plt.legend(fontsize=14)
plt.grid(axis="y")
plt.tight_layout()
save_fig("Boxplot_by_months.pdf")
# plt.show()
