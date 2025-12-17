# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import itertools
import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor


# %%
# Function to save figures
def save_fig(fig_name, fig_dir="output/imgs", dpi=300):
    os.makedirs(fig_dir, exist_ok=True)
    file_path = os.path.join(fig_dir, fig_name)
    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")


# %%

timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

# Base output
base_dir = "output"

img_dir = os.path.join(base_dir, "imgs")

# images directory
os.makedirs(img_dir, exist_ok=True)


# %%
# ----- READING DATASETS -----
def read_files(file_name: str):
    df = pd.read_csv(f"fires/{file_name}.csv")
    df["data_pas"] = pd.to_datetime(df["data_pas"]).dt.date
    df["data_pas"] = pd.to_datetime(df["data_pas"])

    return df


file_names = os.listdir("fires")
dfs = []

for i in file_names:
    file_name = i.split(".")[0]
    dfs.append(read_files(file_name))

dfs[0].columns

dfs[0].head()
dfs[0].shape

df_full = pd.concat(dfs).sort_values("data_pas")

df_full.set_index("data_pas", inplace=True)
df_full.head()

df_full.isna().sum()
# %%
# ----- PROPORTION OF FIRES BY BIOMES IN BRAZIL -----
# df_full['bioma'].unique()
biomes = ["Cerrado", "Mata Atlântica", "Amazônia", "Caatinga", "Pantanal", "Pampa"]

props_biomes = {}
for i in biomes:
    filter = df_full["bioma"] == i
    len_bioma = len(df_full[filter])
    avg_bioma = len_bioma / len(df_full["bioma"])

    props_biomes[i] = avg_bioma * 100

props_biomes

plt.figure(figsize=(12, 6))
plt.bar(props_biomes.keys(), props_biomes.values(), zorder=3)
plt.title("Distribution of Fires by Biomes in Brazil")
plt.xlabel("Biomes")
plt.ylabel("Proportion (%)")
plt.grid(True, zorder=0)
plt.tight_layout()
save_fig("Dist_by_biomes.png")
# plt.show()

# %%

# ----- PROPORTION OF FIRES BY REGIONS IN BRAZIL -----
# df_full["estado"].sort_values().unique()

regions_state = {
    "Norte": ["ACRE", "AMAPÁ", "AMAZONAS", "PARÁ", "RONDÔNIA", "RORAIMA", "TOCANTINS"],
    "Nordeste": [
        "ALAGOAS",
        "BAHIA",
        "CEARÁ",
        "MARANHÃO",
        "PARAÍBA",
        "PERNAMBUCO",
        "PIAUÍ",
        "RIO GRANDE DO NORTE",
        "SERGIPE",
    ],
    "Centro_Oeste": ["DISTRITO FEDERAL", "GOIÁS", "MATO GROSSO", "MATO GROSSO DO SUL"],
    "Sudeste": ["ESPÍRITO SANTO", "MINAS GERAIS", "RIO DE JANEIRO", "SÃO PAULO"],
    "Sul": ["PARANÁ", "SANTA CATARINA", "RIO GRANDE DO SUL"],
}

state_regions = {}

for key, value in regions_state.items():
    for state in value:
        state_regions[state] = key

df_full["regiao"] = df_full["estado"].map(state_regions)

df_full.head()

# OR

# def regions_state(x):
#   for key, value in regions_state.items():
#     if x in value:
#       return key

# df_full["regiao"] = df_full['estado'].apply(regions_state)
# df_full.head()

brazil_regions = ["Norte", "Nordeste", "Centro_Oeste", "Sudeste", "Sul"]

props_regions = {}
for i in brazil_regions:
    filter = df_full["regiao"] == i
    len_region = len(df_full[filter])
    avg_region = len_region / len(df_full["regiao"])

    props_regions[i] = avg_region * 100

props_regions

plt.figure(figsize=(12, 6))
plt.bar(props_regions.keys(), props_regions.values(), zorder=3)
plt.title("Distribution of Fires by Regions in Brazil")
plt.xlabel("Regions")
plt.ylabel("Proportion (%)")
plt.grid(True, zorder=0)
save_fig("Dist_by_regions.png")
# plt.show()

# %%

# ------ GROUPING AND PLOTTING TIME SERIES BY BIOMES (daily) ------

# df_full.head()
df_group = df_full.groupby(by=["bioma"]).resample("D").size()
# df_group.head()
df_group = df_group.unstack(level="bioma")
df_group = df_group.fillna(0)
# df_group.head()
df_group.columns = [
    "Amazônia",
    "Caatinga",
    "Cerrado",
    "Mata Atlântica",
    "Pampa",
    "Pantanal",
]
df_group.head()

plt.figure(figsize=(12, 6))
plt.plot(df_group["Amazônia"])
# plt.plot(df_group["Caatinga"], color='r')
plt.plot(df_group["Cerrado"], color="orange")
# plt.plot(df_group["Mata Atlântica"], color='orange')
# plt.plot(df_group["Pampa"], color='black')
# plt.plot(df_group["Pantanal"], color='pink')
# plt.title("Time series of occurrences of fires for all biomes")
plt.title("Time series of occurrences of fires for Amazônia and Cerrado (Daily)")
plt.xlabel("time")
plt.ylabel("Occurrences of fires")
plt.grid(True)
# plt.legend(["Amazônia","Caatinga","Cerrado","Mata Atlântica","Pampa","Pantanal"])
plt.legend(["Amazônia", "Cerrado"])
save_fig("TS_by_amaz_cerr_daily.png")
# plt.show()

# %%

# biomes = [
#     ["Amazônia", "Caatinga"],
#     ["Cerrado", "Mata Atlântica"],
#     ["Pampa", "Pantanal"]
# ]

# colors = [
#     ['b', 'r'],
#     ['g', 'orange'],
#     ['black', 'purple']
# ]

# fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
fig, axes = plt.subplots(2, figsize=(13, 9), sharex=True)

# for r in range(3):
#   for c in range(2):

#     biome_name = biomes[r][c]
#     color_name = colors[r][c]

#     axes[r, c].plot(df_group[biome_name], color=color_name)
#     axes[r, c].set_title(biome_name)
#     axes[r, c].grid(True)

#     if c == 0:
#       axes[r, c].set_ylabel("Nº Occurrences")

#     if r == 2:
#       axes[r, c].set_xlabel("Year")

axes[0].plot(df_group["Amazônia"])
axes[0].set_title("Amazônia")
axes[0].grid(True)
axes[0].set_ylabel("Nº Occurrences")
axes[0].set_ylim(0, 3500)

axes[1].plot(df_group["Cerrado"], color="orange")
axes[1].set_title("Cerrado")
axes[1].grid(True)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Nº Occurrences")
axes[1].set_ylim(0, 3500)

plt.tight_layout()
save_fig("Sep_TS_amaz_cerr.png")
# plt.show()

# %%
# ------ GROUPING AND PLOTTING TIME SERIES BY BIOMES (monthly) ------

# df_full.head()
df_group_m = df_full.groupby(by=["bioma"]).resample("M").size()
# df_group.head()
df_group_m = df_group_m.unstack(level="bioma")
df_group_m = df_group_m.fillna(0)
# df_group_m.head()
df_group_m.columns = [
    "Amazônia",
    "Caatinga",
    "Cerrado",
    "Mata Atlântica",
    "Pampa",
    "Pantanal",
]
df_group_m.head()

plt.figure(figsize=(15, 7))
plt.plot(df_group_m["Amazônia"], linewidth=2)
# plt.plot(df_group_m["Caatinga"], color='r')
plt.plot(df_group_m["Cerrado"], color="orange", linewidth=2)
# plt.plot(df_group_m["Mata Atlântica"], color='orange')
# plt.plot(df_group_m["Pampa"], color='black')
# plt.plot(df_group_m["Pantanal"], color='pink')
# plt.title("Time series of occurrences of fires for all biomes")
plt.title("Time series of occurrences of fires for Amazônia and Cerrado (Monthly)")
plt.xlabel("Time")
plt.ylabel("Occurrences of fires")
plt.grid(True)
# plt.legend(["Amazônia", "Caatinga", "Cerrado", "Mata Atlântica", "Pampa", "Pantanal"])
plt.legend(["Amazônia", "Cerrado"])
save_fig("TS_by_amaz_cerr_monthly.png")
# plt.show()

# %%

# fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
fig, axes = plt.subplots(2, figsize=(13, 9), sharex=True)

# for r in range(3):
#   for c in range(2):

#     biome_name = biomes[r][c]
#     color_name = colors[r][c]

#     axes[r, c].plot(df_group_m[biome_name], color=color_name)
#     axes[r, c].set_title(biome_name)
#     axes[r, c].grid(True)

#     if c == 0:
#       axes[r, c].set_ylabel("Nº Occurrences")

#     if r == 2:
#       axes[r, c].set_xlabel("Year")

axes[0].plot(df_group_m["Amazônia"], linewidth=2)
axes[0].set_title("Amazônia")
axes[0].grid(True)
axes[0].set_ylabel("Nº Occurrences")
axes[0].set_ylim(0, 45000)

axes[1].plot(df_group_m["Cerrado"], color="orange", linewidth=2)
axes[1].set_title("Cerrado")
axes[1].grid(True)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Nº Occurrences")
axes[1].set_ylim(0, 45000)

plt.tight_layout()
save_fig("Sep_TS_amaz_cerr_monthly.png")
# plt.show()

# %%

# ---- DATASPLIT -----

df_group = df_group.loc[~((df_group.index.month == 2) & (df_group.index.day == 29))]

train = df_group[:1095]
train2 = df_group[:1460]
validation = df_group[1095:1460]
test = df_group[1460:]

# %%

# ---- AMAZÔNIA BIOME ANALYSIS -----

# Autocorrelation
model = LinearRegression()

lags = [7, 30, 180, 365]
residual_norm = []

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.ravel()

for i, lag in enumerate(lags):
    # Aplica os valores de lag as observações
    x = train2[["Amazônia"]].values[:-lag]
    y = train2["Amazônia"].values[lag:]

    # Faz o fit, input/output
    model.fit(x, y)
    # Realiza a previsão, output com relação aos inputs
    pred = model.predict(x)
    print("I", i, model.coef_)

    # Calcula a norma dos resíduos
    residuos = np.sqrt(np.sum((y - pred) ** 2))
    residual_norm.append(residuos)

    axes[i].scatter(x, y, zorder=2)
    axes[i].set_xlabel("Ts")
    axes[i].set_ylabel(f"Ts + Lag ({lag})")
    axes[i].set_title(
        f"Lag {lag} - Daily Autocorrelation Strength Test using Residual Norm (Amazônia): {residual_norm[i]:.2f}"
    )
    axes[i].plot(x, pred, color="r", zorder=3)
    axes[i].grid(True, zorder=0)

plt.tight_layout()
save_fig("Res_norm_amaz.png")
# plt.show()

# %%

# lags = [1, 3, 6, 12]
# residual_norm = []

# fig, axes = plt.subplots(2, 2, figsize=(15, 8))
# axes = axes.ravel()

# for i, lag in enumerate(lags):
#     # lags
#     x = df_group_m[['Amazônia']].values[:-lag]
#     y = df_group_m['Amazônia'].values[lag:]

#     model.fit(x, y)

#     pred = model.predict(x)
#     print("I", i, model.coef_)

#     # Residual Norm
#     residuos = np.sqrt(np.sum((y - pred) ** 2))
#     residual_norm.append(residuos)

#     axes[i].scatter(x, y, zorder=2)
#     axes[i].set_xlabel('Ts')
#     axes[i].set_ylabel(f'Ts + Lag ({lag})')
#     axes[i].set_title(f'Lag {lag} - Monthly Autocorrelation Strength Test using Residual Norm (Amazônia): {residual_norm[i]:.2f}')
#     axes[i].plot(x, pred, color = 'r', zorder=3)
#     axes[i].grid(True, zorder=0)

# plt.tight_layout()
# plt.show()

# %%

# ---- CLASSICAL DECOMPOSITION ----

dec = sm.tsa.seasonal_decompose(
    train2["Amazônia"],
    model="aditive",
    period=365,
    two_sided=True,
)

x_min = train2.index.min()
x_max = train2.index.max()

plots = {
    0: [train2["Amazônia"], "Original time series (Amazônia biome)"],
    1: [dec.trend, "Trend (Amazônia biome)"],
    2: [dec.seasonal, "Seasonal component (Amazônia biome)"],
    3: [dec.resid, "Residuals (Amazônia biome)"],
}

fig, axes = plt.subplots(4, 1, figsize=(12, 8))

for i in range(4):
    axes[i].plot(plots[i][0])
    axes[i].set_title(plots[i][1])
    axes[i].grid(True)
    axes[i].set_xlim(x_min, x_max)

plt.tight_layout()
save_fig("Amaz_ts_dec.png")
# plt.show()

# %%

# Plot ACF e PACF Amazônia (Daily)

fig, axes = plt.subplots(2, figsize=(9, 8), sharex=True)

plot_acf(train2["Amazônia"], ax=axes[0], lags=50)
axes[0].set_title("ACF Amazônia (Daily)")
axes[0].grid(True)
axes[1].set_xlabel("Lags")
axes[0].set_ylabel("Correlation")

plot_pacf(train2["Amazônia"], ax=axes[1], lags=50)
axes[1].set_title("PACF Amazônia (Daily)")
axes[1].grid(True)
axes[1].set_xlabel("Lags")
axes[1].set_ylabel("Correlation")

plt.tight_layout()
save_fig("Amaz_ACF_PACF.png")
# plt.show()

# Plot ACF e PACF Amazônia (Monthly)

# fig, axes = plt.subplots(2, figsize=(9, 8), sharex=True)

# plot_acf(df_group_m["Amazônia"], ax=axes[0], lags=20)
# axes[0].set_title("ACF Amazônia (Monthly)")
# axes[0].grid(True)
# axes[1].set_xlabel("Lags")
# axes[0].set_ylabel("Correlation")

# plot_pacf(df_group_m["Amazônia"], ax=axes[1], lags=20)
# axes[1].set_title("PACF Amazônia (Monthly)")
# axes[1].grid(True)
# axes[1].set_xlabel("Lags")
# axes[1].set_ylabel("Correlation")

# plt.tight_layout()
# plt.show()

# %%

# ----- CERRADO BIOME ANALYSIS -----

model = LinearRegression()

lags = [7, 30, 180, 365]
residual_norm = []

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes = axes.ravel()

for i, lag in enumerate(lags):
    # Aplica os valores de lag as observações
    x = train2[["Cerrado"]].values[:-lag]
    y = train2["Cerrado"].values[lag:]

    # Faz o fit, input/output
    model.fit(x, y)
    # Realiza a previsão, output com relação aos inputs
    pred = model.predict(x)
    print("I", i, model.coef_)

    # Calcula a norma dos resíduos
    residuos = np.sqrt(np.sum((y - pred) ** 2))
    residual_norm.append(residuos)

    axes[i].scatter(x, y, zorder=2)
    axes[i].set_xlabel("Ts")
    axes[i].set_ylabel(f"Ts + Lag ({lag})")
    axes[i].set_title(
        f"Lag {lag} - Daily Autocorrelation Strength Test using Residual Norm (Cerrado): {residual_norm[i]:.2f}"
    )
    axes[i].plot(x, pred, color="r", zorder=3)
    axes[i].grid(True, zorder=0)

plt.tight_layout()
save_fig("Res_norm_cerr.png")
# plt.show()

# %%

# lags = [1, 3, 6, 12]
# residual_norm = []

# fig, axes = plt.subplots(2, 2, figsize=(15, 8))
# axes = axes.ravel()

# for i, lag in enumerate(lags):
#     # lags
#     x = df_group_m[['Cerrado']].values[:-lag]
#     y = df_group_m['Cerrado'].values[lag:]

#     model.fit(x, y)

#     pred = model.predict(x)
#     print("I", i, model.coef_)

#     # Residual Norm
#     residuos = np.sqrt(np.sum((y - pred) ** 2))
#     residual_norm.append(residuos)

#     axes[i].scatter(x, y, zorder=2)
#     axes[i].set_xlabel('Ts')
#     axes[i].set_ylabel(f'Ts + Lag ({lag})')
#     axes[i].set_title(f'Lag {lag} - Monthly Autocorrelation Strength Test using Residual Norm (Cerrado): {residual_norm[i]:.2f}')
#     axes[i].plot(x, pred, color = 'r', zorder=3)
#     axes[i].grid(True, zorder=0)

# plt.tight_layout()
# plt.show()

# %%

# ---- CLASSICAL DECOMPOSITION ----

dec = sm.tsa.seasonal_decompose(
    train2["Cerrado"],
    model="aditive",
    period=365,
    two_sided=True,
)

x_min = train2.index.min()
x_max = train2.index.max()

plots = {
    0: [train2["Cerrado"], "Original time series (Cerrado biome)"],
    1: [dec.trend, "Trend (Cerrado biome)"],
    2: [dec.seasonal, "Seasonal component (Cerrado biome)"],
    3: [dec.resid, "Residuals (Cerrado biome)"],
}

fig, axes = plt.subplots(4, 1, figsize=(12, 8))

for i in range(4):
    axes[i].plot(plots[i][0])
    axes[i].set_title(plots[i][1])
    axes[i].grid(True)
    axes[i].set_xlim(x_min, x_max)

plt.tight_layout()
save_fig("Cerr_ts_dec.png")
# plt.show()

# %%

# Plot ACF e PACF Cerrado (Daily)

fig, axes = plt.subplots(2, figsize=(9, 8), sharex=True)

plot_acf(train2["Cerrado"], ax=axes[0], lags=50)
axes[0].set_title("ACF Cerrado")
axes[0].grid(True)
axes[1].set_xlabel("Lags")
axes[0].set_ylabel("Correlation")

plot_pacf(train2["Cerrado"], ax=axes[1], lags=50)
axes[1].set_title("PACF Cerrado")
axes[1].grid(True)
axes[1].set_xlabel("Lags")
axes[1].set_ylabel("Correlation")

plt.tight_layout()
save_fig("Cerr_ACF_PACF.png")
# plt.show()

# Plot ACF e PACF Cerrado (Monthly)

# fig, axes = plt.subplots(2, figsize=(9, 8), sharex=True)

# plot_acf(df_group_m["Cerrado"], ax=axes[0], lags=20)
# axes[0].set_title("ACF Cerrado (Monthly)")
# axes[0].grid(True)
# axes[0].set_xlabel("Lags")
# axes[0].set_ylabel("Correlation")

# plot_pacf(df_group_m["Cerrado"], ax=axes[1], lags=20)
# axes[1].set_title("PACF Cerrado (Monthly)")
# axes[1].grid(True)
# axes[1].set_xlabel("Lags")
# axes[1].set_ylabel("Correlation")

# plt.tight_layout()
# plt.show()

# %%

# ----- DATA NORMALIZATION -----

scaler = MinMaxScaler()

train2 = df_group[["Amazônia"]][:1460].copy()
test = df_group[["Amazônia"]][365:].copy()

scaler.fit(train2)

train2_transf = scaler.transform(train2)
test_transf = scaler.transform(test)
# %%

# ----- ARIMA AND XGBOOST FORECASTING ------


class SlidingWindow:
    #   Vai instânciar a classe e vai necessitar da definição do tamanho das amostras
    def __init__(self, n_samples, trainw, testw):
        self.n_samples = n_samples
        self.trainw = trainw
        self.testw = testw

        self.n_splits = math.ceil((self.n_samples - self.trainw) / testw)

        #       Faz uma verificação e aponta um erro caso as condições não sejam atendidas
        assert n_samples != self.trainw
        assert self.testw > 0

    #   Realiza as divisões para os conjuntos
    def split(self, X, y=None, groups=None):
        #       Gera duas sequências de dados, indo de treino ao final, com o passo de teste
        for i, k in enumerate(range(self.trainw, self.n_samples, self.testw)):

            #           Faz a separação dos dados
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.testw)

            if i + 1 == self.n_splits:
                testidxs = slice(k, self.n_samples)

            yield trainidxs, testidxs

            if i + 1 == self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# %%

# Configuring directory to save metrics and predictions

# Models outputs
metrics_dir = os.path.join(base_dir, "metrics")
values_dir = os.path.join(base_dir, "values")

# Create directories
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(values_dir, exist_ok=True)

metrics_file_arima = os.path.join(metrics_dir, f"metrics_arima{timestamp}.csv")
metrics_file_xgb = os.path.join(metrics_dir, f"metrics_xgb{timestamp}.csv")

values_file_arima = os.path.join(values_dir, f"values_arima{timestamp}.csv")
values_file_xgb = os.path.join(values_dir, f"values_xgb{timestamp}.csv")

with open(metrics_file_arima, "w") as f:
    f.write("model,order,rmse,rmse_std,mae,mae_std,split\n")

with open(metrics_file_xgb, "w") as f:
    f.write("model,n_est,max_d,max_l,rmse,rmse_std,mae,mae_std,split\n")

with open(values_file_arima, "w") as f:
    f.write("model,ytrue,yhat\n")

with open(values_file_xgb, "w") as f:
    f.write("model,ytrue,yhat\n")

# %%

# ----- ARIMA -----

# --------------------- ARIMA SW3Y VALIDATION (AMAZÔNIA) ---------------------

p_values = [2, 5]
d_values = [0, 1]
q_values = [41]

param_combinations = list(itertools.product(p_values, d_values, q_values))

SW3Y_val = SlidingWindow(
    n_samples=len(df_group["Amazônia"].loc["2020-01-01":"2023-12-31"]),
    trainw=len(df_group["Amazônia"].loc["2020-01-01":"2022-12-31"]),
    testw=7,
)

best_rmse_SW3Y_val = float("inf")
best_mae_SW3Y_val = None
best_params_SW3Y = None

for p, d, q in param_combinations:
    arima_order = (p, d, q)

    resultsSW3Y_val = dict(ytrue=[], yhat=[])
    scoringSW3Y_val = dict(rmse=[], mae=[])

    print("ordem atual", arima_order)

    # Loop para Sliding Window
    for i, (trainidxs, testidxs) in enumerate(SW3Y_val.split(df_group[["Amazônia"]])):
        y = train2_transf[trainidxs]

        # Dados de validação
        y_t = train2_transf[testidxs]

        arima_model = sm.tsa.ARIMA(y, order=arima_order)
        arima_model.initialize_approximate_diffuse()
        arima_fit = arima_model.fit()

        # Previsões para os dados de validação
        predictions = arima_fit.forecast(steps=len(y_t))

        yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
        ytrue = scaler.inverse_transform(y_t)

        if len(ytrue) > 0:
            rmse = math.sqrt(mean_squared_error(ytrue, yhat))
            mae = mean_absolute_error(ytrue, yhat)

            scoringSW3Y_val["rmse"].append(rmse)
            scoringSW3Y_val["mae"].append(mae)

    rmse_mean = round(np.mean(scoringSW3Y_val["rmse"]), 2)
    mae_mean = round(np.mean(scoringSW3Y_val["mae"]), 2)

    if rmse_mean < best_rmse_SW3Y_val:
        best_rmse_SW3Y_val = rmse_mean
        best_mae_SW3Y_val = mae_mean
        best_params_SW3Y = arima_order

print("Best_Params_SW3Y_Val", best_params_SW3Y)
print("Best_RMSE_SW3Y_Val : ", best_rmse_SW3Y_val)
print("Best_MAE_SW3Y_Val:", best_mae_SW3Y_val)


# -------------------------------------------- ARIMA SW3Y TEST (AMAZÔNIA) ------------------------------------------------

SW3Y_test = SlidingWindow(
    n_samples=len(df_group["Amazônia"].loc["2021-01-01":"2024-12-31"]),
    trainw=len(df_group["Amazônia"].loc["2021-01-01":"2023-12-31"]),
    testw=7,
)

resultsSW3Y_test = dict(ytrue=[], yhat=[])
scoringSW3Y = dict(rmse=[], mae=[])

print("Parametros atuais para teste: ", best_params_SW3Y)

for i, (trainidxs, testidxs) in enumerate(SW3Y_test.split(test)):
    y = test_transf[trainidxs]
    print(trainidxs)

    y_t = test_transf[testidxs]
    print(testidxs)

    arima_model = sm.tsa.ARIMA(y, order=best_params_SW3Y)
    arima_fit = arima_model.fit()

    # Previsões para os dados de teste
    predictions = arima_fit.forecast(steps=len(y_t))

    yhat = scaler.inverse_transform(predictions.reshape(-1, 1))
    ytrue = scaler.inverse_transform(y_t)

    resultsSW3Y_test["ytrue"].append(ytrue)
    resultsSW3Y_test["yhat"].append(yhat)

    if len(ytrue) > 0:
        rmse = math.sqrt(mean_squared_error(ytrue, yhat))
        mae = mean_absolute_error(ytrue, yhat)

        scoringSW3Y["rmse"].append(rmse)
        scoringSW3Y["mae"].append(mae)

rmse_mean = round(np.mean(scoringSW3Y["rmse"]), 2)
rmse_std = round(np.std(scoringSW3Y["rmse"]), 2)
mae_mean = round(np.mean(scoringSW3Y["mae"]), 2)
mae_std = round(np.std(scoringSW3Y["mae"]), 2)

# Save metrics into csv
with open(metrics_file_arima, "a") as f:
    f.write(
        f"ARIMA-SW3Y,{best_params_SW3Y},{rmse_mean},{rmse_std},{mae_mean},{mae_std},test\n"
    )

# Save true and preds into csv
with open(values_file_arima, "a") as f:
    for ytrue, yhat in zip(resultsSW3Y_test["ytrue"], resultsSW3Y_test["yhat"]):
        for true, pred in zip(ytrue, yhat):
            f.write(f"ARIMA-SW3Y,{true},{pred}\n")

print("-" * 20)

print("RMSE_SW3Y_Test:", rmse_mean)
print("RMSE_std_SW3Y_Test:", rmse_std)
print("MAE_SW3Y_Test:", mae_mean)
print("MAE_std_SW3Y_Test:", mae_std)

# %%

test_SW3_arima = pd.DataFrame()
dates = pd.date_range(start="2024-01-01", end="2024-12-31")
dates = dates[~((dates.month == 2) & (dates.day == 29))]
test_SW3_arima.index = dates

ytrue_list = []
yhat_list = []

for i in range(len(resultsSW3Y_test["ytrue"])):
    ytrue_list.extend(resultsSW3Y_test["ytrue"][i])
    yhat_list.extend(resultsSW3Y_test["yhat"][i])

test_SW3_arima["ytrue"] = ytrue_list
test_SW3_arima["yhat"] = yhat_list

test_SW3_arima["ytrue"] = test_SW3_arima["ytrue"].apply(lambda x: x[0])
test_SW3_arima["yhat"] = test_SW3_arima["yhat"].apply(lambda x: x[0])
test_SW3_arima

plt.plot(test_SW3_arima["ytrue"])
plt.plot(test_SW3_arima["yhat"])

# %%

# ----- XGBoost -----

# Source: https://www.kaggle.com/code/vitthalmadane/xgboost-on-univariate-time-series


# Define a function to generate features from datetime Index
def Generate_features(df, label=None):

    df["date"] = pd.to_datetime(df.index)
    # df['hour'] = df['date'].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day

    X = df[
        [
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
        ]
    ]
    if label:
        y = df[label]
        return X, y
    return X


X_train, y_train = Generate_features(train2, label="Amazônia")
X_test, y_test = Generate_features(test, label="Amazônia")

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

scaler_x.fit(X_train)
scaler_y.fit(y_train)

X_train_transf = scaler_x.transform(X_train)
y_train_transf = scaler_y.transform(y_train)

X_test_transf = scaler_x.transform(X_test)
y_test_transf = scaler_y.transform(y_test)

n_estimators = [200, 400, 600]
max_depth = [3, 5, 8]
max_leaves = [16, 32, 64]

param_combinations = list(itertools.product(n_estimators, max_depth, max_leaves))

# %%

# --------------------- XGB SW3Y VALIDATION (AMAZÔNIA) ---------------------

SW3Y_val = SlidingWindow(
    n_samples=len(df_group["Amazônia"].loc["2020-01-01":"2023-12-31"]),
    trainw=len(df_group["Amazônia"].loc["2020-01-01":"2022-12-31"]),
    testw=7,
)

best_rmse_SW3Y_val = float("inf")
best_mae_SW3Y_val = None
best_n_estimator_SW3Y = None
best_max_d_SW3Y = None
best_max_l_SW3Y = None

for n_estimator, max_d, max_l in param_combinations:

    resultsSW3Y_val = dict(ytrue=[], yhat=[])
    scoringSW3Y_val = dict(rmse=[], mae=[])

    print("curr hips", n_estimator, max_d, max_l)

    # Loop para Sliding Window
    for i, (trainidxs, testidxs) in enumerate(SW3Y_val.split(train2)):
        X = X_train_transf[trainidxs]
        y = y_train_transf[trainidxs]

        X_t = X_train_transf[testidxs]
        y_t = y_train_transf[testidxs]

        model = XGBRegressor(
            n_estimators=n_estimator,
            max_depth=max_d,
            max_leaves=max_l,
            early_stopping_rounds=60,
        )

        # Declare Evaluation data
        eval_data = [(X, y), (X_t, y_t)]
        model.fit(X, y, eval_set=eval_data, verbose=False)

        predictions = model.predict(X_t)

        yhat = scaler_y.inverse_transform(predictions.reshape(-1, 1))
        ytrue = scaler_y.inverse_transform(y_t)

        if len(ytrue) > 0:
            rmse = math.sqrt(mean_squared_error(ytrue, yhat))
            mae = mean_absolute_error(ytrue, yhat)

            scoringSW3Y_val["rmse"].append(rmse)
            scoringSW3Y_val["mae"].append(mae)

    rmse_mean = round(np.mean(scoringSW3Y_val["rmse"]), 2)
    mae_mean = round(np.mean(scoringSW3Y_val["mae"]), 2)

    if rmse_mean < best_rmse_SW3Y_val:
        best_rmse_SW3Y_val = rmse_mean
        best_mae_SW3Y_val = mae_mean
        best_n_estimator_SW3Y = n_estimator
        best_max_d_SW3Y = max_d
        best_max_l_SW3Y = max_l

print("Best_n_estimator_SW3Y", best_n_estimator_SW3Y)
print("Best_max_d_SW3Y", best_max_d_SW3Y)
print("Best_max_l_SW3Y", best_max_l_SW3Y)
print("Best_RMSE_SW3Y_Val : ", best_rmse_SW3Y_val)
print("Best_MAE_SW3Y_Val:", best_mae_SW3Y_val)

# --------------------- XGB SW3Y TEST (AMAZÔNIA) ---------------------

SW3Y_test = SlidingWindow(
    n_samples=len(df_group["Amazônia"].loc["2021-01-01":"2024-12-31"]),
    trainw=len(df_group["Amazônia"].loc["2021-01-01":"2023-12-31"]),
    testw=7,
)

resultsSW3Y_test = dict(ytrue=[], yhat=[])
scoringSW3Y_test = dict(rmse=[], mae=[])

print("best hips", best_n_estimator_SW3Y, best_max_d_SW3Y, best_max_l_SW3Y)

# Loop para Sliding Window
for i, (trainidxs, testidxs) in enumerate(SW3Y_val.split(test)):
    X = X_test_transf[trainidxs]
    y = y_test_transf[trainidxs]

    X_t = X_test_transf[testidxs]
    y_t = y_test_transf[testidxs]

    model = XGBRegressor(
        n_estimators=best_n_estimator_SW3Y,
        max_depth=best_max_d_SW3Y,
        max_leaves=best_max_l_SW3Y,
        early_stopping_rounds=60,
    )

    # Declare Evaluation data
    eval_data = [(X, y), (X_t, y_t)]
    model.fit(X, y, eval_set=eval_data, verbose=False)

    predictions = model.predict(X_t)

    yhat = scaler_y.inverse_transform(predictions.reshape(-1, 1))
    ytrue = scaler_y.inverse_transform(y_t)

    resultsSW3Y_test["ytrue"].append(ytrue)
    resultsSW3Y_test["yhat"].append(yhat)

    if len(ytrue) > 0:
        rmse = math.sqrt(mean_squared_error(ytrue, yhat))
        mae = mean_absolute_error(ytrue, yhat)

        scoringSW3Y_test["rmse"].append(rmse)
        scoringSW3Y_test["mae"].append(mae)

rmse_mean = round(np.mean(scoringSW3Y_test["rmse"]), 2)
rmse_std = round(np.std(scoringSW3Y_test["rmse"]), 2)
mae_mean = round(np.mean(scoringSW3Y_test["mae"]), 2)
mae_std = round(np.std(scoringSW3Y_test["mae"]), 2)

# Save metrics into csv
with open(metrics_file_xgb, "a") as f:
    f.write(
        f"XGB-SW3Y,{best_n_estimator_SW3Y},{best_max_d_SW3Y},{best_max_l_SW3Y},{rmse_mean},{rmse_std},{mae_mean},{mae_std},test\n"
    )

# Save true and preds into csv
with open(values_file_xgb, "a") as f:
    for ytrue, yhat in zip(resultsSW3Y_test["ytrue"], resultsSW3Y_test["yhat"]):
        for true, pred in zip(ytrue, yhat):
            f.write(f"XGB-SW3Y,{true.item()},{pred.item()}\n")

print("-" * 20)
print("RMSE_SW3Y_Test:", rmse_mean)
print("RMSE_SW3Y_Test_std:", rmse_std)
print("MAE_SW3Y_Test:", mae_mean)
print("MAE_SW3Y_Test_std:", mae_std)

# %%

test_SW3_xg = pd.DataFrame()
dates = pd.date_range(start="2024-01-01", end="2024-12-31")
dates = dates[~((dates.month == 2) & (dates.day == 29))]
test_SW3_xg.index = dates

ytrue_list = []
yhat_list = []

for i in range(len(resultsSW3Y_test["ytrue"])):
    ytrue_list.extend(resultsSW3Y_test["ytrue"][i])
    yhat_list.extend(resultsSW3Y_test["yhat"][i])

test_SW3_xg["ytrue"] = ytrue_list
test_SW3_xg["yhat"] = yhat_list

test_SW3_xg["ytrue"] = test_SW3_xg["ytrue"].apply(lambda x: x[0])
test_SW3_xg["yhat"] = test_SW3_xg["yhat"].apply(lambda x: x[0])
test_SW3_xg

plt.plot(test_SW3_xg["ytrue"])
plt.plot(test_SW3_xg["yhat"])