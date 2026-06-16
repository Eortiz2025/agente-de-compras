import streamlit as st import pandas as pd import numpy as np

APP_VERSION = “v9.0 RIDGE”

MIN_ROTACION_V30D = 3 COMPRA_MINIMA_UNIDAD = 1

Mezcla base

PESO_REGRESION = 0.70 PESO_V30D = 0.30

Seguridad de regresión

MIN_MESES_PARA_REGRESION = 3 MAX_FACTOR_SOBRE_HISTORICO = 2.5
MAX_FACTOR_SOBRE_V30D = 3.0

Ridge

RIDGE_ALPHA = 3.0 MIN_FILAS_ENTRENAMIENTO = 30

Estacionalidad

USAR_ESTACIONALIDAD = True ANOS_ESTACIONALIDAD = [2024, 2025]
PESO_ANO_ESTACIONALIDAD = { 2024: 0.4, 2025: 0.6, }
FACTOR_ESTACIONAL_MIN = 0.5 FACTOR_ESTACIONAL_MAX = 2.5

Ventana de compra

MESES_ANTICIPACION = 1 PESO_MES_ACTUAL = 0.70 PESO_MES_SIGUIENTE = 0.30

st.set_page_config(page_title=“Agente de compras”, layout=“wide”)

=========================

HELPERS

=========================

def norm_code(s): return s.astype(str).str.strip().str.upper()

def round_normal(qty): if pd.isna(qty) or qty <= 0: return 0 return
int(np.ceil(qty))

def current_month(): return pd.Timestamp.today().month

def next_month(m): return 1 if m == 12 else m + 1

def safe_div(a, b): return np.where(np.abs(b) > 1e-9, a / b, 0.0)

=========================

RIDGE REGRESSION CON NUMPY

=========================

class NumpyRidgeRegression: def init(self, alpha=1.0): self.alpha =
alpha self.coef_ = None self.intercept_ = None self.feature_names_ =
None self.is_fitted_ = False

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X debe ser 2D.")
        if y.ndim != 1:
            raise ValueError("y debe ser 1D.")
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud.")

        X_design = np.column_stack([np.ones(len(X)), X])
        n_features = X_design.shape[1]

        I = np.eye(n_features)
        I[0, 0] = 0.0

        XtX = X_design.T @ X_design
        Xty = X_design.T @ y

        beta = np.linalg.solve(XtX + self.alpha * I, Xty)

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self.feature_names_ = feature_names if feature_names is not None else []
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("El modelo no ha sido entrenado.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X debe ser 2D.")
        return self.intercept_ + X @ self.coef_

=========================

ERPLY PARSER

=========================

def read_erply(file): tables = pd.read_html(file, header=None) df =
max(tables, key=lambda x: x.shape[0])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    def is_code(x):
        s = str(x).strip()
        return len(s) >= 3 and not s.lower().startswith("codigo")

    start = 0
    for i in range(min(100, len(df))):
        if is_code(df.iloc[i, 1]):
            start = i
            break

    df = df.iloc[start:].reset_index(drop=True)

    out = pd.DataFrame({
        "Código": df.iloc[:, 1].astype(str).str.strip(),
        "EAN": df.iloc[:, 2].astype(str).str.strip(),  # ← NUEVO
        "Nombre": df.iloc[:, 3].astype(str).fillna(""),
        "V30D": pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0),
        "Stock": pd.to_numeric(df.iloc[:, 6], errors="coerce").fillna(0),
    })

    out["Código"] = norm_code(out["Código"])

    out = out[~out["Código"].str.contains("TOTAL", na=False)]
    out = out[~out["Nombre"].astype(str).str.upper().str.contains("TOTAL", na=False)]

    return out.reset_index(drop=True)

=========================

HISTORICO PREP

=========================

def prepare_hist(hist): hist = hist.copy() hist[“Código”] =
norm_code(hist[“Código”]) hist[“Ventas”] = pd.to_numeric(hist[“Ventas”],
errors=“coerce”).fillna(0) hist[“Importe”] =
pd.to_numeric(hist[“Importe”], errors=“coerce”).fillna(0) hist[“Año”] =
pd.to_numeric(hist[“Año”], errors=“coerce”).fillna(0).astype(int)
hist[“Mes”] = pd.to_numeric(hist[“Mes”],
errors=“coerce”).fillna(0).astype(int)

    hist = hist[(hist["Mes"] >= 1) & (hist["Mes"] <= 12) & (hist["Año"] > 0)].copy()
    return hist

=========================

FEATURES MENSUALES MEJORADAS

=========================

def build_monthly_features(hist): monthly = ( hist.groupby([“Código”,
“Año”, “Mes”], as_index=False) .agg({ “Ventas”: “sum”, “Importe”: “sum”
}) .sort_values([“Código”, “Año”, “Mes”]) .reset_index(drop=True) )

    monthly["Fecha"] = pd.to_datetime(
        monthly["Año"].astype(str) + "-" + monthly["Mes"].astype(str).str.zfill(2) + "-01"
    )

    monthly = monthly.sort_values(["Código", "Fecha"]).reset_index(drop=True)

    g = monthly.groupby("Código")["Ventas"]

    monthly["lag1"] = g.shift(1)
    monthly["lag2"] = g.shift(2)
    monthly["lag3"] = g.shift(3)
    monthly["lag6"] = g.shift(6)
    monthly["lag12"] = g.shift(12)

    monthly["ma3"] = monthly[["lag1", "lag2", "lag3"]].mean(axis=1)
    monthly["std3"] = monthly[["lag1", "lag2", "lag3"]].std(axis=1)
    monthly["max3"] = monthly[["lag1", "lag2", "lag3"]].max(axis=1)
    monthly["min3"] = monthly[["lag1", "lag2", "lag3"]].min(axis=1)

    monthly["diff1"] = monthly["lag1"] - monthly["lag2"]
    monthly["diff2"] = monthly["lag2"] - monthly["lag3"]

    monthly["ratio1"] = safe_div(monthly["lag1"], monthly["lag2"] + 1)
    monthly["ratio2"] = safe_div(monthly["lag2"], monthly["lag3"] + 1)

    monthly["trend_idx"] = monthly.groupby("Código").cumcount() + 1

    monthly["Mes_sin"] = np.sin(2 * np.pi * monthly["Mes"] / 12)
    monthly["Mes_cos"] = np.cos(2 * np.pi * monthly["Mes"] / 12)

    train = monthly.dropna(subset=["lag1", "lag2", "lag3"]).copy()

    numeric_cols = [
        "lag1", "lag2", "lag3", "lag6", "lag12",
        "ma3", "std3", "max3", "min3",
        "diff1", "diff2", "ratio1", "ratio2",
        "trend_idx", "Mes_sin", "Mes_cos", "Ventas"
    ]
    for c in numeric_cols:
        if c in train.columns:
            train[c] = pd.to_numeric(train[c], errors="coerce")

    train = train.replace([np.inf, -np.inf], np.nan)

    return monthly, train

def get_feature_cols(): return [ “lag1”, “lag2”, “lag3”, “lag6”,
“lag12”, “ma3”, “std3”, “max3”, “min3”, “diff1”, “diff2”, “ratio1”,
“ratio2”, “trend_idx”, “Mes_sin”, “Mes_cos” ]

def train_global_regression(train): feature_cols = get_feature_cols()

    if train.empty or len(train) < MIN_FILAS_ENTRENAMIENTO:
        return None, feature_cols

    train = train.copy()
    for c in feature_cols:
        if c not in train.columns:
            train[c] = 0.0

    X = train[feature_cols].fillna(0).values
    y = train["Ventas"].fillna(0).values

    model = NumpyRidgeRegression(alpha=RIDGE_ALPHA).fit(X, y, feature_names=feature_cols)
    return model, feature_cols

def predict_next_month_per_sku(monthly, model, feature_cols): if model
is None: return pd.DataFrame(columns=[“Código”,
“Pred_Regresion_Mensual”, “Meses_Historial”])

    rows = []

    for codigo, g in monthly.groupby("Código"):
        g = g.sort_values("Fecha").reset_index(drop=True)
        ventas = g["Ventas"].tolist()

        meses_historial = int((g["Ventas"] > 0).sum())

        last1 = ventas[-1] if len(ventas) >= 1 else 0
        last2 = ventas[-2] if len(ventas) >= 2 else 0
        last3 = ventas[-3] if len(ventas) >= 3 else 0
        last6 = ventas[-6] if len(ventas) >= 6 else 0
        last12 = ventas[-12] if len(ventas) >= 12 else 0

        vals3 = np.array([last1, last2, last3], dtype=float)

        ma3 = float(np.mean(vals3))
        std3 = float(np.std(vals3))
        max3 = float(np.max(vals3))
        min3 = float(np.min(vals3))

        diff1 = float(last1 - last2)
        diff2 = float(last2 - last3)
        ratio1 = float(last1 / (last2 + 1))
        ratio2 = float(last2 / (last3 + 1))
        trend_idx = float(len(g) + 1)

        last_fecha = g["Fecha"].iloc[-1]
        pred_month = next_month(last_fecha.month)

        mes_sin = float(np.sin(2 * np.pi * pred_month / 12))
        mes_cos = float(np.cos(2 * np.pi * pred_month / 12))

        X_pred = pd.DataFrame([{
            "lag1": last1, "lag2": last2, "lag3": last3,
            "lag6": last6, "lag12": last12,
            "ma3": ma3, "std3": std3, "max3": max3, "min3": min3,
            "diff1": diff1, "diff2": diff2,
            "ratio1": ratio1, "ratio2": ratio2,
            "trend_idx": trend_idx,
            "Mes_sin": mes_sin, "Mes_cos": mes_cos
        }])

        for c in feature_cols:
            if c not in X_pred.columns:
                X_pred[c] = 0.0

        X_pred = X_pred[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred = model.predict(X_pred.values)[0]
        pred = max(0, pred)

        rows.append({
            "Código": codigo,
            "Pred_Regresion_Mensual": pred,
            "Meses_Historial": meses_historial
        })

    return pd.DataFrame(rows)

=========================

COSTO

=========================

def build_cost(hist): cost_2025 = ( hist[hist[“Año”] == 2025]
.groupby(“Código”) .agg({“Ventas”: “sum”, “Importe”: “sum”})
.reset_index() ) cost_2025[“Costo_2025”] = np.where(
cost_2025[“Ventas”] > 0, cost_2025[“Importe”] / cost_2025[“Ventas”],
np.nan )

    cost_all = (
        hist.groupby("Código")
        .agg({"Ventas": "sum", "Importe": "sum"})
        .reset_index()
    )
    cost_all["Costo_All"] = np.where(
        cost_all["Ventas"] > 0,
        cost_all["Importe"] / cost_all["Ventas"],
        np.nan
    )

    cost = cost_2025[["Código", "Costo_2025"]].merge(
        cost_all[["Código", "Costo_All"]],
        on="Código",
        how="outer"
    )

    cost["Costo"] = cost["Costo_2025"].fillna(cost["Costo_All"])
    return cost[["Código", "Costo"]]

=========================

DEMANDA HISTORICA APOYO

=========================

def build_school_demand(hist): hist_escolar =
hist[hist[“Mes”].between(4, 10)]

    dem_2025 = (
        hist_escolar[hist_escolar["Año"] == 2025]
        .groupby("Código")["Ventas"]
        .sum()
        .rename("Dem_2025")
    )

    dem_2024 = (
        hist_escolar[hist_escolar["Año"] == 2024]
        .groupby("Código")["Ventas"]
        .sum()
        .rename("Dem_2024")
    )

    df = pd.DataFrame({
        "Dem_2025": dem_2025,
        "Dem_2024": dem_2024
    }).fillna(0).reset_index()

    df["Ratio"] = np.where(
        df["Dem_2024"] > 0,
        df["Dem_2025"] / df["Dem_2024"],
        np.where(df["Dem_2025"] > 0, 9.99, 1)
    )

    def clas(r):
        if r < 0.7:
            return "SOBRECOMPRA"
        elif r <= 1.1:
            return "ALINEADO"
        else:
            return "SUBESTIMADO"

    df["Tipo"] = df["Ratio"].apply(clas)

    def demanda(row):
        if row["Tipo"] == "SOBRECOMPRA":
            return 0.9 * row["Dem_2025"] + 0.1 * row["Dem_2024"]
        elif row["Tipo"] == "ALINEADO":
            return 0.75 * row["Dem_2025"] + 0.25 * row["Dem_2024"]
        else:
            return 0.6 * row["Dem_2025"] + 0.4 * row["Dem_2024"]

    df["Demanda_Base"] = df.apply(demanda, axis=1)
    df["Demanda_Mensual_Historica"] = df["Demanda_Base"] / 7

    return df

=========================

COLUMNAS ABRIL / MAYO 2025

=========================

def build_v04_v05(hist): v04 = ( hist[(hist[“Año”] == 2025) &
(hist[“Mes”] == 4)] .groupby(“Código”)[“Ventas”] .sum()
.rename(“V04_2025”) )

    v05 = (
        hist[(hist["Año"] == 2025) & (hist["Mes"] == 5)]
        .groupby("Código")["Ventas"]
        .sum()
        .rename("V05_2025")
    )

    return v04, v05

=========================

FALLBACK GLOBAL DE COSTO

=========================

def fill_missing_costs_with_global_average(final, hist): total_ventas =
hist[“Ventas”].sum() total_importe = hist[“Importe”].sum() global_cost =
(total_importe / total_ventas) if total_ventas > 0 else 0

    final["Costo"] = final["Costo"].fillna(global_cost).fillna(0)
    return final

=========================

ESTACIONALIDAD AUTOMATICA POR SKU

=========================

def build_seasonality(hist): hist_seas =
hist[hist[“Año”].isin(ANOS_ESTACIONALIDAD)].copy()

    if hist_seas.empty:
        return pd.DataFrame(columns=["Código", "Mes", "Factor_Estacional"])

    hist_seas["Peso_Ano"] = hist_seas["Año"].map(PESO_ANO_ESTACIONALIDAD).fillna(1.0)
    hist_seas["Ventas_Ponderadas"] = hist_seas["Ventas"] * hist_seas["Peso_Ano"]

    by_month = (
        hist_seas.groupby(["Código", "Mes"], as_index=False)["Ventas_Ponderadas"]
        .sum()
        .rename(columns={"Ventas_Ponderadas": "Ventas_Mes_Pond"})
    )

    skus = pd.DataFrame({"Código": hist_seas["Código"].unique()})
    meses = pd.DataFrame({"Mes": np.arange(1, 13)})
    base = skus.assign(key=1).merge(meses.assign(key=1), on="key").drop(columns="key")

    seas = base.merge(by_month, on=["Código", "Mes"], how="left")
    seas["Ventas_Mes_Pond"] = seas["Ventas_Mes_Pond"].fillna(0)

    total_sku = (
        seas.groupby("Código", as_index=False)["Ventas_Mes_Pond"]
        .sum()
        .rename(columns={"Ventas_Mes_Pond": "Ventas_Total_Pond"})
    )

    seas = seas.merge(total_sku, on="Código", how="left")

    seas["Factor_Estacional"] = np.where(
        seas["Ventas_Total_Pond"] > 0,
        (seas["Ventas_Mes_Pond"] * 12.0) / seas["Ventas_Total_Pond"],
        1.0
    )

    seas["Factor_Estacional"] = seas["Factor_Estacional"].clip(
        lower=FACTOR_ESTACIONAL_MIN,
        upper=FACTOR_ESTACIONAL_MAX
    )

    return seas[["Código", "Mes", "Factor_Estacional"]]

def build_current_seasonality_for_purchase(seasonality_df): if
seasonality_df.empty: return pd.DataFrame(columns=[“Código”,
“Factor_Estacional_Compra”])

    mes_actual = current_month()
    mes_sig = next_month(mes_actual)

    f_actual = (
        seasonality_df[seasonality_df["Mes"] == mes_actual]
        .rename(columns={"Factor_Estacional": "Factor_Estacional_Actual"})
        [["Código", "Factor_Estacional_Actual"]]
    )

    f_sig = (
        seasonality_df[seasonality_df["Mes"] == mes_sig]
        .rename(columns={"Factor_Estacional": "Factor_Estacional_Siguiente"})
        [["Código", "Factor_Estacional_Siguiente"]]
    )

    out = f_actual.merge(f_sig, on="Código", how="outer")
    out["Factor_Estacional_Actual"] = out["Factor_Estacional_Actual"].fillna(1.0)
    out["Factor_Estacional_Siguiente"] = out["Factor_Estacional_Siguiente"].fillna(1.0)

    if MESES_ANTICIPACION == 0:
        out["Factor_Estacional_Compra"] = out["Factor_Estacional_Actual"]
    else:
        out["Factor_Estacional_Compra"] = (
            PESO_MES_ACTUAL * out["Factor_Estacional_Actual"] +
            PESO_MES_SIGUIENTE * out["Factor_Estacional_Siguiente"]
        )

    return out[["Código", "Factor_Estacional_Compra"]]

=========================

SEGURIDAD DE REGRESION

=========================

def apply_regression_safety(final): final = final.copy()

    final["Meses_Historial"] = final["Meses_Historial"].fillna(0)
    final["Pred_Regresion_Usable"] = final["Pred_Regresion_Mensual"]

    final["Pred_Regresion_Usable"] = np.where(
        final["Meses_Historial"] >= MIN_MESES_PARA_REGRESION,
        final["Pred_Regresion_Usable"],
        final["Demanda_Mensual_Historica"]
    )

    limite_hist = np.where(
        final["Demanda_Mensual_Historica"] > 0,
        final["Demanda_Mensual_Historica"] * MAX_FACTOR_SOBRE_HISTORICO,
        np.nan
    )

    limite_v30d = np.where(
        final["V30D"] > 0,
        final["V30D"] * MAX_FACTOR_SOBRE_V30D,
        np.nan
    )

    limite_final = np.where(
        ~np.isnan(limite_hist) & ~np.isnan(limite_v30d),
        np.minimum(limite_hist, limite_v30d),
        np.where(~np.isnan(limite_hist), limite_hist, limite_v30d)
    )

    final["Pred_Regresion_Usable"] = np.where(
        ~np.isnan(limite_final),
        np.minimum(final["Pred_Regresion_Usable"], limite_final),
        final["Pred_Regresion_Usable"]
    )

    final["Pred_Regresion_Usable"] = final["Pred_Regresion_Usable"].clip(lower=0)

    return final

=========================

MODELO FINAL

=========================

def build_final_table(vs, hist): cost = build_cost(hist) school =
build_school_demand(hist) v04, v05 = build_v04_v05(hist)

    monthly, train = build_monthly_features(hist)
    model, feature_cols = train_global_regression(train)
    pred_reg = predict_next_month_per_sku(monthly, model, feature_cols)

    seasonality_full = build_seasonality(hist)
    seasonality_buy = build_current_seasonality_for_purchase(seasonality_full)

    final = vs.merge(school, on="Código", how="left")
    final = final.merge(cost, on="Código", how="left")
    final = final.merge(v04, on="Código", how="left")
    final = final.merge(v05, on="Código", how="left")
    final = final.merge(pred_reg, on="Código", how="left")
    final = final.merge(seasonality_buy, on="Código", how="left")

    final["V04_2025"] = final["V04_2025"].fillna(0)
    final["V05_2025"] = final["V05_2025"].fillna(0)
    final["Tipo"] = final["Tipo"].fillna("SIN_HISTORICO")

    final = fill_missing_costs_with_global_average(final, hist)

    final["Demanda_Mensual_Historica"] = final["Demanda_Mensual_Historica"].fillna(final["V30D"])
    final["Pred_Regresion_Mensual"] = final["Pred_Regresion_Mensual"].fillna(final["Demanda_Mensual_Historica"])
    final["Factor_Estacional_Compra"] = final["Factor_Estacional_Compra"].fillna(1.0)

    final["Factor_Estacional_Compra"] = np.where(
        final["V30D"] >= 3,
        np.maximum(final["Factor_Estacional_Compra"], 1.0),
        final["Factor_Estacional_Compra"]
    )

    final = apply_regression_safety(final)

    final["Demanda_Base_Modelo"] = (
        PESO_REGRESION * final["Pred_Regresion_Usable"] +
        PESO_V30D * final["V30D"]
    ).clip(lower=0)

    if USAR_ESTACIONALIDAD:
        final["Demanda_Ajustada_Estacional"] = final["Demanda_Base_Modelo"] * final["Factor_Estacional_Compra"]
    else:
        final["Demanda_Ajustada_Estacional"] = final["Demanda_Base_Modelo"]

    final["Demanda30"] = np.ceil(final["Demanda_Ajustada_Estacional"]).clip(lower=0)

    final["Demanda30"] = np.where(
        (final["V30D"] > 0) & (final["Demanda30"] == 0),
        np.ceil(final["V30D"] * 0.30),
        final["Demanda30"]
    )

    final["Compra_Base"] = final["Demanda30"] - final["Stock"]

    final["Compra_Base"] = np.where(
        final["Stock"] >= final["Demanda30"],
        0,
        final["Compra_Base"]
    )

    final["Compra_Base"] = final["Compra_Base"].clip(lower=0)

    final["Compra_Base"] = np.where(
        (final["Compra_Base"] == 0) &
        (final["V30D"] > MIN_ROTACION_V30D) &
        (final["Stock"] < final["Demanda30"]),
        COMPRA_MINIMA_UNIDAD,
        final["Compra_Base"]
    )

    final["Compra"] = final["Compra_Base"].apply(round_normal)
    final["Costo"] = final["Costo"].round(2)
    final["Importe"] = (final["Compra"] * final["Costo"]).round(2)

    final["Cobertura"] = np.where(
        final["Demanda30"] > 0,
        final["Stock"] / final["Demanda30"],
        1
    )

    def nivel(c):
        if c < 0.3:
            return "CRITICO"
        elif c < 0.8:
            return "MEDIO"
        else:
            return "SANO"

    final["Nivel"] = final["Cobertura"].apply(nivel)

    tabla = final[[
        "Código",
        "EAN",        # ← NUEVO
        "Nombre",
        "Compra",
        "Stock",
        "Demanda30",
        "V30D",
        "V04_2025",
        "V05_2025",
        "Costo",
        "Importe",
        "Nivel",
        "Tipo"
    ]].copy()

    tabla["Costo"] = tabla["Costo"].round(2)
    tabla["Importe"] = tabla["Importe"].round(2)

    tabla = tabla.sort_values("Importe", ascending=False).reset_index(drop=True)

    return tabla

=========================

UI

=========================

st.title(f”Agente de compras {APP_VERSION}“)

hist_file = st.file_uploader(“Histórico”, type=[“xlsx”]) erply_file =
st.file_uploader(“Erply”, type=[“xls”, “xlsx”, “html”])

if hist_file is None or erply_file is None: st.stop()

try: hist = pd.read_excel(hist_file) vs = read_erply(erply_file) hist =
prepare_hist(hist)

    tabla = build_final_table(vs, hist)

    m1, m2, m3 = st.columns(3)
    m1.metric("SKUs", len(tabla))
    m2.metric("SKUs Compra", int((tabla["Compra"] > 0).sum()))
    m3.metric("Importe Total", f"${tabla['Importe'].fillna(0).sum():,.2f}")

    st.markdown("### Tabla de compra")
    st.dataframe(tabla, use_container_width=True, height=650)

    st.download_button(
        "Descargar CSV",
        tabla.to_csv(index=False).encode("utf-8-sig"),
        "compra_v9_0.csv"
    )

except Exception as e: st.error(f”Error al procesar archivos: {e}“)
