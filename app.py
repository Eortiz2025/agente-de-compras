import streamlit as st
import pandas as pd
import numpy as np

APP_VERSION = "2026-03-28-v6.2 REGRESION SEGURA SIN MULTIPLOS"

MIN_ROTACION_V30D = 3
COMPRA_MINIMA_UNIDAD = 1

# mezcla final
PESO_REGRESION = 0.70
PESO_V30D = 0.30

# reglas de seguridad para regresión
MIN_MESES_PARA_REGRESION = 3
MAX_FACTOR_SOBRE_HISTORICO = 2.5   # la regresión no puede ser > 2.5x demanda histórica fallback
MAX_FACTOR_SOBRE_V30D = 3.0        # la regresión no puede ser > 3x V30D si V30D existe

st.set_page_config(page_title="Agente de compras", layout="wide")


# =========================
# HELPERS
# =========================
def norm_code(s):
    return s.astype(str).str.strip().str.upper()


def round_normal(qty):
    if pd.isna(qty) or qty <= 0:
        return 0
    return int(np.ceil(qty))


# =========================
# REGRESION LINEAL CON NUMPY
# =========================
class NumpyLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        X_design = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self.feature_names_ = feature_names if feature_names is not None else []
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not self.is_fitted_:
            raise ValueError("El modelo no ha sido entrenado.")
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


# =========================
# ERPLY PARSER
# =========================
def read_erply(file):
    tables = pd.read_html(file, header=None)
    df = max(tables, key=lambda x: x.shape[0])

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
        "Nombre": df.iloc[:, 3].astype(str).fillna(""),
        "V30D": pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0),
        "Stock": pd.to_numeric(df.iloc[:, 6], errors="coerce").fillna(0),
    })

    out["Código"] = norm_code(out["Código"])
    return out


# =========================
# HISTORICO PREP
# =========================
def prepare_hist(hist):
    hist = hist.copy()
    hist["Código"] = norm_code(hist["Código"])
    hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)
    hist["Importe"] = pd.to_numeric(hist["Importe"], errors="coerce").fillna(0)
    hist["Año"] = pd.to_numeric(hist["Año"], errors="coerce").fillna(0).astype(int)
    hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce").fillna(0).astype(int)

    hist = hist[(hist["Mes"] >= 1) & (hist["Mes"] <= 12) & (hist["Año"] > 0)].copy()
    return hist


# =========================
# FEATURES MENSUALES
# =========================
def build_monthly_features(hist):
    monthly = (
        hist.groupby(["Código", "Año", "Mes"], as_index=False)
        .agg({
            "Ventas": "sum",
            "Importe": "sum"
        })
        .sort_values(["Código", "Año", "Mes"])
        .reset_index(drop=True)
    )

    monthly["Fecha"] = pd.to_datetime(
        monthly["Año"].astype(str) + "-" + monthly["Mes"].astype(str).str.zfill(2) + "-01"
    )

    monthly = monthly.sort_values(["Código", "Fecha"]).reset_index(drop=True)

    monthly["lag1"] = monthly.groupby("Código")["Ventas"].shift(1)
    monthly["lag2"] = monthly.groupby("Código")["Ventas"].shift(2)
    monthly["lag3"] = monthly.groupby("Código")["Ventas"].shift(3)
    monthly["ma3"] = monthly[["lag1", "lag2", "lag3"]].mean(axis=1)

    # mes cíclico para no meter sesgo lineal bruto
    monthly["Mes_sin"] = np.sin(2 * np.pi * monthly["Mes"] / 12)
    monthly["Mes_cos"] = np.cos(2 * np.pi * monthly["Mes"] / 12)

    train = monthly.dropna(subset=["lag1", "lag2", "lag3"]).copy()
    return monthly, train


def train_global_regression(train):
    feature_cols = ["lag1", "lag2", "lag3", "ma3", "Mes_sin", "Mes_cos"]

    if len(train) < 20:
        return None, feature_cols

    X = train[feature_cols].fillna(0).values
    y = train["Ventas"].fillna(0).values

    model = NumpyLinearRegression().fit(X, y, feature_names=feature_cols)
    return model, feature_cols


def predict_next_month_per_sku(monthly, model, feature_cols):
    if model is None:
        return pd.DataFrame(columns=["Código", "Pred_Regresion_Mensual", "Meses_Historial"])

    rows = []

    for codigo, g in monthly.groupby("Código"):
        g = g.sort_values("Fecha").reset_index(drop=True)

        meses_historial = int((g["Ventas"] > 0).sum())

        last1 = g["Ventas"].iloc[-1] if len(g) >= 1 else 0
        last2 = g["Ventas"].iloc[-2] if len(g) >= 2 else 0
        last3 = g["Ventas"].iloc[-3] if len(g) >= 3 else 0
        ma3 = np.mean([last1, last2, last3])

        last_fecha = g["Fecha"].iloc[-1]
        next_month = (last_fecha.month % 12) + 1

        mes_sin = np.sin(2 * np.pi * next_month / 12)
        mes_cos = np.cos(2 * np.pi * next_month / 12)

        X_pred = pd.DataFrame([{
            "lag1": last1,
            "lag2": last2,
            "lag3": last3,
            "ma3": ma3,
            "Mes_sin": mes_sin,
            "Mes_cos": mes_cos
        }])

        pred = model.predict(X_pred[feature_cols].fillna(0).values)[0]
        pred = max(0, pred)

        rows.append({
            "Código": codigo,
            "Pred_Regresion_Mensual": pred,
            "Meses_Historial": meses_historial
        })

    return pd.DataFrame(rows)


# =========================
# COSTO
# =========================
def build_cost(hist):
    # costo por SKU usando primero 2025
    cost_2025 = (
        hist[hist["Año"] == 2025]
        .groupby("Código")
        .agg({"Ventas": "sum", "Importe": "sum"})
        .reset_index()
    )
    cost_2025["Costo_2025"] = np.where(
        cost_2025["Ventas"] > 0,
        cost_2025["Importe"] / cost_2025["Ventas"],
        np.nan
    )

    # fallback: todos los años
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


# =========================
# DEMANDA ESCOLAR COMO APOYO
# =========================
def build_school_demand(hist):
    hist_escolar = hist[hist["Mes"].between(4, 10)]

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


# =========================
# COLUMNAS ABRIL / MAYO 2025
# =========================
def build_v04_v05(hist):
    v04 = (
        hist[(hist["Año"] == 2025) & (hist["Mes"] == 4)]
        .groupby("Código")["Ventas"]
        .sum()
        .rename("V04_2025")
    )

    v05 = (
        hist[(hist["Año"] == 2025) & (hist["Mes"] == 5)]
        .groupby("Código")["Ventas"]
        .sum()
        .rename("V05_2025")
    )

    return v04, v05


# =========================
# FALLBACK GLOBAL DE COSTO
# =========================
def fill_missing_costs_with_global_average(final, hist):
    total_ventas = hist["Ventas"].sum()
    total_importe = hist["Importe"].sum()
    global_cost = (total_importe / total_ventas) if total_ventas > 0 else 0

    final["Costo"] = final["Costo"].fillna(global_cost).fillna(0)
    return final


# =========================
# SEGURIDAD DE REGRESION
# =========================
def apply_regression_safety(final):
    final = final.copy()

    final["Meses_Historial"] = final["Meses_Historial"].fillna(0)

    # Base segura inicial
    final["Pred_Regresion_Usable"] = final["Pred_Regresion_Mensual"]

    # Si no hay suficientes meses, usar histórico
    final["Pred_Regresion_Usable"] = np.where(
        final["Meses_Historial"] >= MIN_MESES_PARA_REGRESION,
        final["Pred_Regresion_Usable"],
        final["Demanda_Mensual_Historica"]
    )

    # Tope contra histórico
    limite_hist = np.where(
        final["Demanda_Mensual_Historica"] > 0,
        final["Demanda_Mensual_Historica"] * MAX_FACTOR_SOBRE_HISTORICO,
        np.nan
    )

    # Tope contra V30D
    limite_v30d = np.where(
        final["V30D"] > 0,
        final["V30D"] * MAX_FACTOR_SOBRE_V30D,
        np.nan
    )

    # Si existen ambos límites, usar el menor
    limite_final = np.where(
        ~np.isnan(limite_hist) & ~np.isnan(limite_v30d),
        np.minimum(limite_hist, limite_v30d),
        np.where(~np.isnan(limite_hist), limite_hist, limite_v30d)
    )

    # Aplicar tope cuando exista
    final["Pred_Regresion_Usable"] = np.where(
        ~np.isnan(limite_final),
        np.minimum(final["Pred_Regresion_Usable"], limite_final),
        final["Pred_Regresion_Usable"]
    )

    # nunca negativa
    final["Pred_Regresion_Usable"] = final["Pred_Regresion_Usable"].clip(lower=0)

    return final


# =========================
# MODELO FINAL
# =========================
def build_final_table(vs, hist):
    cost = build_cost(hist)
    school = build_school_demand(hist)
    v04, v05 = build_v04_v05(hist)

    monthly, train = build_monthly_features(hist)
    model, feature_cols = train_global_regression(train)
    pred_reg = predict_next_month_per_sku(monthly, model, feature_cols)

    final = vs.merge(school, on="Código", how="left")
    final = final.merge(cost, on="Código", how="left")
    final = final.merge(v04, on="Código", how="left")
    final = final.merge(v05, on="Código", how="left")
    final = final.merge(pred_reg, on="Código", how="left")

    final["V04_2025"] = final["V04_2025"].fillna(0)
    final["V05_2025"] = final["V05_2025"].fillna(0)
    final["Tipo"] = final["Tipo"].fillna("SIN_HISTORICO")

    final = fill_missing_costs_with_global_average(final, hist)

    # fallback histórico
    final["Demanda_Mensual_Historica"] = final["Demanda_Mensual_Historica"].fillna(final["V30D"])

    # fallback regresión
    final["Pred_Regresion_Mensual"] = final["Pred_Regresion_Mensual"].fillna(final["Demanda_Mensual_Historica"])

    # aplicar seguridades
    final = apply_regression_safety(final)

    # mezcla final
    final["Demanda30"] = np.ceil(
        PESO_REGRESION * final["Pred_Regresion_Usable"] +
        PESO_V30D * final["V30D"]
    ).clip(lower=0)

    # compra base
    final["Compra_Base"] = (final["Demanda30"] - final["Stock"]).clip(lower=0)

    # compra mínima si hay movimiento reciente
    final["Compra_Base"] = np.where(
        (final["Compra_Base"] == 0) & (final["V30D"] > MIN_ROTACION_V30D),
        COMPRA_MINIMA_UNIDAD,
        final["Compra_Base"]
    )

    # sin múltiplos
    final["Compra"] = final["Compra_Base"].apply(round_normal)

    # importe
    final["Importe"] = final["Compra"] * final["Costo"]

    # cobertura
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
        "Tipo",
        "Pred_Regresion_Mensual",
        "Pred_Regresion_Usable",
        "Demanda_Mensual_Historica",
        "Meses_Historial",
        "Cobertura"
    ]].copy()

    tabla = tabla.sort_values("Importe", ascending=False).reset_index(drop=True)

    return tabla, final, model, feature_cols


# =========================
# UI
# =========================
st.title(f"Agente de compras {APP_VERSION}")

st.markdown("""
### Archivos requeridos
- **Histórico**: archivo Excel con columnas `Código`, `Ventas`, `Importe`, `Año`, `Mes`
- **Erply**: archivo `.xls`
""")

hist_file = st.file_uploader("Histórico", type=["xlsx"])
erply_file = st.file_uploader("Erply", type=["xls", "xlsx", "html"])

if hist_file is None or erply_file is None:
    st.stop()

try:
    hist = pd.read_excel(hist_file)
    vs = read_erply(erply_file)
    hist = prepare_hist(hist)

    tabla, final, model, feature_cols = build_final_table(vs, hist)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SKUs", len(tabla))
    m2.metric("SKUs Compra", int((tabla["Compra"] > 0).sum()))
    m3.metric("Importe Total", f"${tabla['Importe'].fillna(0).sum():,.0f}")
    m4.metric("Demanda Total", f"{tabla['Demanda30'].fillna(0).sum():,.0f}")

    st.markdown("### Tabla de compra")
    st.dataframe(tabla, use_container_width=True, height=650)

    st.markdown("### Top 30 por importe")
    st.dataframe(tabla.head(30), use_container_width=True, height=500)

    st.download_button(
        "Descargar CSV",
        tabla.to_csv(index=False).encode("utf-8-sig"),
        "compra_v6_2_regresion_segura_sin_multiplos.csv"
    )

    st.download_button(
        "Descargar detalle completo",
        final.to_csv(index=False).encode("utf-8-sig"),
        "compra_v6_2_regresion_segura_sin_multiplos_detalle.csv"
    )

    st.markdown("### Cómo calcula ahora")
    st.write("""
1. Lee **Código, Nombre, V30D y Stock** desde Erply.
2. Lee **Ventas, Importe, Año y Mes** desde el histórico.
3. Arma series mensuales por SKU.
4. Entrena una regresión lineal global con:
   - `lag1`
   - `lag2`
   - `lag3`
   - `ma3`
   - `Mes_sin`
   - `Mes_cos`
5. Predice la demanda mensual siguiente por SKU.
6. Si el SKU tiene menos de 3 meses con historial, usa fallback histórico.
7. Aplica topes de seguridad para evitar sobrepredicción.
8. Mezcla:
   - 70% regresión usable
   - 30% V30D
9. Resta stock para calcular compra.
10. Ya no usa múltiplos.
""")

    if model is not None:
        coef_df = pd.DataFrame({
            "Variable": feature_cols,
            "Coeficiente": model.coef_
        })
        coef_df.loc[len(coef_df)] = ["Intercepto", model.intercept_]

        st.markdown("### Coeficientes del modelo")
        st.dataframe(coef_df, use_container_width=True)
    else:
        st.warning("No hubo suficientes datos para entrenar la regresión. Se usó fallback histórico.")

except Exception as e:
    st.error(f"Error al procesar archivos: {e}")
