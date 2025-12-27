# app.py
# Streamlit app: Compra sugerida 30 días (Ene/Feb máx + V30D)
# Reglas:
# 1) Ene_max y Feb_max (máximo histórico por mes)
# 2) Demanda_hist = wEne*Ene_max + wFeb*Feb_max  (default 90/10)
# 3) Integrar V30D: Demanda_final = (1-α)*Demanda_hist + α*V30D  (α default 0.30)
# 4) (Opcional) Cap de V30D: V30D_cap ∈ [0.70*Demanda_hist, 1.30*Demanda_hist]
# 5) Compra = max(0, Demanda_final - Stock)

import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, timedelta

st.set_page_config(page_title="Compra sugerida 30 días", layout="wide")

st.title("Compra sugerida (30 días) — Ene/Feb Máx + V30D")

# -------------------------
# Helpers
# -------------------------
def norm_code(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def compute_weights_real_days(op_date: date, horizon_days: int = 30):
    """
    Calcula pesos por días reales por mes en la ventana [op_date, op_date + horizon_days).
    Devuelve dict {(year, month): weight}
    """
    start = op_date
    end = op_date + timedelta(days=horizon_days)
    total = (end - start).days
    cur = start
    counts = {}
    while cur < end:
        key = (cur.year, cur.month)
        counts[key] = counts.get(key, 0) + 1
        cur += timedelta(days=1)
    weights = {k: v / total for k, v in counts.items()}
    return weights

def excel_bytes(df: pd.DataFrame, sheet_name="Compra"):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    bio.seek(0)
    return bio.getvalue()

# -------------------------
# Inputs
# -------------------------
st.sidebar.header("1) Cargar archivos")

hist_file = st.sidebar.file_uploader(
    "Histórico 24+ meses (Excel) — debe incluir columnas: Código, Nombre, Mes, Ventas",
    type=["xlsx", "xls"]
)

v30_stock_file = st.sidebar.file_uploader(
    "V30D + Stock (Excel/CSV) — columnas mínimas: Código, V30D, Stock (Nombre opcional)",
    type=["xlsx", "xls", "csv"]
)

st.sidebar.header("2) Parámetros")

use_real_day_weights = st.sidebar.checkbox("Usar pesos por días reales (recomendado si cambias la fecha)", value=False)

op_date = st.sidebar.date_input("Fecha de operación", value=date.today())
horizon_days = st.sidebar.number_input("Horizonte (días)", min_value=7, max_value=90, value=30, step=1)

if use_real_day_weights:
    wmap = compute_weights_real_days(op_date, int(horizon_days))
    # Para tu regla original (Ene/Feb), nos quedamos con pesos de los meses tocados.
    # Si atraviesa más de 2 meses, se usa la distribución real.
else:
    # Regla fija validada: 90% enero / 10% febrero
    # (Esto es "modo libreta/ene-feb" tradicional; si el cálculo cae en otro mes,
    # puedes activar pesos reales arriba.)
    wmap = {(op_date.year, 1): 0.90, (op_date.year, 2): 0.10}

alpha_default = st.sidebar.slider("α default (peso V30D)", 0.0, 1.0, 0.30, 0.05)

use_v30_cap = st.sidebar.checkbox("Aplicar freno a V30D (cap 0.70x–1.30x vs demanda histórica)", value=True)

cap_low = st.sidebar.slider("Cap inferior (x Demanda_hist)", 0.30, 1.00, 0.70, 0.05)
cap_high = st.sidebar.slider("Cap superior (x Demanda_hist)", 1.00, 2.00, 1.30, 0.05)

st.sidebar.header("3) Overrides de α (opcional)")
critical_skus = st.sidebar.text_area(
    "SKUs críticos (α=0.40) — uno por línea",
    value=""
).strip().splitlines()

slow_skus = st.sidebar.text_area(
    "SKUs lentos/intermitentes (α=0.15) — uno por línea",
    value=""
).strip().splitlines()

alpha_critical = 0.40
alpha_slow = 0.15

critical_set = set(map(norm_code, critical_skus)) if critical_skus != [""] else set()
slow_set = set(map(norm_code, slow_skus)) if slow_skus != [""] else set()

# -------------------------
# Main logic
# -------------------------
if not hist_file or not v30_stock_file:
    st.info("Carga el **Histórico** y el archivo de **V30D + Stock** para calcular la compra.")
    st.stop()

# Load historical
hist = pd.read_excel(hist_file) if hist_file.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(hist_file)
need_hist_cols = {"Código", "Nombre", "Mes", "Ventas"}
missing = need_hist_cols - set(hist.columns)
if missing:
    st.error(f"Histórico: faltan columnas: {sorted(missing)}")
    st.stop()

hist = hist.copy()
hist["Código"] = hist["Código"].map(norm_code)
hist["Nombre"] = hist["Nombre"].astype(str)
hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce").astype("Int64")
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)

# Load V30D + Stock
if v30_stock_file.name.lower().endswith(("xlsx", "xls")):
    vs = pd.read_excel(v30_stock_file)
else:
    vs = pd.read_csv(v30_stock_file)

need_vs_cols = {"Código", "V30D", "Stock"}
missing2 = need_vs_cols - set(vs.columns)
if missing2:
    st.error(f"V30D+Stock: faltan columnas: {sorted(missing2)}")
    st.stop()

vs = vs.copy()
vs["Código"] = vs["Código"].map(norm_code)
vs["V30D"] = pd.to_numeric(vs["V30D"], errors="coerce").fillna(0)
vs["Stock"] = pd.to_numeric(vs["Stock"], errors="coerce").fillna(0)

# En caso de que venga "Nombre" en vs, úsalo; si no, se toma del histórico.
if "Nombre" not in vs.columns:
    vs["Nombre"] = ""

# 1) Máximos por mes (Ene y Feb) por SKU
ene_max = (
    hist.loc[hist["Mes"] == 1]
    .groupby(["Código"], as_index=False)["Ventas"].max()
    .rename(columns={"Ventas": "Ene_max"})
)

feb_max = (
    hist.loc[hist["Mes"] == 2]
    .groupby(["Código"], as_index=False)["Ventas"].max()
    .rename(columns={"Ventas": "Feb_max"})
)

# Nombre “canónico” por SKU desde histórico (último no importa, solo para etiqueta)
name_map = (
    hist.groupby("Código", as_index=False)["Nombre"]
    .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "")
    .rename(columns={"Nombre": "Nombre_hist"})
)

base = (
    vs.merge(name_map, on="Código", how="left")
      .merge(ene_max, on="Código", how="left")
      .merge(feb_max, on="Código", how="left")
)

base["Ene_max"] = base["Ene_max"].fillna(0)
base["Feb_max"] = base["Feb_max"].fillna(0)

# Resolver nombre final
base["Nombre_final"] = base["Nombre"].where(base["Nombre"].astype(str).str.strip() != "", base["Nombre_hist"])
base["Nombre_final"] = base["Nombre_final"].fillna("")

# 2) Demanda histórica
# Si el usuario activa pesos reales, se usan los meses involucrados en la ventana.
if use_real_day_weights:
    # Calcula demanda como suma (peso_mes * max_mes) para los meses tocados.
    # Para meses distintos a Ene/Feb, el max mensual no está calculado aquí.
    # Entonces: en modo pesos reales, solo aplicamos a Ene/Feb si la ventana toca esos meses;
    # si toca otros meses, los ignoramos (peso 0) y avisamos.
    w_ene = 0.0
    w_feb = 0.0
    for (yy, mm), w in wmap.items():
        if mm == 1:
            w_ene += w
        elif mm == 2:
            w_feb += w
    if (sum(wmap.values()) > 0) and (w_ene + w_feb < 0.999):
        st.warning("La ventana de días reales toca meses fuera de Ene/Feb. "
                   "En este modelo (Ene/Feb) esos días se ignoran. "
                   "Si quieres, ampliamos el modelo a 12 meses.")
    base["wEne"] = w_ene
    base["wFeb"] = w_feb
else:
    base["wEne"] = 0.90
    base["wFeb"] = 0.10

base["Demanda_hist"] = (base["wEne"] * base["Ene_max"] + base["wFeb"] * base["Feb_max"])

# 3) α por SKU
def alpha_for_sku(code: str) -> float:
    if code in critical_set:
        return alpha_critical
    if code in slow_set:
        return alpha_slow
    return alpha_default

base["alpha"] = base["Código"].apply(alpha_for_sku)

# 4) V30D con cap opcional
if use_v30_cap:
    low = cap_low * base["Demanda_hist"]
    high = cap_high * base["Demanda_hist"]
    base["V30D_cap"] = base["V30D"].clip(lower=low, upper=high)
else:
    base["V30D_cap"] = base["V30D"]

# 5) Demanda final + Compra
base["Demanda_final"] = (1 - base["alpha"]) * base["Demanda_hist"] + base["alpha"] * base["V30D_cap"]

# Redondeo: unidades enteras
base["Demanda_hist_r"] = base["Demanda_hist"].round().astype(int)
base["Demanda_final_r"] = base["Demanda_final"].round().astype(int)

base["Compra"] = (base["Demanda_final_r"] - base["Stock"]).clip(lower=0).round().astype(int)

# Output: solo compra > 0 (como vienes trabajando)
compra_df = base.loc[base["Compra"] > 0, [
    "Código",
    "Nombre_final",
    "Stock",
    "V30D",
    "Ene_max",
    "Feb_max",
    "wEne",
    "wFeb",
    "Demanda_hist_r",
    "alpha",
    "V30D_cap",
    "Demanda_final_r",
    "Compra"
]].rename(columns={
    "Nombre_final": "Nombre",
    "Demanda_hist_r": "Demanda_hist",
    "Demanda_final_r": "Demanda_final"
}).sort_values(["Compra", "Demanda_final"], ascending=[False, False])

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs en archivo V30D+Stock", len(vs))
c2.metric("SKUs con compra > 0", len(compra_df))
c3.metric("Unidades a comprar (total)", int(compra_df["Compra"].sum()) if len(compra_df) else 0)
c4.metric("α default", alpha_default)

st.subheader("Compra sugerida (solo Compra > 0)")
st.dataframe(compra_df, use_container_width=True, height=520)

# Download
st.subheader("Descargar")
fname = f"Compra_sugerida_{op_date.isoformat()}_{int(horizon_days)}d.xlsx"
st.download_button(
    "Descargar Excel",
    data=excel_bytes(compra_df, sheet_name="Compra"),
    file_name=fname,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("Ver reglas activas"):
    st.write({
        "Fecha_operación": str(op_date),
        "Horizonte_días": int(horizon_days),
        "Pesos": "Días reales" if use_real_day_weights else "Fijo 90% Ene / 10% Feb",
        "Cap_V30D": use_v30_cap,
        "Cap_low": cap_low if use_v30_cap else None,
        "Cap_high": cap_high if use_v30_cap else None,
        "alpha_default": alpha_default,
        "alpha_critical": alpha_critical,
        "alpha_slow": alpha_slow,
        "SKUs_críticos": len(critical_set),
        "SKUs_lentos": len(slow_set),
    })
