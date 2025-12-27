# app.py
# FORZAR LIMPIEZA DE CACHE EN STREAMLIT CLOUD
import streamlit as st

# 🔥 LIMPIEZA FORZADA (TEMPORAL)
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

import pandas as pd

st.set_page_config(page_title="Ventas Enero 2024 vs 2025", layout="wide")
st.title("Histórico — Ventas Enero 2024 vs Enero 2025")

# -------------------------
# Cargar archivo
# -------------------------
st.sidebar.header("Cargar histórico")
hist_file = st.sidebar.file_uploader(
    "Archivo histórico (.xlsx) con columnas: Código, Nombre, Año, Mes, Ventas, Importe",
    type=["xlsx"]
)

if hist_file is None:
    st.info("Sube el archivo histórico para continuar.")
    st.stop()

# -------------------------
# Leer archivo
# -------------------------
df = pd.read_excel(hist_file)

required_cols = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Faltan columnas: {sorted(missing)}")
    st.stop()

# Normalizar
df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
df["Ventas"] = pd.to_numeric(df["Ventas"], errors="coerce").fillna(0)
if "Importe" in df.columns:
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce").fillna(0)

# -------------------------
# Filtros Enero
# -------------------------
ene_2024 = df[(df["Año"] == 2024) & (df["Mes"] == 1)]
ene_2025 = df[(df["Año"] == 2025) & (df["Mes"] == 1)]

u24 = ene_2024["Ventas"].sum()
u25 = ene_2025["Ventas"].sum()

i24 = ene_2024["Importe"].sum() if "Importe" in df.columns else None
i25 = ene_2025["Importe"].sum() if "Importe" in df.columns else None

# -------------------------
# Resultados
# -------------------------
st.subheader("Resultado Enero")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unidades Ene 2024", f"{u24:,.0f}")
c2.metric("Unidades Ene 2025", f"{u25:,.0f}", f"{u25 - u24:,.0f}")

if i24 is not None:
    c3.metric("Importe Ene 2024", f"${i24:,.2f}")
    c4.metric("Importe Ene 2025", f"${i25:,.2f}", f"${i25 - i24:,.2f}")

# -------------------------
# Detalle por SKU
# -------------------------
st.subheader("Detalle por SKU (Enero)")

sku24 = ene_2024.groupby(["Código", "Nombre"], as_index=False)["Ventas"].sum().rename(columns={"Ventas": "Ene_2024"})
sku25 = ene_2025.groupby(["Código", "Nombre"], as_index=False)["Ventas"].sum().rename(columns={"Ventas": "Ene_2025"})

detalle = sku24.merge(sku25, on=["Código", "Nombre"], how="outer").fillna(0)
detalle["Diferencia"] = detalle["Ene_2025"] - detalle["Ene_2024"]

st.dataframe(detalle.sort_values("Diferencia"), use_container_width=True)
