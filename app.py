# app.py
# Base 1 — Histórico
# Comparativo por mes (por defecto: Enero 2024 vs Enero 2025)
# Lee histórico y muestra resumen + detalle por SKU

import pandas as pd
import streamlit as st

# -------------------------
# Configuración
# -------------------------
st.set_page_config(page_title="Histórico — Comparativos", layout="wide")
st.title("Histórico — Comparativo por mes")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("1) Cargar histórico")
hist_file = st.sidebar.file_uploader(
    "Histórico (.xlsx) con columnas: Código, Nombre, Año, Mes, Ventas (Importe opcional)",
    type=["xlsx"]
)

st.sidebar.header("2) Parámetros")
mes = st.sidebar.selectbox("Mes", options=list(range(1, 13)), index=0)  # 1 = Enero
anio_a = st.sidebar.number_input("Año A", min_value=2000, max_value=2100, value=2024, step=1)
anio_b = st.sidebar.number_input("Año B", min_value=2000, max_value=2100, value=2025, step=1)

if hist_file is None:
    st.info("Sube el archivo histórico para continuar.")
    st.stop()

# -------------------------
# Leer y validar archivo
# -------------------------
try:
    df = pd.read_excel(hist_file)
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

required_cols = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Faltan columnas requeridas: {sorted(missing)}")
    st.stop()

# Normalizar datos
df = df.copy()
df["Código"] = df["Código"].astype(str).str.strip()
df["Nombre"] = df["Nombre"].astype(str).fillna("")
df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
df["Ventas"] = pd.to_numeric(df["Ventas"], errors="coerce").fillna(0)

has_importe = "Importe" in df.columns
if has_importe:
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce").fillna(0)

# -------------------------
# Filtrar mes y años
# -------------------------
sub_a = df[(df["Año"] == int(anio_a)) & (df["Mes"] == int(mes))]
sub_b = df[(df["Año"] == int(anio_b)) & (df["Mes"] == int(mes))]

# -------------------------
# Cálculos resumen
# -------------------------
u_a = float(sub_a["Ventas"].sum())
u_b = float(sub_b["Ventas"].sum())
delta_u = u_b - u_a
pct_u = (delta_u / u_a * 100) if u_a != 0 else None

if has_importe:
    imp_a = float(sub_a["Importe"].sum())
    imp_b = float(sub_b["Importe"].sum())
    delta_imp = imp_b - imp_a
    pct_imp = (delta_imp / imp_a * 100) if imp_a != 0 else None

# -------------------------
# UI — Resumen
# -------------------------
st.subheader(f"Resumen — Mes {mes} | {int(anio_a)} vs {int(anio_b)}")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Unidades {int(anio_a)}", f"{u_a:,.0f}")
c2.metric(
    f"Unidades {int(anio_b)}",
    f"{u_b:,.0f}",
    f"{delta_u:,.0f}" + (f" ({pct_u:.2f}%)" if pct_u is not None else "")
)

if has_importe:
    c3.metric(f"Importe {int(anio_a)}", f"${imp_a:,.2f}")
    c4.metric(
        f"Importe {int(anio_b)}",
        f"${imp_b:,.2f}",
        f"${delta_imp:,.2f}" + (f" ({pct_imp:.2f}%)" if pct_imp is not None else "")
    )
else:
    c3.metric("SKUs únicos (A)", f"{sub_a['Código'].nunique():,}")
    c4.metric("SKUs únicos (B)", f"{sub_b['Código'].nunique():,}")

st.caption(
    f"Renglones: {int(anio_a)}={len(sub_a):,} | {int(anio_b)}={len(sub_b):,} | "
    f"SKUs únicos: {int(anio_a)}={sub_a['Código'].nunique():,} | {int(anio_b)}={sub_b['Código'].nunique():,}"
)

# -------------------------
# Detalle por SKU
# -------------------------
st.subheader("Detalle por SKU (unidades)")

a_units = (
    sub_a.groupby(["Código", "Nombre"], as_index=False)["Ventas"]
    .sum()
    .rename(columns={"Ventas": f"Unidades_{int(anio_a)}"})
)

b_units = (
    sub_b.groupby(["Código", "Nombre"], as_index=False)["Ventas"]
    .sum()
    .rename(columns={"Ventas": f"Unidades_{int(anio_b)}"})
)

detalle = a_units.merge(b_units, on=["Código", "Nombre"], how="outer").fillna(0)
detalle["Delta_unidades"] = detalle[f"Unidades_{int(anio_b)}"] - detalle[f"Unidades_{int(anio_a)}"]

cols = ["Código", "Nombre", f"Unidades_{int(anio_a)}", f"Unidades_{int(anio_b)}", "Delta_unidades"]
detalle = detalle[cols]

tab1, tab2 = st.tabs(["Bajan más", "Suben más"])
with tab1:
    st.dataframe(detalle.sort_values("Delta_unidades", ascending=True).head(50),
                 use_container_width=True, height=520)
with tab2:
    st.dataframe(detalle.sort_values("Delta_unidades", ascending=False).head(50),
                 use_container_width=True, height=520)

# -------------------------
# Descarga
# -------------------------
st.subheader("Descargar detalle por SKU")
csv = detalle.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name=f"detalle_mes{int(mes)}_{int(anio_a)}_vs_{int(anio_b)}.csv",
    mime="text/csv"
)
