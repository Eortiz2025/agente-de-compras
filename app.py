# app.py
# Streamlit: lee histórico y calcula ventas de Enero 2024 vs Enero 2025 (unidades e importe)

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ventas Enero 2024 vs 2025", layout="wide")
st.title("Histórico — Ventas Enero 2024 vs Enero 2025")

st.sidebar.header("1) Cargar histórico")
hist_file = st.sidebar.file_uploader(
    "Sube tu archivo histórico (.xlsx) con columnas: Código, Nombre, Año, Mes, Ventas, Importe",
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
# Leer y validar
# -------------------------
try:
    df = pd.read_excel(hist_file)
except Exception as e:
    st.error(f"No pude leer el Excel. Error: {e}")
    st.stop()

required = {"Código", "Nombre", "Año", "Mes", "Ventas"}
optional_importe = "Importe"

missing = required - set(df.columns)
if missing:
    st.error(f"Faltan columnas en el archivo: {sorted(missing)}")
    st.stop()

# Normalización / tipos
df = df.copy()
df["Código"] = df["Código"].astype(str).str.strip()
df["Nombre"] = df["Nombre"].astype(str).fillna("")
df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
df["Ventas"] = pd.to_numeric(df["Ventas"], errors="coerce").fillna(0)

has_importe = optional_importe in df.columns
if has_importe:
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce").fillna(0)

# -------------------------
# Cálculos
# -------------------------
def resumen(yy: int):
    sub = df[(df["Año"] == yy) & (df["Mes"] == mes)]
    unidades = float(sub["Ventas"].sum())
    importe = float(sub["Importe"].sum()) if has_importe else None
    skus = int(sub["Código"].nunique())
    renglones = int(len(sub))
    return unidades, importe, skus, renglones, sub

u_a, imp_a, skus_a, rows_a, sub_a = resumen(int(anio_a))
u_b, imp_b, skus_b, rows_b, sub_b = resumen(int(anio_b))

delta_u = u_b - u_a
pct_u = (delta_u / u_a * 100.0) if u_a != 0 else None

if has_importe:
    delta_imp = imp_b - imp_a
    pct_imp = (delta_imp / imp_a * 100.0) if imp_a != 0 else None

# -------------------------
# UI
# -------------------------
st.subheader(f"Resumen Mes {mes}: {int(anio_a)} vs {int(anio_b)}")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Unidades {int(anio_a)}", f"{u_a:,.0f}")
c2.metric(f"Unidades {int(anio_b)}", f"{u_b:,.0f}", f"{delta_u:,.0f}" + (f" ({pct_u:.2f}%)" if pct_u is not None else ""))

if has_importe:
    c3.metric(f"Importe {int(anio_a)}", f"{imp_a:,.2f}")
    c4.metric(f"Importe {int(anio_b)}", f"{imp_b:,.2f}", f"{delta_imp:,.2f}" + (f" ({pct_imp:.2f}%)" if pct_imp is not None else ""))
else:
    c3.metric(f"SKUs {int(anio_a)}", f"{skus_a:,}")
    c4.metric(f"SKUs {int(anio_b)}", f"{skus_b:,}")

st.caption(f"Renglones: {int(anio_a)}={rows_a:,} | {int(anio_b)}={rows_b:,} | SKUs únicos: {int(anio_a)}={skus_a:,} | {int(anio_b)}={skus_b:,}")

# Top por SKU para explicar diferencia
st.subheader("Top SKUs por unidades (mes seleccionado)")

agg_cols = ["Código", "Nombre"]
a_units = sub_a.groupby(agg_cols, as_index=False)["Ventas"].sum().rename(columns={"Ventas": f"Unidades_{int(anio_a)}"})
b_units = sub_b.groupby(agg_cols, as_index=False)["Ventas"].sum().rename(columns={"Ventas": f"Unidades_{int(anio_b)}"})

top = a_units.merge(b_units, on=agg_cols, how="outer").fillna(0)
top["Delta_unidades"] = top[f"Unidades_{int(anio_b)}"] - top[f"Unidades_{int(anio_a)}"]
top = top.sort_values("Delta_unidades", ascending=False)

st.write("**Suben más (Top 20):**")
st.dataframe(top.head(20), use_container_width=True, height=360)

st.write("**Bajan más (Top 20):**")
st.dataframe(top.sort_values("Delta_unidades", ascending=True).head(20), use_container_width=True, height=360)

# Export opcional
st.subheader("Descargar tabla comparativa (por SKU)")
csv = top.to_csv(index=False).encode("utf-8-sig")
st.download_button("Descargar CSV", data=csv, file_name=f"comparativo_mes{mes}_{int(anio_a)}_{int(anio_b)}.csv", mime="text/csv")
