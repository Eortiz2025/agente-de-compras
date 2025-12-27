import streamlit as st
import pandas as pd

# ✅ Marca visible para confirmar que Streamlit está usando ESTE archivo
st.set_page_config(page_title="PRUEBA DEPLOY", layout="wide")
st.title("🚨 CAMBIO DE PRUEBA: 26-DIC-2025 999 🚨")
st.write("Si ves este texto, Streamlit sí está leyendo este app.py del repo/branch correcto.")

st.divider()
st.header("Comparativo Enero 2024 vs Enero 2025 (desde histórico)")

hist_file = st.file_uploader(
    "Sube tu histórico (.xlsx) con columnas: Código, Nombre, Año, Mes, Ventas (Importe opcional)",
    type=["xlsx"]
)

if hist_file is None:
    st.info("Sube el archivo para continuar.")
    st.stop()

df = pd.read_excel(hist_file)

# Validación mínima
req = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = req - set(df.columns)
if missing:
    st.error(f"Faltan columnas: {sorted(missing)}")
    st.stop()

# Normalizar tipos
df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
df["Ventas"] = pd.to_numeric(df["Ventas"], errors="coerce").fillna(0)

has_importe = "Importe" in df.columns
if has_importe:
    df["Importe"] = pd.to_numeric(df["Importe"], errors="coerce").fillna(0)

# Filtrar enero
ene_2024 = df[(df["Año"] == 2024) & (df["Mes"] == 1)]
ene_2025 = df[(df["Año"] == 2025) & (df["Mes"] == 1)]

u24 = float(ene_2024["Ventas"].sum())
u25 = float(ene_2025["Ventas"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unidades Ene 2024", f"{u24:,.0f}")
c2.metric("Unidades Ene 2025", f"{u25:,.0f}", f"{u25-u24:,.0f}")

if has_importe:
    i24 = float(ene_2024["Importe"].sum())
    i25 = float(ene_2025["Importe"].sum())
    c3.metric("Importe Ene 2024", f"${i24:,.2f}")
    c4.metric("Importe Ene 2025", f"${i25:,.2f}", f"${i25-i24:,.2f}")
else:
    c3.metric("Renglones Ene 2024", f"{len(ene_2024):,}")
    c4.metric("Renglones Ene 2025", f"{len(ene_2025):,}")

st.subheader("Detalle por SKU (Enero)")
sku24 = ene_2024.groupby(["Código", "Nombre"], as_index=False)["Ventas"].sum().rename(columns={"Ventas": "Ene_2024"})
sku25 = ene_2025.groupby(["Código", "Nombre"], as_index=False)["Ventas"].sum().rename(columns={"Ventas": "Ene_2025"})

detalle = sku24.merge(sku25, on=["Código", "Nombre"], how="outer").fillna(0)
detalle["Diferencia"] = detalle["Ene_2025"] - detalle["Ene_2024"]
st.dataframe(detalle.sort_values("Diferencia"), use_container_width=True, height=520)
