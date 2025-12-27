import streamlit as st
import pandas as pd

st.set_page_config(page_title="MaxEne / MaxFeb (2024-2025)", layout="wide")

st.divider()
st.header("Máximos por SKU — MaxEne y MaxFeb (entre 2024 y 2025)")

hist_file = st.file_uploader(
    "Sube histórico (.xlsx) con columnas: Código, Nombre, Año, Mes, Ventas (Importe opcional)",
    type=["xlsx"]
)

if hist_file is None:
    st.info("Sube el archivo para continuar.")
    st.stop()

df = pd.read_excel(hist_file)

req = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = req - set(df.columns)
if missing:
    st.error(f"Faltan columnas: {sorted(missing)}")
    st.stop()

# Normalizar
df["Código"] = df["Código"].astype(str).str.strip()
df["Nombre"] = df["Nombre"].astype(str).fillna("")
df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
df["Ventas"] = pd.to_numeric(df["Ventas"], errors="coerce").fillna(0)

# Filtrar a 2024-2025 y meses Ene/Feb
base = df[df["Año"].isin([2024, 2025]) & df["Mes"].isin([1, 2])].copy()

# Max por SKU por (Año, Mes) (por si hay duplicados)
g = (
    base.groupby(["Código", "Nombre", "Año", "Mes"], as_index=False)["Ventas"]
    .max()
)

# Pivot a columnas (Año, Mes)
p = g.pivot_table(
    index=["Código", "Nombre"],
    columns=["Año", "Mes"],
    values="Ventas",
    aggfunc="max",
    fill_value=0
)

# Asegurar columnas aunque falten combinaciones
needed = [(2024, 1), (2025, 1), (2024, 2), (2025, 2)]
for col in needed:
    if col not in p.columns:
        p[col] = 0

out = p[needed].copy()
out.columns = ["Max_Ene_2024", "Max_Ene_2025", "Max_Feb_2024", "Max_Feb_2025"]
out = out.reset_index()

# Colapsar a lo que quieres ver
out["MaxEne"] = out[["Max_Ene_2024", "Max_Ene_2025"]].max(axis=1)
out["MaxFeb"] = out[["Max_Feb_2024", "Max_Feb_2025"]].max(axis=1)

# Tabla final (solo lo esencial)
final = out[["Código", "Nombre", "MaxEne", "MaxFeb"]].copy()

# KPIs
c1, c2, c3 = st.columns(3)
c1.metric("SKUs", f"{final['Código'].nunique():,}")
c2.metric("Suma MaxEne", f"{final['MaxEne'].sum():,.0f}")
c3.metric("Suma MaxFeb", f"{final['MaxFeb'].sum():,.0f}")

st.subheader("Tabla final (por SKU)")
st.dataframe(
    final.sort_values(["MaxEne", "MaxFeb"], ascending=False),
    use_container_width=True,
    height=560
)

# Descarga
csv = final.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="maxene_maxfeb_2024_2025_por_sku.csv",
    mime="text/csv"
)
