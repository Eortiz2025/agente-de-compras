import streamlit as st
import pandas as pd

st.set_page_config(page_title="Máximos Ene/Feb 2024-2025", layout="wide")

st.divider()
st.header("Máximos de ventas por SKU — Ene y Feb (2024 y 2025)")

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

df["Código"] = df["Código"].astype(str).str.strip()
df["Nombre"] = df["Nombre"].astype(str).fillna("")
df["Año"] = pd.to_numeric(df["Año"], errors="coerce")
df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
df["Ventas"] = pd.to_numeric(df["Ventas"], errors="coerce").fillna(0)

# Filtrar solo Ene/Feb y años 2024/2025
base = df[df["Año"].isin([2024, 2025]) & df["Mes"].isin([1, 2])].copy()

# Si tu histórico es mensual por SKU, el "max" por (Año, Mes) suele ser igual al valor.
# Aun así, calculamos MAX por seguridad (por si hay duplicados).
g = (
    base.groupby(["Código", "Nombre", "Año", "Mes"], as_index=False)["Ventas"]
    .max()
)

# Pivot a columnas
p = g.pivot_table(
    index=["Código", "Nombre"],
    columns=["Año", "Mes"],
    values="Ventas",
    aggfunc="max",
    fill_value=0
)

# Aplanar nombres de columnas
def colname(y, m):
    mm = "Ene" if m == 1 else "Feb"
    return f"Max_{mm}_{y}"

# Asegurar columnas aunque falte alguna combinación
cols_needed = [(2024, 1), (2025, 1), (2024, 2), (2025, 2)]
for c in cols_needed:
    if c not in p.columns:
        p[c] = 0

out = p[cols_needed].copy()
out.columns = [colname(y, m) for (y, m) in cols_needed]
out = out.reset_index()

# Extras útiles
out["Max_Ene_2425"] = out[["Max_Ene_2024", "Max_Ene_2025"]].max(axis=1)
out["Max_Feb_2425"] = out[["Max_Feb_2024", "Max_Feb_2025"]].max(axis=1)

# Mostrar
c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs", f"{out['Código'].nunique():,}")
c2.metric("Total Max Ene 2024", f"{out['Max_Ene_2024'].sum():,.0f}")
c3.metric("Total Max Ene 2025", f"{out['Max_Ene_2025'].sum():,.0f}")
c4.metric("Total Max Feb (24/25 máx)", f"{out['Max_Feb_2425'].sum():,.0f}")

st.subheader("Tabla (por SKU)")
st.dataframe(
    out.sort_values(["Max_Ene_2425", "Max_Feb_2425"], ascending=False),
    use_container_width=True,
    height=560
)

# Descarga CSV
csv = out.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="max_ene_feb_2024_2025_por_sku.csv",
    mime="text/csv"
)
