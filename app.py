import streamlit as st
import pandas as pd

st.set_page_config(page_title="MaxEne/MaxFeb + V30D", layout="wide")

st.divider()
st.header("Unión: Histórico (MaxEne/MaxFeb) + V30D + Stock")

# -------------------------
# Uploaders
# -------------------------
colL, colR = st.columns(2)

with colL:
    hist_file = st.file_uploader(
        "1) Sube HISTÓRICO (.xlsx) — columnas: Código, Nombre, Año, Mes, Ventas",
        type=["xlsx"]
    )

with colR:
    v30_file = st.file_uploader(
        "2) Sube V30D + Stock (.xlsx/.csv) — columnas mínimas: Código, V30D, Stock (Nombre opcional)",
        type=["xlsx", "csv"]
    )

if hist_file is None or v30_file is None:
    st.info("Sube ambos archivos para continuar: Histórico y V30D+Stock.")
    st.stop()

# -------------------------
# Leer histórico y validar
# -------------------------
hist = pd.read_excel(hist_file)

req_hist = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = req_hist - set(hist.columns)
if missing:
    st.error(f"Histórico: faltan columnas {sorted(missing)}")
    st.stop()

hist = hist.copy()
hist["Código"] = hist["Código"].astype(str).str.strip()
hist["Nombre"] = hist["Nombre"].astype(str).fillna("")
hist["Año"] = pd.to_numeric(hist["Año"], errors="coerce")
hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce")
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)

# -------------------------
# Calcular MaxEne / MaxFeb (entre 2024 y 2025)
# -------------------------
base = hist[hist["Año"].isin([2024, 2025]) & hist["Mes"].isin([1, 2])].copy()

# Max por SKU por (Año, Mes) por si hay duplicados
g = (
    base.groupby(["Código", "Nombre", "Año", "Mes"], as_index=False)["Ventas"]
    .max()
)

p = g.pivot_table(
    index=["Código", "Nombre"],
    columns=["Año", "Mes"],
    values="Ventas",
    aggfunc="max",
    fill_value=0
)

needed = [(2024, 1), (2025, 1), (2024, 2), (2025, 2)]
for col in needed:
    if col not in p.columns:
        p[col] = 0

out = p[needed].copy()
out.columns = ["Max_Ene_2024", "Max_Ene_2025", "Max_Feb_2024", "Max_Feb_2025"]
out = out.reset_index()

out["MaxEne"] = out[["Max_Ene_2024", "Max_Ene_2025"]].max(axis=1)
out["MaxFeb"] = out[["Max_Feb_2024", "Max_Feb_2025"]].max(axis=1)

max_df = out[["Código", "Nombre", "MaxEne", "MaxFeb"]].copy()

# -------------------------
# Leer V30D + Stock y validar
# -------------------------
if v30_file.name.lower().endswith(".xlsx"):
    vs = pd.read_excel(v30_file)
else:
    vs = pd.read_csv(v30_file)

req_vs = {"Código", "V30D", "Stock"}
missing2 = req_vs - set(vs.columns)
if missing2:
    st.error(f"V30D+Stock: faltan columnas {sorted(missing2)}")
    st.stop()

vs = vs.copy()
vs["Código"] = vs["Código"].astype(str).str.strip()
vs["V30D"] = pd.to_numeric(vs["V30D"], errors="coerce").fillna(0)
vs["Stock"] = pd.to_numeric(vs["Stock"], errors="coerce").fillna(0)

# Nombre en V30D+Stock es opcional
if "Nombre" not in vs.columns:
    vs["Nombre"] = ""

# -------------------------
# Unión
# -------------------------
final = vs.merge(max_df, on="Código", how="left", suffixes=("_v30", "_hist"))

# Resolver nombre final: preferir Nombre del archivo V30D si viene; si no, el del histórico
final["Nombre_final"] = final["Nombre"].where(final["Nombre"].astype(str).str.strip() != "", final["Nombre_hist"])
final["Nombre_final"] = final["Nombre_final"].fillna("")

# Si no existe en histórico (SKU nuevo), MaxEne/MaxFeb quedan 0
final["MaxEne"] = pd.to_numeric(final["MaxEne"], errors="coerce").fillna(0)
final["MaxFeb"] = pd.to_numeric(final["MaxFeb"], errors="coerce").fillna(0)

# Tabla limpia
tabla = final[["Código", "Nombre_final", "V30D", "Stock", "MaxEne", "MaxFeb"]].rename(
    columns={"Nombre_final": "Nombre"}
)

# -------------------------
# UI
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs en V30D+Stock", f"{tabla['Código'].nunique():,}")
c2.metric("SKUs con MaxEne/MaxFeb encontrados", f"{int((tabla['MaxEne'] > 0).sum()):,}")
c3.metric("Suma V30D", f"{tabla['V30D'].sum():,.0f}")
c4.metric("Suma Stock", f"{tabla['Stock'].sum():,.0f}")

st.subheader("Tabla unificada")
st.dataframe(
    tabla.sort_values(["V30D", "MaxEne", "MaxFeb"], ascending=False),
    use_container_width=True,
    height=560
)

# Descarga
csv = tabla.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="union_maxene_maxfeb_v30d_stock.csv",
    mime="text/csv"
)
