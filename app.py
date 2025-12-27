import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="MaxEne/MaxFeb + V30D", layout="wide")

st.divider()
st.header("Unión: Histórico (MaxEne/MaxFeb) + V30D + Stock")

# -------------------------
# Lector .xls (HTML) tipo Erply — tomado del script que compartiste
# -------------------------
def _detect_data_start(df):
    """Primera fila donde col1 parece código y col3 es nombre de producto."""
    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(60, len(df))):
        c1 = df.iloc[i, 1] if df.shape[1] > 1 else None
        c3 = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(c1) and isinstance(c3, str) and len(str(c3).strip()) > 2 and "nombre" not in str(c3).lower():
            return i
    return 0

def _read_erply_xls_like_html(file_obj) -> pd.DataFrame:
    """
    Lee el .xls (que en realidad es HTML) exportado por Erply.
    Devuelve un DataFrame con columnas fijas por posición.
    """
    file_obj.seek(0)
    df0 = pd.read_html(file_obj, header=None)[0]
    start = _detect_data_start(df0)
    df = df0.iloc[start:, :12].copy()
    df.columns = [
        "No", "Código", "Código EAN", "Nombre",
        "Stock (total)", "Stock (apartado)", "Stock (disponible)",
        "Proveedor",
        "V30D", "Ventas corto ($)",
        "V365", "Ventas 365 ($)"
    ]
    return df.dropna(how="all").reset_index(drop=True)

def read_v30_stock(uploaded) -> pd.DataFrame:
    """
    Lee V30D+Stock en:
    - .csv (normal)
    - .xlsx (normal)
    - .xls real (xlrd)
    - .xls HTML de Erply (read_html + parse por posición)
    Devuelve SIEMPRE un df con columnas: Código, Nombre (opcional), V30D, Stock
    """
    name = uploaded.name.lower()

    # CSV
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        return df

    # XLSX
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded, engine="openpyxl")
        return df

    # XLS: puede ser real o HTML disfrazado
    if name.endswith(".xls"):
        # 1) intentar como HTML-Erply (es lo que te está pasando)
        try:
            t = _read_erply_xls_like_html(uploaded)

            # Convertir a formato estándar (Código, Nombre, V30D, Stock)
            out = pd.DataFrame({
                "Código": t["Código"],
                "Nombre": t["Nombre"],
                "V30D": pd.to_numeric(t["V30D"], errors="coerce").fillna(0),
                "Stock": pd.to_numeric(t["Stock (total)"], errors="coerce").fillna(0),
            })
            return out

        except Exception:
            # 2) si no es HTML, intentar como XLS real con xlrd
            uploaded.seek(0)
            df = pd.read_excel(uploaded, engine="xlrd")
            return df

    raise ValueError("Formato de V30D+Stock no soportado (usa .xlsx, .csv o .xls).")

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
        "2) Sube V30D + Stock (.xls/.xlsx/.csv)",
        type=["xls", "xlsx", "csv"]
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
try:
    vs = read_v30_stock(v30_file)
except Exception as e:
    st.error(f"No pude leer el archivo V30D+Stock. Error: {e}")
    st.stop()

# Si el archivo NO venía en formato estándar, aquí normalizamos a lo mínimo
# (por ejemplo si lo leímos como XLS real y trae nombres distintos, fallará abajo)
req_vs = {"Código", "V30D", "Stock"}
missing2 = req_vs - set(vs.columns)
if missing2:
    st.error(
        "V30D+Stock: faltan columnas.\n"
        f"Faltan: {sorted(missing2)}\n\n"
        f"Columnas encontradas: {list(vs.columns)}\n\n"
        "Si tu archivo tiene otros nombres de columnas, dímelos y lo mapeo."
    )
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

final["Nombre_final"] = final["Nombre"].where(final["Nombre"].astype(str).str.strip() != "", final["Nombre_hist"])
final["Nombre_final"] = final["Nombre_final"].fillna("")

final["MaxEne"] = pd.to_numeric(final.get("MaxEne"), errors="coerce").fillna(0)
final["MaxFeb"] = pd.to_numeric(final.get("MaxFeb"), errors="coerce").fillna(0)

tabla = final[["Código", "Nombre_final", "V30D", "Stock", "MaxEne", "MaxFeb"]].rename(
    columns={"Nombre_final": "Nombre"}
)

# -------------------------
# UI
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs en V30D+Stock", f"{tabla['Código'].nunique():,}")
c2.metric("SKUs con MaxEne/MaxFeb (match)", f"{int(((tabla['MaxEne'] > 0) | (tabla['MaxFeb'] > 0)).sum()):,}")
c3.metric("Suma V30D", f"{tabla['V30D'].sum():,.0f}")
c4.metric("Suma Stock", f"{tabla['Stock'].sum():,.0f}")

st.subheader("Tabla unificada")
st.dataframe(
    tabla.sort_values(["V30D", "MaxEne", "MaxFeb"], ascending=False),
    use_container_width=True,
    height=560
)

csv = tabla.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="union_maxene_maxfeb_v30d_stock.csv",
    mime="text/csv"
)
