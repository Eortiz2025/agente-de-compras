import streamlit as st
import pandas as pd
import re
import io

st.set_page_config(page_title="MaxEne/MaxFeb + V30D", layout="wide")

st.divider()
st.header("Unión: Histórico (MaxEne/MaxFeb) + V30D + Stock")

# -------------------------
# Helpers para Erply .xls (HTML disfrazado)
# -------------------------
def _looks_like_erply_table(df: pd.DataFrame) -> bool:
    """
    Heurística: buscamos que existan columnas suficientes
    y que en alguna fila haya:
    - B: código (alfa-num con guión)
    - D: nombre (texto)
    """
    if df is None or df.empty or df.shape[1] < 8:
        return False

    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()

    # revisar primeras filas
    for i in range(min(80, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None  # B
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None  # D
        if is_code(cB) and isinstance(cD, str) and len(cD.strip()) > 2 and "nombre" not in cD.lower():
            return True
    return False

def _detect_data_start(df: pd.DataFrame) -> int:
    """Encuentra la fila real donde empiezan los datos (B=código, D=nombre)."""
    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(160, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(cB) and isinstance(cD, str) and len(str(cD).strip()) > 2 and "nombre" not in str(cD).lower():
            return i
    return 0

def _read_erply_xls_like_html(uploaded) -> pd.DataFrame:
    """
    Lee el .xls exportado por Erply (en realidad HTML).
    Maneja:
    - Streamlit UploadedFile (bytes)
    - HTML con varias tablas: elige la tabla que parece catálogo
    """
    data = uploaded.getvalue()

    # read_html devuelve lista de tablas
    tables = pd.read_html(io.BytesIO(data), header=None)

    # elegir tabla correcta
    chosen = None
    for t in tables:
        if _looks_like_erply_table(t):
            chosen = t
            break

    if chosen is None:
        # si ninguna tabla pasa heurística, usa la primera pero probablemente falle luego
        chosen = tables[0]

    start = _detect_data_start(chosen)

    # Tomamos hasta 12 columnas por si vienen las clásicas
    df = chosen.iloc[start:, :12].copy()

    # Si vinieron menos columnas, rellenamos para asignar nombres sin reventar
    while df.shape[1] < 12:
        df[df.shape[1]] = None

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
    Lee V30D+Stock:
    - .csv
    - .xlsx
    - .xls HTML de Erply (preferido)
    - .xls real binario (xlrd)
    Devuelve SIEMPRE: Código, Nombre (puede venir vacío), V30D, Stock
    """
    name = uploaded.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        return df

    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded, engine="openpyxl")
        return df

    if name.endswith(".xls"):
        # 1) HTML-Erply
        try:
            t = _read_erply_xls_like_html(uploaded)

            # IMPORTANTE: stock en tu caso es columna G = Stock (disponible)
            out = pd.DataFrame({
                "Código": t["Código"].astype(str).str.strip(),
                "Nombre": t["Nombre"].astype(str).fillna(""),
                "V30D": pd.to_numeric(t["V30D"], errors="coerce").fillna(0),
                "Stock": pd.to_numeric(t["Stock (disponible)"], errors="coerce").fillna(0),
            })
            return out
        except Exception:
            # 2) XLS binario real
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
hist = pd.read_excel(hist_file, engine="openpyxl")

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
max_df = max_df.rename(columns={"Nombre": "Nombre_hist"})

# -------------------------
# Leer V30D + Stock y normalizar
# -------------------------
try:
    vs = read_v30_stock(v30_file)
except Exception as e:
    st.error(
        "No pude leer el archivo V30D+Stock.\n"
        f"Error: {e}"
    )
    st.stop()

# Si el archivo viene “normal” pero con otros nombres, aquí tronaría.
# En tu caso Erply ya regresa Código/Nombre/V30D/Stock.
req_vs = {"Código", "V30D", "Stock"}
missing2 = req_vs - set(vs.columns)
if missing2:
    st.error(
        "V30D+Stock: faltan columnas.\n"
        f"Faltan: {sorted(missing2)}\n\n"
        f"Columnas encontradas: {list(vs.columns)}"
    )
    st.stop()

vs = vs.copy()
vs["Código"] = vs["Código"].astype(str).str.strip()
vs["V30D"] = pd.to_numeric(vs["V30D"], errors="coerce").fillna(0)
vs["Stock"] = pd.to_numeric(vs["Stock"], errors="coerce").fillna(0)

if "Nombre" not in vs.columns:
    vs["Nombre"] = ""
vs["Nombre"] = vs["Nombre"].astype(str).fillna("")

# -------------------------
# Unión robusta
# -------------------------
final = vs.merge(max_df, on="Código", how="left")

# Asegurar columnas
if "Nombre_hist" not in final.columns:
    final["Nombre_hist"] = ""
final["Nombre_hist"] = final["Nombre_hist"].astype(str).fillna("")

# Nombre final: si vs trae Nombre úsalo; si no, usa el del histórico
nombre_vs = final["Nombre"].astype(str).fillna("").str.strip()
nombre_hist = final["Nombre_hist"].astype(str).fillna("").str.strip()
final["Nombre_final"] = nombre_vs.where(nombre_vs != "", nombre_hist).fillna("")

final["MaxEne"] = pd.to_numeric(final["MaxEne"], errors="coerce").fillna(0)
final["MaxFeb"] = pd.to_numeric(final["MaxFeb"], errors="coerce").fillna(0)

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
