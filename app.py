import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo

# =========================
# CONFIG + VERSION
# =========================
APP_VERSION = "2025-12-27-v3 (Métrica Enero + cache reset + bugfix Nombre_hist)"

st.set_page_config(page_title="Agente de compras", layout="wide")
st.title("Agente de compras")
st.caption(f"Versión app: {APP_VERSION}")

# Sidebar: controles duros para que Streamlit no te “muestre viejo”
with st.sidebar:
    st.header("Controles")
    if st.button("Limpiar caché y reiniciar"):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# LECTURA ERPLY .XLS (HTML) por posiciones:
# B = Código (idx 1)
# C = EAN    (idx 2)
# D = Nombre (idx 3)
# E = V30D   (idx 4)
# G = Stock  (idx 6)
# L = Importe venta $ (idx 11)
# =========================================================
def _looks_like_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or df.shape[1] < 12:
        return False

    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(120, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(cB) and isinstance(cD, str) and len(str(cD).strip()) > 2:
            return True
    return False

def _detect_data_start(df: pd.DataFrame) -> int:
    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(250, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(cB) and isinstance(cD, str) and len(str(cD).strip()) > 2:
            return i
    return 0

@st.cache_data(show_spinner=False)
def read_erply_html_bytes(data: bytes) -> pd.DataFrame:
    tables = pd.read_html(io.BytesIO(data), header=None)

    chosen = None
    for t in tables:
        if _looks_like_table(t):
            chosen = t
            break
    if chosen is None:
        chosen = tables[0]

    start = _detect_data_start(chosen)
    raw = chosen.iloc[start:, :12].copy().reset_index(drop=True)

    out = pd.DataFrame({
        "Código": raw.iloc[:, 1].astype(str).str.strip(),                              # B
        "EAN": raw.iloc[:, 2].astype(str).fillna("").str.strip(),                      # C
        "Nombre": raw.iloc[:, 3].astype(str).fillna("").str.strip(),                   # D
        "V30D": pd.to_numeric(raw.iloc[:, 4], errors="coerce").fillna(0),              # E
        "Stock": pd.to_numeric(raw.iloc[:, 6], errors="coerce").fillna(0),             # G
        "V30D_Pesos": pd.to_numeric(raw.iloc[:, 11], errors="coerce").fillna(0),       # L
    })

    # limpiar vacíos y "nan"
    out = out[out["Código"].astype(str).str.strip().ne("")]
    out = out[out["Código"].astype(str).str.lower().ne("nan")]

    # quitar fila de totales (ej. "total ($)")
    cod_l = out["Código"].astype(str).str.strip().str.lower()
    out = out[~cod_l.isin(["total ($)", "total($)", "total", "totales"])]
    out = out[~cod_l.str.contains(r"^total\b", regex=True, na=False)]

    # normalizar EAN: si viene "nan"
    out["EAN"] = out["EAN"].replace({"nan": "", "None": "", "NONE": ""})

    # tipos
    out["V30D"] = pd.to_numeric(out["V30D"], errors="coerce").fillna(0)
    out["Stock"] = pd.to_numeric(out["Stock"], errors="coerce").fillna(0)
    out["V30D_Pesos"] = pd.to_numeric(out["V30D_Pesos"], errors="coerce").fillna(0)

    return out.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def read_hist_xlsx_bytes(data: bytes) -> pd.DataFrame:
    hist = pd.read_excel(io.BytesIO(data), engine="openpyxl")
    return hist

# =========================================================
# UPLOADERS
# =========================================================
colL, colR = st.columns(2)
with colL:
    hist_file = st.file_uploader(
        "1) Histórico (.xlsx) — columnas: Código, Nombre, Año, Mes, Ventas",
        type=["xlsx"]
    )
with colR:
    erply_file = st.file_uploader(
        "2) Erply V30D + Stock (.xls)",
        type=["xls"]
    )

if hist_file is None or erply_file is None:
    st.info("Sube ambos archivos para continuar.")
    st.stop()

# =========================================================
# LEER ARCHIVOS (bytes) — evita que Streamlit te muestre viejo
# =========================================================
hist_bytes = hist_file.getvalue()
erply_bytes = erply_file.getvalue()

hist = read_hist_xlsx_bytes(hist_bytes)
vs = read_erply_html_bytes(erply_bytes)

# =========================================================
# VALIDAR HISTÓRICO
# =========================================================
req = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = req - set(hist.columns)
if missing:
    st.error(f"Histórico: faltan columnas {sorted(missing)}")
    st.stop()

hist = hist.copy()
hist["Código"] = hist["Código"].astype(str).str.strip()
hist["Nombre"] = hist["Nombre"].astype(str).fillna("").str.strip()
hist["Año"] = pd.to_numeric(hist["Año"], errors="coerce")
hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce")
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)

# SOLO 2024-2025 (tu lógica original)
hist = hist[hist["Año"].isin([2024, 2025])].copy()

# =========================================================
# HISTÓRICO -> MAX POR MES (por Código)
# =========================================================
nombre_hist = (
    hist.loc[hist["Nombre"].ne(""), ["Código", "Nombre"]]
        .drop_duplicates(subset=["Código"], keep="first")
        .rename(columns={"Nombre": "Nombre_hist"})
)

g = hist.groupby(["Código", "Mes"], as_index=False)["Ventas"].max()

p = g.pivot_table(
    index=["Código"],
    columns="Mes",
    values="Ventas",
    aggfunc="max",
    fill_value=0
).reset_index()

for m in range(1, 13):
    if m not in p.columns:
        p[m] = 0

p = p.rename(columns={m: f"Max_M{m:02d}" for m in range(1, 13)})
max_mes_df = p.merge(nombre_hist, on="Código", how="left")

# =========================================================
# MERGE ERPLY + HISTÓRICO
# =========================================================
final = vs.merge(max_mes_df, on="Código", how="left")

# =========================================================
# NOMBRE: manda Erply; fallback histórico; si no hay "(sin nombre)"
# (FIX REAL: garantizar columna Nombre_hist válida)
# =========================================================
final["Nombre_erply"] = final["Nombre"].astype(str).fillna("").str.strip()

if "Nombre_hist" not in final.columns:
    final["Nombre_hist"] = ""
else:
    final["Nombre_hist"] = final["Nombre_hist"].fillna("").astype(str).str.strip()

final["Nombre"] = np.where(
    final["Nombre_erply"].ne(""),
    final["Nombre_erply"],
    np.where(final["Nombre_hist"].ne(""), final["Nombre_hist"], "(sin nombre)")
)
final = final.drop(columns=["Nombre_erply"], errors="ignore")

# =========================================================
# FECHA MAZATLÁN + PESOS
# =========================================================
tz = ZoneInfo("America/Mazatlan")
hoy = datetime.now(tz).date()

dias_mes = calendar.monthrange(hoy.year, hoy.month)[1]
dias_restantes_incl_hoy = dias_mes - hoy.day + 1

peso_actual = dias_restantes_incl_hoy / dias_mes
peso_siguiente = 1 - peso_actual

mes_actual = hoy.month
mes_siguiente = 1 if mes_actual == 12 else mes_actual + 1

col_act = f"Max_M{mes_actual:02d}"
col_sig = f"Max_M{mes_siguiente:02d}"

st.caption(
    f"Fecha Mazatlán: {hoy} | Mes actual={mes_actual} peso={peso_actual:.4f} | "
    f"Mes siguiente={mes_siguiente} peso={peso_siguiente:.4f}"
)
st.caption(f"Columnas usadas: {col_act} y {col_sig}")

# =========================================================
# DEMANDA 30 DÍAS (PONDERADA) + FALLBACK A V30D
# =========================================================
if col_act not in final.columns:
    final[col_act] = 0
if col_sig not in final.columns:
    final[col_sig] = 0

final[col_act] = pd.to_numeric(final[col_act], errors="coerce").fillna(0)
final[col_sig] = pd.to_numeric(final[col_sig], errors="coerce").fillna(0)

final["Demanda30"] = (peso_actual * final[col_act]) + (peso_siguiente * final[col_sig])

suma_max_2m = final[col_act] + final[col_sig]
mask_fallback = (suma_max_2m == 0) & (final["V30D"] > 0)
final.loc[mask_fallback, "Demanda30"] = final.loc[mask_fallback, "V30D"]

# =========================================================
# COMPRA SUGERIDA
# =========================================================
final["Compra_sugerida"] = np.ceil(final["Demanda30"] - final["Stock"]).clip(lower=0).astype(int)
final["Demanda30_mostrar"] = np.round(final["Demanda30"], 0).astype(int)

# =========================================================
# TABLA FINAL
# =========================================================
tabla = final[
    ["Código", "EAN", "Nombre", "Compra_sugerida", "Stock", "V30D", col_act, col_sig, "Demanda30_mostrar"]
].rename(columns={
    "Compra_sugerida": "Compra",
    col_act: f"MaxMes_{mes_actual:02d}",
    col_sig: f"MaxMes_{mes_siguiente:02d}",
    "Demanda30_mostrar": "Demanda30"
})

# =========================================================
# MÉTRICA 4: VENTAS ENERO (HISTÓRICO)
# Nota: asumo que hist["Ventas"] es el “importe” que quieres.
# Prioridad: Enero del año actual si existe; si no, Enero más reciente disponible.
# =========================================================
jan_current = hist[(hist["Año"] == hoy.year) & (hist["Mes"] == 1)]["Ventas"].sum()

if jan_current > 0:
    ventas_enero_importe = float(jan_current)
else:
    years_with_jan = hist.loc[hist["Mes"] == 1, "Año"].dropna()
    if len(years_with_jan) > 0:
        y_last = int(years_with_jan.max())
        ventas_enero_importe = float(hist[(hist["Año"] == y_last) & (hist["Mes"] == 1)]["Ventas"].sum())
    else:
        ventas_enero_importe = 0.0

# =========================================================
# UI (MÉTRICAS)
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("SKUs Erply", f"{tabla['Código'].nunique():,}")
m2.metric("Suma Stock", f"{tabla['Stock'].sum():,.0f}")
m3.metric("Suma V30D (info)", f"{vs['V30D_Pesos'].sum():,.0f}")
m4.metric("Ventas Enero (histórico)", f"${ventas_enero_importe:,.2f}")

# =========================================================
# UI (TABLA)
# =========================================================
st.subheader("Tabla unificada + compra sugerida")
st.dataframe(
    tabla.sort_values(["Compra", "Demanda30"], ascending=False),
    use_container_width=True,
    height=600,
    hide_index=True
)

csv = tabla.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="compra_sugerida_30d_ponderada.csv",
    mime="text/csv"
)
