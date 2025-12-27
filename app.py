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
APP_VERSION = "2025-12-27-v3-fast (UI button top + metrics tweak)"

st.set_page_config(page_title="Agente de compras", layout="wide")

# =========================================================
# TOP BAR: TÍTULO + BOTÓN (AL LADO)
# =========================================================
tcol1, tcol2 = st.columns([0.78, 0.22], vertical_alignment="center")
with tcol1:
    st.title("Agente de compras")
    st.caption(f"Versión app: {APP_VERSION}")
with tcol2:
    if st.button("Limpiar caché y reiniciar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =========================================================
# REGEX COMPILADO (micro-optimización)
# =========================================================
CODE_RE = re.compile(r"^[A-Za-z0-9\-]+$")

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
        return bool(CODE_RE.match(s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(100, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(cB) and isinstance(cD, str) and len(str(cD).strip()) > 2:
            return True
    return False

def _detect_data_start(df: pd.DataFrame) -> int:
    def is_code(x):
        s = str(x).strip()
        return bool(CODE_RE.match(s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(180, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(cB) and isinstance(cD, str) and len(str(cD).strip()) > 2:
            return i
    return 0

@st.cache_data(show_spinner=False)
def read_erply_html_bytes(data: bytes) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.BytesIO(data), header=None, flavor="lxml")
    except Exception:
        tables = pd.read_html(io.BytesIO(data), header=None)

    candidates = [t for t in tables if t is not None and not t.empty and t.shape[1] >= 12]
    chosen = max(candidates, key=lambda t: t.shape[0]) if candidates else tables[0]

    if not _looks_like_table(chosen):
        for t in candidates:
            if _looks_like_table(t):
                chosen = t
                break

    start = _detect_data_start(chosen)
    raw = chosen.iloc[start:, :12].reset_index(drop=True)

    out = pd.DataFrame({
        "Código": raw.iloc[:, 1].astype(str).str.strip(),                              # B
        "EAN": raw.iloc[:, 2].astype(str).fillna("").str.strip(),                      # C
        "Nombre": raw.iloc[:, 3].astype(str).fillna("").str.strip(),                   # D
        "V30D": pd.to_numeric(raw.iloc[:, 4], errors="coerce").fillna(0),              # E
        "Stock": pd.to_numeric(raw.iloc[:, 6], errors="coerce").fillna(0),             # G
        "V30D_Pesos": pd.to_numeric(raw.iloc[:, 11], errors="coerce").fillna(0),       # L
    })

    cod = out["Código"].astype(str).str.strip()
    cod_l = cod.str.lower()
    out = out[cod.ne("") & cod_l.ne("nan")]

    out = out[~cod_l.isin(["total ($)", "total($)", "total", "totales"])]
    out = out[~cod_l.str.contains(r"^total\b", regex=True, na=False)]

    out["EAN"] = out["EAN"].replace({"nan": "", "None": "", "NONE": ""})

    out["V30D"] = pd.to_numeric(out["V30D"], errors="coerce").fillna(0)
    out["Stock"] = pd.to_numeric(out["Stock"], errors="coerce").fillna(0)
    out["V30D_Pesos"] = pd.to_numeric(out["V30D_Pesos"], errors="coerce").fillna(0)

    return out.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def read_hist_xlsx_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")

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
# LEER ARCHIVOS (bytes)
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

# SOLO 2024-2025 (se deja IGUAL para no cambiar resultados)
hist = hist[hist["Año"].isin([2024, 2025])].copy()

# Tipos más ligeros
hist["Mes"] = hist["Mes"].astype("int16", errors="ignore")
hist["Ventas"] = hist["Ventas"].astype("float32", errors="ignore")

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
# =========================================================
final["Nombre"] = final["Nombre"].astype(str).fillna("").str.strip()
if "Nombre_hist" not in final.columns:
    final["Nombre_hist"] = ""
final["Nombre_hist"] = final["Nombre_hist"].fillna("").astype(str).str.strip()

final["Nombre"] = np.where(
    final["Nombre"].ne(""),
    final["Nombre"],
    np.where(final["Nombre_hist"].ne(""), final["Nombre_hist"], "(sin nombre)")
)

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

# =========================================================
# TABLA FINAL (misma info)
# =========================================================
tabla = final[
    ["Código", "EAN", "Nombre", "Compra_sugerida", "Stock", "V30D", col_act, col_sig, "Demanda30"]
].rename(columns={
    "Compra_sugerida": "Compra",
    col_act: f"MaxMes_{mes_actual:02d}",
    col_sig: f"MaxMes_{mes_siguiente:02d}",
})
tabla["Demanda30"] = np.round(tabla["Demanda30"], 0).astype(int)

# =========================================================
# HELPERS MÉTRICAS IMPORTE POR MES (HISTÓRICO)
# Prioridad: año actual si existe; si no, el más reciente disponible para ese mes.
# =========================================================
def _importe_mes(hist_df: pd.DataFrame, mes: int, prefer_year: int) -> float:
    s_prefer = hist_df[(hist_df["Año"] == prefer_year) & (hist_df["Mes"] == mes)]["Ventas"].sum()
    if s_prefer > 0:
        return float(s_prefer)

    years = hist_df.loc[hist_df["Mes"] == mes, "Año"].dropna()
    if len(years) == 0:
        return 0.0
    y_last = int(years.max())
    return float(hist_df[(hist_df["Año"] == y_last) & (hist_df["Mes"] == mes)]["Ventas"].sum())

# Ventas Enero (histórico) sin decimales
ventas_enero_importe = _importe_mes(hist, 1, hoy.year)

# También mostrar importes de los 2 meses usados para la compra (mes actual y mes siguiente)
ventas_mes_actual_importe = _importe_mes(hist, mes_actual, hoy.year)
ventas_mes_siguiente_importe = _importe_mes(hist, mes_siguiente, hoy.year)

# =========================================================
# UI (MÉTRICAS) — quitamos Suma Stock
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("SKUs Erply", f"{tabla['Código'].nunique():,}")
m2.metric("Suma V30D (info)", f"{vs['V30D_Pesos'].sum():,.0f}")
m3.metric(f"Ventas Mes {mes_actual:02d} (hist)", f"${ventas_mes_actual_importe:,.0f}")
m4.metric(f"Ventas Mes {mes_siguiente:02d} (hist)", f"${ventas_mes_siguiente_importe:,.0f}")
m5.metric("Ventas Enero (hist)", f"${ventas_enero_importe:,.0f}")

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

