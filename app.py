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
APP_VERSION = "2026-02-02-v4.1 (P90 entero por mes + V30D siempre 25%)"

# Parámetros clave
ALPHA_V30D = 0.25  # 25% influencia de V30D (últimos 30 días)

st.set_page_config(page_title="Agente de compras", layout="wide")

# =========================================================
# TOP BAR: TÍTULO + BOTÓN
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
# REGEX
# =========================================================
CODE_RE = re.compile(r"^[A-Za-z0-9\-]+$")

# =========================================================
# ERPLY HTML
# =========================================================
def _looks_like_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or df.shape[1] < 12:
        return False

    def is_code(x):
        s = str(x).strip()
        return bool(CODE_RE.match(s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(100, len(df))):
        if is_code(df.iloc[i, 1]) and isinstance(df.iloc[i, 3], str):
            return True
    return False

def _detect_data_start(df: pd.DataFrame) -> int:
    def is_code(x):
        s = str(x).strip()
        return bool(CODE_RE.match(s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(180, len(df))):
        if is_code(df.iloc[i, 1]) and isinstance(df.iloc[i, 3], str):
            return i
    return 0

@st.cache_data(show_spinner=False)
def read_erply_html_bytes(data: bytes) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.BytesIO(data), header=None, flavor="lxml")
    except Exception:
        tables = pd.read_html(io.BytesIO(data), header=None)

    candidates = [t for t in tables if t.shape[1] >= 12]
    chosen = max(candidates, key=lambda t: t.shape[0]) if candidates else tables[0]

    if not _looks_like_table(chosen):
        for t in candidates:
            if _looks_like_table(t):
                chosen = t
                break

    start = _detect_data_start(chosen)
    raw = chosen.iloc[start:, :12].reset_index(drop=True)

    out = pd.DataFrame({
        "Código": raw.iloc[:, 1].astype(str).str.strip(),
        "EAN": raw.iloc[:, 2].astype(str).fillna("").str.strip(),
        "Nombre": raw.iloc[:, 3].astype(str).fillna("").str.strip(),
        "V30D": pd.to_numeric(raw.iloc[:, 4], errors="coerce").fillna(0),
        "Stock": pd.to_numeric(raw.iloc[:, 6], errors="coerce").fillna(0),
        "V30D_Pesos": pd.to_numeric(raw.iloc[:, 11], errors="coerce").fillna(0),
    })

    cod = out["Código"].astype(str).str.strip().str.lower()
    out = out[(out["Código"].astype(str).str.strip() != "") & (cod != "nan")]
    out = out[~cod.str.contains(r"^total", regex=True, na=False)]

    out["EAN"] = out["EAN"].replace({"nan": "", "None": "", "NONE": ""})

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
        "1) Histórico (.xlsx) — columnas: Código, Nombre, Año, Mes, Ventas, Importe",
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

hist = read_hist_xlsx_bytes(hist_file.getvalue())
vs = read_erply_html_bytes(erply_file.getvalue())

# =========================================================
# HISTÓRICO
# =========================================================
req = {"Código", "Nombre", "Año", "Mes", "Ventas", "Importe"}
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
hist["Importe"] = pd.to_numeric(hist["Importe"], errors="coerce").fillna(0)

# SOLO 2024-2025 (igual que antes)
hist = hist[hist["Año"].isin([2024, 2025])].copy()
hist["Mes"] = hist["Mes"].astype("int16", errors="ignore")
hist["Ventas"] = hist["Ventas"].astype("float32", errors="ignore")
hist["Importe"] = hist["Importe"].astype("float32", errors="ignore")

# =========================================================
# P90 POR MES (por Código) - reemplaza MAX
# =========================================================
nombre_hist = (
    hist.loc[hist["Nombre"] != "", ["Código", "Nombre"]]
    .drop_duplicates("Código")
    .rename(columns={"Nombre": "Nombre_hist"})
)

def p90_int(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return 0
    return int(np.round(np.percentile(x, 90), 0))

g = (
    hist.groupby(["Código", "Mes"])["Ventas"]
        .apply(p90_int)
        .reset_index(name="P90")
)

p = g.pivot_table(index="Código", columns="Mes", values="P90", fill_value=0).reset_index()

for m in range(1, 13):
    if m not in p.columns:
        p[m] = 0

# Conservamos nombres para tocar lo mínimo
p = p.rename(columns={m: f"Max_M{m:02d}" for m in range(1, 13)})

max_mes_df = p.merge(nombre_hist, on="Código", how="left")

# =========================================================
# MERGE
# =========================================================
final = vs.merge(max_mes_df, on="Código", how="left")

final["Nombre"] = final["Nombre"].fillna("").astype(str).str.strip()
final["Nombre_hist"] = final.get("Nombre_hist", "").fillna("").astype(str).str.strip()
final["Nombre"] = np.where(
    final["Nombre"] != "",
    final["Nombre"],
    np.where(final["Nombre_hist"] != "", final["Nombre_hist"], "(sin nombre)")
)

# =========================================================
# FECHA + DEMANDA
# =========================================================
tz = ZoneInfo("America/Mazatlan")
hoy = datetime.now(tz).date()

dias_mes = calendar.monthrange(hoy.year, hoy.month)[1]
peso_actual = (dias_mes - hoy.day + 1) / dias_mes
peso_siguiente = 1 - peso_actual

mes_actual = hoy.month
mes_siguiente = 1 if mes_actual == 12 else mes_actual + 1

col_act = f"Max_M{mes_actual:02d}"
col_sig = f"Max_M{mes_siguiente:02d}"

if col_act not in final.columns:
    final[col_act] = 0
if col_sig not in final.columns:
    final[col_sig] = 0

# =========================================================
# FIX: FORZAR NUMÉRICOS + NaN->0
# =========================================================
final["Stock"] = pd.to_numeric(final["Stock"], errors="coerce").fillna(0)
final["V30D"] = pd.to_numeric(final["V30D"], errors="coerce").fillna(0)
final[col_act] = pd.to_numeric(final[col_act], errors="coerce").fillna(0)
final[col_sig] = pd.to_numeric(final[col_sig], errors="coerce").fillna(0)

# =========================================================
# DEMANDA30: base estacional (P90) + V30D siempre 25%
# =========================================================
base_hist = (peso_actual * final[col_act]) + (peso_siguiente * final[col_sig])
base_hist = pd.to_numeric(base_hist, errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)

final["Demanda30"] = ((1 - ALPHA_V30D) * base_hist) + (ALPHA_V30D * final["V30D"])
final["Demanda30"] = pd.to_numeric(final["Demanda30"], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)

# Compra con demanda redondeada (opcional, mantiene coherencia)
final["Demanda30"] = np.round(final["Demanda30"], 0).astype(int)

# =========================================================
# COMPRA
# =========================================================
final["Compra"] = (
    np.ceil(final["Demanda30"] - final["Stock"])
      .clip(lower=0)
      .replace([np.inf, -np.inf], 0)
      .fillna(0)
      .astype(int)
)

# =========================================================
# TABLA
# =========================================================
tabla = final[
    ["Código", "EAN", "Nombre", "Compra", "Stock", "V30D", col_act, col_sig, "Demanda30"]
].rename(columns={
    col_act: f"P90Mes_{mes_actual:02d}",
    col_sig: f"P90Mes_{mes_siguiente:02d}",
})

# Garantiza enteros sin decimales en P90 y Demanda30
tabla[f"P90Mes_{mes_actual:02d}"] = pd.to_numeric(tabla[f"P90Mes_{mes_actual:02d}"], errors="coerce").fillna(0).astype(int)
tabla[f"P90Mes_{mes_siguiente:02d}"] = pd.to_numeric(tabla[f"P90Mes_{mes_siguiente:02d}"], errors="coerce").fillna(0).astype(int)
tabla["Demanda30"] = pd.to_numeric(tabla["Demanda30"], errors="coerce").fillna(0).astype(int)

# =========================================================
# MÉTRICAS
# =========================================================
def importe_mes(hist_df, mes):
    s = hist_df[(hist_df["Mes"] == mes) & (hist_df["Año"] == hoy.year)]["Importe"].sum()
    if s > 0:
        return float(s)
    years = hist_df.loc[hist_df["Mes"] == mes, "Año"].dropna()
    if len(years) == 0:
        return 0.0
    y = int(years.max())
    return float(hist_df[(hist_df["Mes"] == mes) & (hist_df["Año"] == y)]["Importe"].sum())

skus_total = tabla["Código"].nunique()
skus_compra = tabla.loc[tabla["Compra"] > 0, "Código"].nunique()

m1, m2, m3, m4 = st.columns(4)
m1.metric("SKUs Erply", f"{skus_total:,}")
m2.metric("SKUs Compra", f"{skus_compra:,}")
m3.metric(f"Importe Mes {mes_actual:02d} (hist)", f"${importe_mes(hist, mes_actual):,.0f}")
m4.metric(f"Importe Mes {mes_siguiente:02d} (hist)", f"${importe_mes(hist, mes_siguiente):,.0f}")

# =========================================================
# UI TABLA
# =========================================================
st.subheader("Compra Sugerida (P90 + V30D 25%)")
st.dataframe(
    tabla.sort_values(["Compra", "Demanda30"], ascending=False),
    use_container_width=True,
    height=600,
    hide_index=True
)

st.download_button(
    "Descargar CSV",
    data=tabla.to_csv(index=False).encode("utf-8-sig"),
    file_name="compra_sugerida_30d_p90_v30d25.csv",
    mime="text/csv"
)
