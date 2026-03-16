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
APP_VERSION = "2026-03-16-v4.6 + múltiplos empaque"

# Parámetros clave
ALPHA_V30D = 0.25  # 25% influencia de V30D (últimos 30 días) cuando hay histórico

# Reglas de empaque (prefijo del nombre -> múltiplo)
PACK_RULES = [
    ("POLITEC 100M", 12),
    ("POLITEC 250M", 6),
    ("POLITEC 30M", 15),
    ("PINTURA OLEO ATL 160CC", 3),
    ("PINTURA OLEO ATL 40CC", 3),
]

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
# HELPERS
# =========================================================
CODE_RE = re.compile(r"^[A-Za-z0-9\-]+$")

def norm_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def norm_name_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").str.strip().str.upper()

def round_up_to_multiple(qty: float, multiple: int) -> int:
    qty = pd.to_numeric(qty, errors="coerce")
    if pd.isna(qty) or qty <= 0 or multiple <= 0:
        return 0
    return int(np.ceil(qty / multiple) * multiple)

def detect_pack_rule(name: str):
    n = str(name).strip().upper()
    for prefix, multiple in PACK_RULES:
        if n.startswith(prefix):
            return prefix, int(multiple)
    return "", np.nan

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
    out["Código"] = norm_code(out["Código"])

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
hist["Código"] = norm_code(hist["Código"])
hist["Nombre"] = hist["Nombre"].astype(str).fillna("").str.strip()
hist["Año"] = pd.to_numeric(hist["Año"], errors="coerce")
hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce")
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)
hist["Importe"] = pd.to_numeric(hist["Importe"], errors="coerce").fillna(0)

hist = hist[hist["Año"].isin([2024, 2025])].copy()

hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce").fillna(0).astype("int16")
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0).astype("float32")
hist["Importe"] = pd.to_numeric(hist["Importe"], errors="coerce").fillna(0).astype("float32")

# =========================================================
# COSTO UNITARIO
# =========================================================
costo_unit_df = (
    hist.groupby("Código", as_index=False)
        .agg(Ventas_total=("Ventas", "sum"), Importe_total=("Importe", "sum"))
)
costo_unit_df["Costo_Unitario"] = np.where(
    costo_unit_df["Ventas_total"] > 0,
    costo_unit_df["Importe_total"] / costo_unit_df["Ventas_total"],
    np.nan
)
costo_unit_df["Costo_Unitario"] = pd.to_numeric(costo_unit_df["Costo_Unitario"], errors="coerce")

# =========================================================
# HISTÓRICO TOTAL
# =========================================================
hist_total_24_25 = (
    hist.groupby("Código", as_index=False)["Ventas"]
        .sum()
        .rename(columns={"Ventas": "Hist_24_25_Ventas"})
)

# =========================================================
# MERGE
# =========================================================
final = vs.merge(hist_total_24_25, on="Código", how="left")
final = final.merge(costo_unit_df[["Código", "Costo_Unitario"]], on="Código", how="left")

# =========================================================
# DEMANDA + COMPRA
# =========================================================
final["Demanda30"] = final["V30D"]
final["Compra"] = (final["Demanda30"] - final["Stock"]).clip(lower=0).astype(int)

# =========================================================
# IMPORTE A COMPRAR
# =========================================================
final["Importe_Compra"] = final["Compra"] * final["Costo_Unitario"]

# =========================================================
# TABLA
# =========================================================
tabla = final[
    [
        "Código","EAN","Nombre","Compra","Stock","V30D",
        "Demanda30","Costo_Unitario","Importe_Compra"
    ]
]

tabla["Costo_Unitario"] = tabla["Costo_Unitario"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
tabla["Importe_Compra"] = tabla["Importe_Compra"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

# =========================================================
# UI
# =========================================================
st.subheader("Compra Sugerida")
st.dataframe(
    tabla.sort_values(["Compra","Demanda30"],ascending=False),
    use_container_width=True,
    height=600,
    hide_index=True
)

st.download_button(
    "Descargar CSV",
    data=tabla.to_csv(index=False).encode("utf-8-sig"),
    file_name="Compra sugerida.csv",
    mime="text/csv"
)
