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

ALPHA_V30D = 0.25

PACK_RULES = [
    ("POLITEC 100M", 12),
    ("POLITEC 250M", 6),
    ("POLITEC 30M", 15),
    ("PINTURA OLEO ATL 160CC", 3),
    ("PINTURA OLEO ATL 40CC", 3),
]

st.set_page_config(page_title="Agente de compras", layout="wide")

tcol1, tcol2 = st.columns([0.78, 0.22], vertical_alignment="center")
with tcol1:
    st.title("Agente de compras")
    st.caption(f"Versión app: {APP_VERSION}")
with tcol2:
    if st.button("Limpiar caché y reiniciar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

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

hist["Mes"] = hist["Mes"].fillna(0).astype("int16")
hist["Ventas"] = hist["Ventas"].astype("float32")
hist["Importe"] = hist["Importe"].astype("float32")

costo_unit_df = (
    hist.groupby("Código", as_index=False)
        .agg(Ventas_total=("Ventas", "sum"), Importe_total=("Importe", "sum"))
)

costo_unit_df["Costo_Unitario"] = np.where(
    costo_unit_df["Ventas_total"] > 0,
    costo_unit_df["Importe_total"] / costo_unit_df["Ventas_total"],
    np.nan
)

hist_total_24_25 = (
    hist.groupby("Código", as_index=False)["Ventas"]
        .sum()
        .rename(columns={"Ventas": "Hist_24_25_Ventas"})
)

nombre_hist = (
    hist.loc[hist["Nombre"] != "", ["Código", "Nombre"]]
    .drop_duplicates("Código")
    .rename(columns={"Nombre": "Nombre_hist"})
)

def p90_int(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return 0
    return int(np.percentile(x, 90))

g = (
    hist.groupby(["Código", "Mes"])["Ventas"]
        .apply(p90_int)
        .reset_index(name="P90")
)

p = g.pivot_table(index="Código", columns="Mes", values="P90", fill_value=0).reset_index()

for m in range(1, 13):
    if m not in p.columns:
        p[m] = 0

p = p.rename(columns={m: f"Max_M{m:02d}" for m in range(1, 13)})

max_mes_df = p.merge(nombre_hist, on="Código", how="left")

final = vs.merge(max_mes_df, on="Código", how="left")
final = final.merge(hist_total_24_25, on="Código", how="left")
final = final.merge(costo_unit_df[["Código", "Costo_Unitario"]], on="Código", how="left")

final["Hist_24_25_Ventas"] = final["Hist_24_25_Ventas"].fillna(0)

tz = ZoneInfo("America/Mazatlan")
hoy = datetime.now(tz).date()

dias_mes = calendar.monthrange(hoy.year, hoy.month)[1]
peso_actual = (dias_mes - hoy.day + 1) / dias_mes
peso_siguiente = 1 - peso_actual

mes_actual = hoy.month
mes_siguiente = 1 if mes_actual == 12 else mes_actual + 1

col_act = f"Max_M{mes_actual:02d}"
col_sig = f"Max_M{mes_siguiente:02d}"

base_hist = (peso_actual * final[col_act]) + (peso_siguiente * final[col_sig])

R = np.where(base_hist > 0, final["V30D"] / base_hist, 1.0)
R_eff = np.maximum(R, 1.0)

alpha_dyn = ALPHA_V30D / np.sqrt(R_eff)
alpha_dyn = np.clip(alpha_dyn, 0.0, ALPHA_V30D)

V30D_adj = base_hist * np.sqrt(R_eff)

demanda_mix = ((1 - alpha_dyn) * base_hist) + (alpha_dyn * V30D_adj)

final["Demanda30"] = np.where(
    final["Hist_24_25_Ventas"] <= 0,
    final["V30D"],
    np.where(base_hist <= 0, final["V30D"], demanda_mix)
)

final["Demanda30"] = np.ceil(final["Demanda30"]).astype(int)

final["Compra_Base"] = (final["Demanda30"] - final["Stock"]).clip(lower=0).astype(int)

name_norm = norm_name_series(final["Nombre"])
pack_info = name_norm.apply(detect_pack_rule)

final["Multiplo_Empaque"] = pack_info.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)

final["Compra"] = np.where(
    (final["Compra_Base"] > 0) & (final["Multiplo_Empaque"].fillna(0) > 0),
    [
        round_up_to_multiple(qty, int(mult))
        for qty, mult in zip(final["Compra_Base"], final["Multiplo_Empaque"].fillna(0))
    ],
    final["Compra_Base"]
).astype(int)

final["Importe_Compra"] = final["Compra"] * final["Costo_Unitario"]

tabla = final[
    [
        "Código",
        "EAN",
        "Nombre",
        "Compra",
        "Stock",
        "V30D",
        col_act,
        col_sig,
        "Demanda30",
        "Costo_Unitario",
        "Importe_Compra"
    ]
].rename(columns={
    col_act: f"P90Mes_{mes_actual:02d}",
    col_sig: f"P90Mes_{mes_siguiente:02d}",
})

st.subheader("Compra Sugerida")

st.dataframe(
    tabla.sort_values(["Compra", "Demanda30"], ascending=False),
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
