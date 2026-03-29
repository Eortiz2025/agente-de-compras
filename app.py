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
APP_VERSION = "2026-03-28-v5.0 DEMANDA ESCOLAR + DELTA"

ALPHA_V30D = 0.25

PACK_RULES = [
    ("POLITEC 100M", 6),
    ("POLITEC 250M", 6),
    ("POLITEC 30M", 15),
    ("POLITEC 500M", 3),
    ("PINTURA OLEO ATL 40CC", 3),
    ("HOJA EUROCOLOR", 100),
    ("HOJA OPALINA", 100),
    ("CARTULINA CASCARON", 10),
    ("CARTULINA FLUORESCENTE", 10),
    ("CARTULINA BRISTOL", 25),
    ("PAPEL CHINA", 100),
    ("PAPEL CREPE", 10),
    ("PAPEL LUSTRE", 25),
    ("PLUMA BIC", 12),
]

st.set_page_config(page_title="Agente de compras", layout="wide")

# =========================
# HELPERS
# =========================
def norm_code(s):
    return s.astype(str).str.strip().str.upper()

def norm_name(s):
    return s.astype(str).fillna("").str.strip().str.upper()

def round_up(qty, mult):
    if pd.isna(qty) or qty <= 0 or mult <= 0:
        return 0
    return int(np.ceil(qty / mult) * mult)

def detect_pack(name):
    n = str(name).upper()
    for p, m in PACK_RULES:
        if n.startswith(p):
            return m
    return np.nan

# =========================
# UI
# =========================
st.title("Agente de compras V5")
st.caption(APP_VERSION)

hist_file = st.file_uploader("Histórico", type=["xlsx"])
erply_file = st.file_uploader("Erply", type=["xls"])

if hist_file is None or erply_file is None:
    st.stop()

# =========================
# LOAD
# =========================
hist = pd.read_excel(hist_file)
vs = pd.read_html(erply_file)[0]

vs = vs.rename(columns={
    vs.columns[1]: "Código",
    vs.columns[3]: "Nombre",
    vs.columns[4]: "V30D",
    vs.columns[6]: "Stock"
})

hist["Código"] = norm_code(hist["Código"])
vs["Código"] = norm_code(vs["Código"])

hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)
hist["Importe"] = pd.to_numeric(hist["Importe"], errors="coerce").fillna(0)

# =========================
# COSTO UNITARIO (2025)
# =========================
cost = hist[hist["Año"]==2025].groupby("Código").agg({
    "Ventas":"sum","Importe":"sum"
}).reset_index()

cost["Costo_Unitario"] = cost["Importe"] / cost["Ventas"]

# =========================
# DEMANDA ESCOLAR V5
# =========================
hist_escolar = hist[hist["Mes"].between(4,10)]

dem_2025 = hist_escolar[hist_escolar["Año"]==2025].groupby("Código")["Ventas"].sum()
dem_2024 = hist_escolar[hist_escolar["Año"]==2024].groupby("Código")["Ventas"].sum()

df = pd.DataFrame({
    "Dem_2025": dem_2025,
    "Dem_2024": dem_2024
}).fillna(0).reset_index()

df["Ratio"] = np.where(df["Dem_2024"]>0,
                       df["Dem_2025"]/df["Dem_2024"],1)

def clas(r):
    if r < 0.7: return "SOBRECOMPRA"
    elif r <= 1.1: return "ALINEADO"
    else: return "SUBESTIMADO"

df["Tipo"] = df["Ratio"].apply(clas)

def demanda(row):
    if row["Tipo"]=="SOBRECOMPRA":
        return 0.9*row["Dem_2025"] + 0.1*row["Dem_2024"]
    elif row["Tipo"]=="ALINEADO":
        return 0.75*row["Dem_2025"] + 0.25*row["Dem_2024"]
    elif row["Tipo"]=="SUBESTIMADO":
        return 0.6*row["Dem_2025"] + 0.4*row["Dem_2024"]
    else:
        return row["Dem_2025"]

df["Demanda_Base"] = df.apply(demanda, axis=1)
df["Demanda_Mensual"] = df["Demanda_Base"] / 7

# =========================
# MERGE
# =========================
final = vs.merge(df, on="Código", how="left")
final = final.merge(cost[["Código","Costo_Unitario"]], on="Código", how="left")

final["Demanda_Mensual"] = final["Demanda_Mensual"].fillna(final["V30D"])

# =========================
# DEMANDA FINAL
# =========================
final["Demanda30"] = np.ceil(
    0.7 * final["Demanda_Mensual"] +
    0.3 * final["V30D"]
)

# =========================
# COMPRA
# =========================
final["Compra_Base"] = (final["Demanda30"] - final["Stock"]).clip(lower=0)

name_norm = norm_name(final["Nombre"])
final["Multiplo"] = name_norm.apply(detect_pack)

final["Compra"] = np.where(
    (final["Compra_Base"] > 0) & (final["Multiplo"].fillna(0) > 0),
    [round_up(q, m) for q, m in zip(final["Compra_Base"], final["Multiplo"].fillna(0))],
    final["Compra_Base"]
)

# =========================
# IMPORTE
# =========================
final["Importe"] = final["Compra"] * final["Costo_Unitario"]

# =========================
# COBERTURA
# =========================
final["Cobertura"] = np.where(final["Demanda30"]>0,
                             final["Stock"]/final["Demanda30"],1)

def nivel(c):
    if c < 0.3: return "CRITICO"
    elif c < 0.8: return "MEDIO"
    else: return "SANO"

final["Nivel"] = final["Cobertura"].apply(nivel)

# =========================
# TABLA FINAL
# =========================
tabla = final[[
    "Código","Nombre","Stock",
    "Demanda30","Compra",
    "Costo_Unitario","Importe",
    "Nivel","Tipo"
]]

tabla = tabla.sort_values("Importe", ascending=False)

# =========================
# METRICAS
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("SKUs", len(tabla))
m2.metric("SKUs Compra", (tabla["Compra"]>0).sum())
m3.metric("Importe Total", f"${tabla['Importe'].sum():,.0f}")

# =========================
# UI
# =========================
st.dataframe(tabla, use_container_width=True, height=600)

st.download_button(
    "Descargar CSV",
    data=tabla.to_csv(index=False).encode("utf-8-sig"),
    file_name="Compra_v5.csv"
)
