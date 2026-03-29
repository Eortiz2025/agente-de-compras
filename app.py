import streamlit as st
import pandas as pd
import numpy as np

APP_VERSION = "2026-03-28-v5.3 COLUMNAS"

ALPHA_V30D = 0.5

PACK_RULES = [
    ("HOJA EUROCOLOR", 100),
    ("HOJA OPALINA", 100),
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

def round_up(qty, mult):
    if pd.isna(qty) or qty <= 0 or pd.isna(mult) or mult <= 0:
        return int(qty) if qty > 0 else 0
    return int(np.ceil(qty / mult) * mult)

def detect_pack(name):
    n = str(name).upper()
    for p, m in PACK_RULES:
        if n.startswith(p):
            return m
    return np.nan

# =========================
# ERPLY PARSER
# =========================
def read_erply(file):
    tables = pd.read_html(file, header=None)
    df = max(tables, key=lambda x: x.shape[0])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    def is_code(x):
        s = str(x).strip()
        return len(s) >= 3 and not s.lower().startswith("codigo")

    start = 0
    for i in range(min(100, len(df))):
        if is_code(df.iloc[i, 1]):
            start = i
            break

    df = df.iloc[start:].reset_index(drop=True)

    out = pd.DataFrame({
        "Código": df.iloc[:, 1].astype(str).str.strip(),
        "Nombre": df.iloc[:, 3].astype(str).fillna(""),
        "V30D": pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0),
        "Stock": pd.to_numeric(df.iloc[:, 6], errors="coerce").fillna(0),
    })

    out["Código"] = norm_code(out["Código"])
    return out

# =========================
# UI
# =========================
st.title("Agente de compras V5.3")

hist_file = st.file_uploader("Histórico", type=["xlsx"])
erply_file = st.file_uploader("Erply", type=["xls"])

if hist_file is None or erply_file is None:
    st.stop()

hist = pd.read_excel(hist_file)
vs = read_erply(erply_file)

hist["Código"] = norm_code(hist["Código"])
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)
hist["Importe"] = pd.to_numeric(hist["Importe"], errors="coerce").fillna(0)
hist["Año"] = pd.to_numeric(hist["Año"], errors="coerce").fillna(0).astype(int)
hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce").fillna(0).astype(int)

# =========================
# NUEVAS COLUMNAS (ABRIL / MAYO 2025)
# =========================
v04 = (
    hist[(hist["Año"] == 2025) & (hist["Mes"] == 4)]
    .groupby("Código")["Ventas"]
    .sum()
    .rename("V04_2025")
)

v05 = (
    hist[(hist["Año"] == 2025) & (hist["Mes"] == 5)]
    .groupby("Código")["Ventas"]
    .sum()
    .rename("V05_2025")
)

# =========================
# COSTO
# =========================
cost = (
    hist[hist["Año"] == 2025]
    .groupby("Código")
    .agg({"Ventas": "sum", "Importe": "sum"})
    .reset_index()
)

cost["Costo"] = np.where(
    cost["Ventas"] > 0,
    cost["Importe"] / cost["Ventas"],
    np.nan
)

# =========================
# DEMANDA ESCOLAR
# =========================
hist_escolar = hist[hist["Mes"].between(4, 10)]

dem_2025 = (
    hist_escolar[hist_escolar["Año"] == 2025]
    .groupby("Código")["Ventas"]
    .sum()
)

dem_2024 = (
    hist_escolar[hist_escolar["Año"] == 2024]
    .groupby("Código")["Ventas"]
    .sum()
)

df = pd.DataFrame({
    "Dem_2025": dem_2025,
    "Dem_2024": dem_2024
}).fillna(0).reset_index()

df["Ratio"] = np.where(
    df["Dem_2024"] > 0,
    df["Dem_2025"] / df["Dem_2024"],
    1
)

def clas(r):
    if r < 0.7:
        return "SOBRECOMPRA"
    elif r <= 1.1:
        return "ALINEADO"
    else:
        return "SUBESTIMADO"

df["Tipo"] = df["Ratio"].apply(clas)

def demanda(row):
    if row["Tipo"] == "SOBRECOMPRA":
        return 0.9 * row["Dem_2025"] + 0.1 * row["Dem_2024"]
    elif row["Tipo"] == "ALINEADO":
        return 0.75 * row["Dem_2025"] + 0.25 * row["Dem_2024"]
    else:
        return 0.6 * row["Dem_2025"] + 0.4 * row["Dem_2024"]

df["Demanda_Base"] = df.apply(demanda, axis=1)
df["Demanda_Mensual"] = df["Demanda_Base"] / 7

# =========================
# MERGE
# =========================
final = vs.merge(df, on="Código", how="left")
final = final.merge(cost[["Código", "Costo"]], on="Código", how="left")
final = final.merge(v04, on="Código", how="left")
final = final.merge(v05, on="Código", how="left")

final["V04_2025"] = final["V04_2025"].fillna(0)
final["V05_2025"] = final["V05_2025"].fillna(0)

final["Demanda_Mensual"] = final["Demanda_Mensual"].fillna(final["V30D"])

# =========================
# DEMANDA FINAL
# =========================
final["Demanda30"] = np.ceil(
    (1 - ALPHA_V30D) * final["Demanda_Mensual"] +
    ALPHA_V30D * final["V30D"]
)

# =========================
# COMPRA
# =========================
final["Compra_Base"] = (final["Demanda30"] - final["Stock"]).clip(lower=0)

final["Compra_Base"] = np.where(
    (final["Compra_Base"] == 0) & (final["V30D"] > 3),
    1,
    final["Compra_Base"]
)

final["Multiplo"] = final["Nombre"].apply(detect_pack)

final["Compra"] = [
    round_up(q, m)
    for q, m in zip(final["Compra_Base"], final["Multiplo"])
]

# =========================
# IMPORTE
# =========================
final["Importe"] = final["Compra"] * final["Costo"]

# =========================
# COBERTURA
# =========================
final["Cobertura"] = np.where(
    final["Demanda30"] > 0,
    final["Stock"] / final["Demanda30"],
    1
)

def nivel(c):
    if c < 0.3:
        return "CRITICO"
    elif c < 0.8:
        return "MEDIO"
    else:
        return "SANO"

final["Nivel"] = final["Cobertura"].apply(nivel)

# =========================
# TABLA FINAL (ORDEN EXACTO)
# =========================
tabla = final[[
    "Código",
    "Nombre",
    "Compra",
    "Stock",
    "Demanda30",
    "V30D",
    "V04_2025",
    "V05_2025",
    "Costo",
    "Importe",
    "Nivel",
    "Tipo"
]]

tabla = tabla.sort_values("Importe", ascending=False)

# =========================
# METRICAS
# =========================
m1, m2, m3 = st.columns(3)
m1.metric("SKUs", len(tabla))
m2.metric("SKUs Compra", int((tabla["Compra"] > 0).sum()))
m3.metric("Importe Total", f"${tabla['Importe'].fillna(0).sum():,.0f}")

# =========================
# UI
# =========================
st.dataframe(tabla, use_container_width=True, height=600)

st.download_button(
    "Descargar CSV",
    tabla.to_csv(index=False).encode("utf-8-sig"),
    "compra_v5_3.csv"
)
