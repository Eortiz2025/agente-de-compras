import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Compra sugerida 30 días (ponderada)", layout="wide")
st.title("Compra sugerida – Inventario 30 días ponderado por mes")

# =========================================================
# LECTURA ERPLY .XLS (HTML)
# B = Código | D = Nombre | E = V30D | G = Stock
# =========================================================
def _looks_like_table(df):
    if df is None or df.empty or df.shape[1] < 7:
        return False

    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3

    for i in range(min(100, len(df))):
        if is_code(df.iloc[i, 1]) and isinstance(df.iloc[i, 3], str):
            return True
    return False

def _read_erply_html(uploaded):
    data = uploaded.getvalue()
    tables = pd.read_html(io.BytesIO(data), header=None)

    chosen = None
    for t in tables:
        if _looks_like_table(t):
            chosen = t
            break
    if chosen is None:
        chosen = tables[0]

    # encontrar inicio real de datos
    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3

    start = 0
    for i in range(min(150, len(chosen))):
        if is_code(chosen.iloc[i, 1]):
            start = i
            break

    df = chosen.iloc[start:, :8].copy()

    out = pd.DataFrame({
        "Código": df.iloc[:, 1].astype(str).str.strip(),
        "Nombre": df.iloc[:, 3].astype(str).fillna(""),
        "V30D": pd.to_numeric(df.iloc[:, 4], errors="coerce").fillna(0),
        "Stock": pd.to_numeric(df.iloc[:, 6], errors="coerce").fillna(0),
    })

    out = out[out["Código"] != ""]
    return out.reset_index(drop=True)

# =========================================================
# UPLOADERS
# =========================================================
c1, c2 = st.columns(2)

with c1:
    hist_file = st.file_uploader(
        "1) Histórico (.xlsx) — Código, Nombre, Año, Mes, Ventas",
        type=["xlsx"]
    )

with c2:
    erply_file = st.file_uploader(
        "2) Erply V30D + Stock (.xls)",
        type=["xls"]
    )

if hist_file is None or erply_file is None:
    st.info("Sube ambos archivos para continuar")
    st.stop()

# =========================================================
# HISTÓRICO
# =========================================================
hist = pd.read_excel(hist_file, engine="openpyxl")

req = {"Código", "Nombre", "Año", "Mes", "Ventas"}
if not req.issubset(hist.columns):
    st.error(f"Faltan columnas en histórico: {req - set(hist.columns)}")
    st.stop()

hist["Código"] = hist["Código"].astype(str).str.strip()
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)

# ---- Max mensual conservador (2024–2025)
hist = hist[hist["Año"].isin([2024, 2025])]

max_mes = (
    hist.groupby(["Código", "Nombre", "Mes"], as_index=False)["Ventas"]
    .max()
    .pivot_table(
        index=["Código", "Nombre"],
        columns="Mes",
        values="Ventas",
        fill_value=0
    )
    .reset_index()
)

for m in range(1, 13):
    if m not in max_mes.columns:
        max_mes[m] = 0

max_mes = max_mes.rename(columns={m: f"Max_M{m:02d}" for m in range(1, 13)})
max_mes = max_mes.rename(columns={"Nombre": "Nombre_hist"})

# =========================================================
# ERPLY
# =========================================================
vs = _read_erply_html(erply_file)

# =========================================================
# UNIÓN
# =========================================================
final = vs.merge(max_mes, on="Código", how="left")

final["Nombre_hist"] = final.get("Nombre_hist", "").fillna("")
final["Nombre"] = final["Nombre"].where(final["Nombre"] != "", final["Nombre_hist"])

# =========================================================
# DEMANDA 30 DÍAS PONDERADA
# =========================================================
tz = ZoneInfo("America/Mazatlan")
hoy = datetime.now(tz).date()

dias_mes = calendar.monthrange(hoy.year, hoy.month)[1]
dias_restantes = dias_mes - hoy.day + 1

peso_actual = dias_restantes / dias_mes
peso_siguiente = 1 - peso_actual

mes_actual = hoy.month
mes_siguiente = 1 if mes_actual == 12 else mes_actual + 1

col_act = f"Max_M{mes_actual:02d}"
col_sig = f"Max_M{mes_siguiente:02d}"

final[col_act] = pd.to_numeric(final[col_act], errors="coerce").fillna(0)
final[col_sig] = pd.to_numeric(final[col_sig], errors="coerce").fillna(0)

final["Demanda30"] = (
    peso_actual * final[col_act] +
    peso_siguiente * final[col_sig]
)

# =========================================================
# COMPRA SUGERIDA (TU REGLA)
# =========================================================
final["Compra_sugerida"] = np.ceil(final["Demanda30"] - final["Stock"]).clip(lower=0).astype(int)

# =========================================================
# SALIDA
# =========================================================
tabla = final[
    ["Código", "Nombre", "Stock", "V30D", col_act, col_sig, "Demanda30", "Compra_sugerida"]
].rename(columns={
    col_act: f"MaxMes_{mes_actual:02d}",
    col_sig: f"MaxMes_{mes_siguiente:02d}"
})

st.subheader("Resultado")
st.dataframe(
    tabla.sort_values("Compra_sugerida", ascending=False),
    use_container_width=True,
    height=600
)

csv = tabla.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar compra sugerida",
    data=csv,
    file_name="compra_sugerida_30d_ponderada.csv",
    mime="text/csv"
)
