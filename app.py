import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Agente de compras", layout="wide")
st.title("Agente de compras")

# =========================================================
# LECTURA ERPLY .XLS (HTML) por posiciones:
# B = Código (idx 1)
# D = Nombre (idx 3)
# E = V30D  (idx 4)
# G = Stock (idx 6)
# =========================================================
def _looks_like_table(df: pd.DataFrame) -> bool:
    if df is None or df.empty or df.shape[1] < 7:
        return False

    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()

    for i in range(min(120, len(df))):
        cB = df.iloc[i, 1] if df.shape[1] > 1 else None
        cD = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(cB) and isinstance(cD, str) and len(cD.strip()) > 2:
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

def _read_erply_html(uploaded) -> pd.DataFrame:
    data = uploaded.getvalue()
    tables = pd.read_html(io.BytesIO(data), header=None)

    chosen = None
    for t in tables:
        if _looks_like_table(t):
            chosen = t
            break
    if chosen is None:
        chosen = tables[0]

    start = _detect_data_start(chosen)

    # Tomar columnas suficientes para llegar a G
    raw = chosen.iloc[start:, :10].copy().reset_index(drop=True)

    out = pd.DataFrame({
        "Código": raw.iloc[:, 1].astype(str).str.strip(),
        "Nombre": raw.iloc[:, 3].astype(str).fillna(""),
        "V30D": pd.to_numeric(raw.iloc[:, 4], errors="coerce").fillna(0),   # E
        "Stock": pd.to_numeric(raw.iloc[:, 6], errors="coerce").fillna(0),  # G
    })

    # Quitar filas basura
    out = out[out["Código"].astype(str).str.strip().ne("")]
    out = out[out["Código"].astype(str).str.lower().ne("nan")]

    return out.reset_index(drop=True)

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
# LEER HISTÓRICO Y CALCULAR MAX POR MES (2024-2025)
# =========================================================
hist = pd.read_excel(hist_file, engine="openpyxl")

req = {"Código", "Nombre", "Año", "Mes", "Ventas"}
missing = req - set(hist.columns)
if missing:
    st.error(f"Histórico: faltan columnas {sorted(missing)}")
    st.stop()

hist = hist.copy()
hist["Código"] = hist["Código"].astype(str).str.strip()
hist["Nombre"] = hist["Nombre"].astype(str).fillna("")
hist["Año"] = pd.to_numeric(hist["Año"], errors="coerce")
hist["Mes"] = pd.to_numeric(hist["Mes"], errors="coerce")
hist["Ventas"] = pd.to_numeric(hist["Ventas"], errors="coerce").fillna(0)

hist = hist[hist["Año"].isin([2024, 2025])].copy()

g = hist.groupby(["Código", "Nombre", "Mes"], as_index=False)["Ventas"].max()

p = g.pivot_table(
    index=["Código", "Nombre"],
    columns="Mes",
    values="Ventas",
    aggfunc="max",
    fill_value=0
).reset_index()

for m in range(1, 13):
    if m not in p.columns:
        p[m] = 0

p = p.rename(columns={m: f"Max_M{m:02d}" for m in range(1, 13)})
max_mes_df = p.rename(columns={"Nombre": "Nombre_hist"})

# =========================================================
# LEER ERPLY
# =========================================================
vs = _read_erply_html(erply_file)
vs["V30D"] = pd.to_numeric(vs["V30D"], errors="coerce").fillna(0)
vs["Stock"] = pd.to_numeric(vs["Stock"], errors="coerce").fillna(0)

# =========================================================
# MERGE
# =========================================================
final = vs.merge(max_mes_df, on="Código", how="left")

final["Nombre_hist"] = final.get("Nombre_hist", "").fillna("")
final["Nombre"] = final["Nombre"].astype(str).fillna("")
final["Nombre"] = final["Nombre"].where(final["Nombre"].str.strip().ne(""), final["Nombre_hist"])

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
# DEMANDA 30 DÍAS (PONDERADA) + FALLBACK A V30D:
# Si (MaxMes_actual + MaxMes_sig) == 0 => usar V30D (si V30D>0)
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
# COMPRA SUGERIDA = max(0, Demanda30 - Stock) redondeo arriba
# =========================================================
final["Compra_sugerida"] = np.ceil(final["Demanda30"] - final["Stock"]).clip(lower=0).astype(int)

# =========================================================
# TABLA FINAL
# =========================================================
tabla = final[
    ["Código", "Nombre", "Stock", "V30D", col_act, col_sig, "Demanda30", "Compra_sugerida"]
].rename(columns={
    col_act: f"MaxMes_{mes_actual:02d}",
    col_sig: f"MaxMes_{mes_siguiente:02d}",
})

# =========================================================
# UI
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("SKUs Erply", f"{tabla['Código'].nunique():,}")
m2.metric("Suma Stock", f"{tabla['Stock'].sum():,.0f}")
m3.metric("Suma V30D (info)", f"{tabla['V30D'].sum():,.0f}")
m4.metric("Suma Compra sugerida", f"{tabla['Compra_sugerida'].sum():,.0f}")

st.subheader("Tabla unificada + compra sugerida")
st.dataframe(
    tabla.sort_values(["Compra_sugerida", "Demanda30"], ascending=False),
    use_container_width=True,
    height=600
)

csv = tabla.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar CSV",
    data=csv,
    file_name="compra_sugerida_30d_ponderada.csv",
    mime="text/csv"
)
