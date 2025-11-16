import pandas as pd
import numpy as np
import streamlit as st
import io
import re
import math

st.set_page_config(page_title="Agente de Compras", page_icon="💼", layout="wide")
st.title("💼 Agente de Compras")
st.caption("Divisor fijo para V730: 684 (días hábiles). Días fijos: 30.")

# -------- Utilidades --------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _detect_data_start(df):
    """Primera fila donde col1 parece código y col3 es nombre de producto."""
    def is_code(x):
        s = str(x).strip()
        return bool(re.match(r"^[A-Za-z0-9\-]+$", s)) and len(s) >= 3 and "codigo" not in s.lower()
    for i in range(min(60, len(df))):
        c1 = df.iloc[i, 1] if df.shape[1] > 1 else None
        c3 = df.iloc[i, 3] if df.shape[1] > 3 else None
        if is_code(c1) and isinstance(c3, str) and len(str(c3).strip()) > 2 and "nombre" not in str(c3).lower():
            return i
    return 0

def _read_erply_xls_like_html(file_obj):
    """Lee el .xls (HTML) de Erply por posición."""
    file_obj.seek(0)
    df0 = pd.read_html(file_obj, header=None)[0]
    start = _detect_data_start(df0)
    df = df0.iloc[start:, :12].copy()
    df.columns = [
        "No", "Código", "Código EAN", "Nombre",
        "Stock (total)", "Stock (apartado)", "Stock (disponible)",
        "Proveedor",
        "V30D", "Ventas corto ($)",
        "V730", "Ventas 730 ($)"  # <---- CAMBIO AQUÍ
    ]
    return df.dropna(how="all").reset_index(drop=True)

def _norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

MISSING_PROV_TOKENS = {"", "nan", "none", "null", "s/n", "sin proveedor", "na"}

# -------- Entradas --------
archivo = st.file_uploader("🗂️ Sube el archivo exportado desde Erply (.xls)", type=["xls"])

colf = st.columns(3)
with colf[0]:
    proveedor_unico = st.checkbox("Filtrar por proveedor específico", value=False)
with colf[1]:
    mostrar_proveedor = st.checkbox("Mostrar Proveedor en resultados", value=False)
with colf[2]:
    solo_stock_cero = st.checkbox("Solo Stock = 0", value=False)
solo_con_ventas = st.checkbox("Solo con ventas en 730 días (>0)", value=False)

if not archivo:
    st.info("Sube el archivo para continuar.")
    st.stop()

# -------- Proceso --------
try:
    divisor_v730 = 684  # fijo 2 años hábiles
    dias = 30

    tabla = _read_erply_xls_like_html(archivo)

    # Exclusión descontinuados
    tabla["Proveedor_raw"] = tabla["Proveedor"]
    tabla["Proveedor_norm"] = tabla["Proveedor_raw"].apply(_norm_str)
    excl_mask = tabla["Proveedor_norm"].isin(MISSING_PROV_TOKENS)
    excluidos = int(excl_mask.sum())
    tabla = tabla.loc[~excl_mask].copy()

    tabla = tabla[tabla["Proveedor"].astype(str).str.strip().ne("")]

    if proveedor_unico:
        provs = sorted(
            p for p in tabla["Proveedor"].dropna().astype(str).str.strip().unique()
            if _norm_str(p) not in MISSING_PROV_TOKENS
        )
        proveedor_sel = st.selectbox("Proveedor:", provs)
        tabla = tabla[tabla["Proveedor"].astype(str).str.strip() == proveedor_sel]

    tabla["Stock"] = _to_num(tabla["Stock (total)"]).round()
    tabla["V30D"] = _to_num(tabla["V30D"]).round()
    tabla["V730"] = _to_num(tabla["V730"]).round()  # <---- CAMBIO

    if solo_stock_cero:
        tabla = tabla[tabla["Stock"].eq(0)]
    if solo_con_ventas:
        tabla = tabla[tabla["V730"] > 0]  # <---- CAMBIO

    # Nuevo cálculo de Promedio
    tabla["VtaDiaria"] = tabla["V730"] / divisor_v730  # <---- CAMBIO
    tabla["Prom"] = np.rint(tabla["VtaDiaria"] * dias).astype(int)  # <---- CAMBIO

    v30, vprom = tabla["V30D"], tabla["Prom"]
    intermedio = np.maximum(0.6 * v30 + 0.4 * vprom, v30)
    max_calc = np.minimum(intermedio, 1.5 * v30)
    tabla["Max"] = np.where(v30.eq(0), 0.5 * vprom, max_calc)
    tabla["Max"] = np.rint(tabla["Max"]).astype(int)

    compra_raw = (tabla["Max"] - tabla["Stock"]).clip(lower=0)
    tabla["Compra"] = compra_raw.apply(lambda x: int(math.ceil(x/5.0)*5) if x > 0 else 0)

    cols = ["Código", "Nombre", "Compra", "Stock", "V30D", "Max", "V730", "Prom"]
    if "Código EAN" in tabla.columns:
        cols.insert(1, "Código EAN")
    if mostrar_proveedor:
        cols.insert(3, "Proveedor")

    final = (tabla[tabla["Compra"] > 0]
             .sort_values("Nombre", na_position="last"))[cols]

    st.success("✅ Archivo procesado correctamente")
    if excluidos > 0:
        st.caption(f"🧹 Excluidos por proveedor vacío/descontinuado: {excluidos}")

    st.dataframe(final, use_container_width=True, height=520)

    exp = final.copy()
    for c in ["Stock", "V30D", "V730", "Prom", "Max", "Compra"]:
        if c in exp.columns:
            exp[c] = pd.to_numeric(exp[c], errors="coerce").fillna(0).astype(int)

    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        exp.to_excel(w, index=False, sheet_name="Compra del día")
        w.sheets["Compra del día"].freeze_panes = "A2"

    st.download_button(
        "📄 Descargar Excel (.xlsx)",
        data=out_xlsx.getvalue(),
        file_name="Compra del día.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("🔥 Top 10: V30D > Prom (orden alfabético)")
    hot = exp[exp["V30D"] > exp["Prom"]].sort_values("Nombre").head(10)
    if hot.empty:
        st.info("✅ No hay productos con V30D > Prom.")
    else:
        st.dataframe(hot[["Código", "Nombre", "V730", "Prom", "V30D"]], use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}")
