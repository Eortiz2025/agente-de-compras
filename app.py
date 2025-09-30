import pandas as pd
import numpy as np
import streamlit as st
import io
import re
import math

st.set_page_config(page_title="Agente de Compras", page_icon="💼", layout="wide")
st.title("💼 Agente de Compras")
st.caption("Divisor fijo para V365: 342 (días hábiles). Días fijos: 30.")

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
        "V365", "Ventas 365 ($)"
    ]
    return df.dropna(how="all").reset_index(drop=True)

# -------- Entradas --------
archivo = st.file_uploader("🗂️ Sube el archivo exportado desde Erply (.xls)", type=["xls"])

colf = st.columns(3)
with colf[0]:
    proveedor_unico = st.checkbox("Filtrar por proveedor específico", value=False)
with colf[1]:
    mostrar_proveedor = st.checkbox("Mostrar Proveedor en resultados", value=False)  # oculto por defecto
with colf[2]:
    solo_stock_cero = st.checkbox("Solo Stock = 0", value=False)
solo_con_ventas_365 = st.checkbox("Solo con ventas en 365 días (>0)", value=False)

if not archivo:
    st.info("Sube el archivo para continuar.")
    st.stop()

# -------- Proceso --------
try:
    divisor_v365 = 342  # fijo
    dias = 30           # fijo

    tabla = _read_erply_xls_like_html(archivo)

    # Filtrado básico
    tabla = tabla[tabla["Proveedor"].astype(str).str.strip().ne("")]

    if proveedor_unico:
        provs = sorted(tabla["Proveedor"].dropna().astype(str).unique())
        proveedor_sel = st.selectbox("Proveedor:", provs)
        tabla = tabla[tabla["Proveedor"] == proveedor_sel]

    # Tipificación
    tabla["Stock"] = _to_num(tabla["Stock (total)"]).round()
    tabla["V30D"]  = _to_num(tabla["V30D"]).round()
    tabla["V365"]  = _to_num(tabla["V365"]).round()

    if solo_stock_cero:
        tabla = tabla[tabla["Stock"].eq(0)]
    if solo_con_ventas_365:
        tabla = tabla[tabla["V365"] > 0]

    # Cálculos
    tabla["VtaDiaria"] = tabla["V365"] / divisor_v365
    tabla["VtaProm"]   = np.rint(tabla["VtaDiaria"] * dias).astype(int)

    v30, vprom = tabla["V30D"], tabla["VtaProm"]
    intermedio = np.maximum(0.6 * v30 + 0.4 * vprom, v30)
    max_calc   = np.minimum(intermedio, 1.5 * v30)
    tabla["Max"] = np.where(v30.eq(0), 0.5 * vprom, max_calc)
    tabla["Max"] = np.rint(tabla["Max"]).astype(int)

    # Compra redondeada al múltiplo de 5 hacia arriba
    compra_raw = (tabla["Max"] - tabla["Stock"]).clip(lower=0)
    tabla["Compra"] = compra_raw.apply(lambda x: int(math.ceil(x/5.0)*5) if x > 0 else 0)

    # Salida: Compra inmediatamente después de Nombre
    cols = ["Código", "Nombre", "Compra", "Stock", "V365", "VtaProm", "V30D", "Max"]
    if "Código EAN" in tabla.columns:
        cols.insert(1, "Código EAN")
    if mostrar_proveedor:
        cols.insert(3, "Proveedor")  # se muestra sólo si marcas la casilla

    final = (tabla[tabla["Compra"] > 0]
             .sort_values("Nombre", na_position="last"))[cols]

    st.success("✅ Archivo procesado correctamente")
    st.dataframe(final, use_container_width=True, height=520)

    # Descarga Excel (.xlsx)
    exp = final.copy()
    for c in ["Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]:
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

    # Alerta
    st.subheader("🔥 Top 10: V30D > VtaProm (orden alfabético)")
    hot = exp[exp["V30D"] > exp["VtaProm"]].sort_values("Nombre").head(10)
    if hot.empty:
        st.info("✅ No hay productos con V30D > VtaProm.")
    else:
        st.dataframe(hot[["Código", "Nombre", "V365", "VtaProm", "V30D"]], use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}")
