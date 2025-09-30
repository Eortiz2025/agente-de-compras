import pandas as pd
import numpy as np
import streamlit as st
import io

st.set_page_config(page_title="Agente de Compras", page_icon="💼", layout="wide")
st.title("💼 Agente de Compras")

# ----------------------------
# Utilidades
# ----------------------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _dedupe_cols(cols):
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out

def leer_erply_xls_html(file_obj):
    """
    Erply exporta .xls en HTML. En este archivo, los encabezados reales están
    después de 4 filas 'basura'. Usamos skiprows=4 para tomar esa fila como encabezado.
    """
    file_obj.seek(0)
    df = pd.read_html(file_obj, skiprows=4)[0]
    # Asegurar nombres únicos y sin espacios raros
    df.columns = _dedupe_cols(df.columns)
    # Quitar filas totalmente vacías
    df = df.dropna(how="all")
    return df

# ----------------------------
# Entradas
# ----------------------------
archivo = st.file_uploader("🗂️ Sube el archivo exportado desde Erply (.xls)", type=["xls"])

colp = st.columns(3)
with colp[0]:
    dias = st.number_input("⏰ Días para VtaProm", min_value=1, step=1, value=30, format="%d")
with colp[1]:
    divisor_v365 = st.selectbox("📅 Divisor para V365", options=[365, 360, 342], index=0,
                                help="365 días naturales; 360 comercial; 342 si consideras días hábiles.")
with colp[2]:
    proveedor_unico = st.checkbox("Filtrar por proveedor específico", value=False)

colf = st.columns(3)
with colf[0]:
    solo_stock_cero = st.checkbox("Solo artículos con Stock = 0", value=False)
with colf[1]:
    solo_con_ventas_365 = st.checkbox("Solo artículos con ventas últimos 365 días (>0)", value=False)
with colf[2]:
    mostrar_proveedor = st.checkbox("Mostrar columna Proveedor", value=True)

if not archivo:
    st.info("Sube el archivo para continuar.")
    st.stop()

try:
    # Leer archivo
    tabla = leer_erply_xls_html(archivo)

    # Mapear columnas principales
    colmap = {}

    # Búsqueda flexible de columnas base
    def _find_col(busca):
        for c in tabla.columns:
            if str(c).strip().lower() == busca.lower():
                return c
        return None

    codigo_col   = _find_col("Código")
    ean_col      = _find_col("Código EAN")
    nombre_col   = _find_col("Nombre")
    stock_total  = _find_col("Stock (total)")
    proveedor_col= _find_col("Proveedor")

    # Cantidad vendida (dos periodos)
    cant_cols = [c for c in tabla.columns if str(c).strip().lower().startswith("cantidad vendida")]
    cant_cols = sorted(cant_cols, key=lambda x: (len(str(x)), str(x)))  # orden estable

    # Validaciones mínimas
    faltantes = []
    if codigo_col   is None: faltantes.append("Código")
    if nombre_col   is None: faltantes.append("Nombre")
    if stock_total  is None: faltantes.append("Stock (total)")
    if proveedor_col is None: faltantes.append("Proveedor")
    if len(cant_cols) < 1:   faltantes.append("Cantidad vendida (periodo corto)")
    if len(cant_cols) < 2:   faltantes.append("Cantidad vendida (365 días)")

    if faltantes:
        st.error("❌ Columnas faltantes: " + ", ".join(faltantes) +
                 "\nColumnas disponibles: " + ", ".join(map(str, tabla.columns)))
        st.stop()

    # Renombres a claves estables
    colmap[codigo_col]    = "Código"
    if ean_col is not None:
        colmap[ean_col]   = "Código EAN"
    colmap[nombre_col]    = "Nombre"
    colmap[stock_total]   = "Stock"
    colmap[proveedor_col] = "Proveedor"
    colmap[cant_cols[0]]  = "V30D"   # periodo corto del archivo
    colmap[cant_cols[1]]  = "V365"   # periodo 365 días

    tabla = tabla.rename(columns=colmap)

    # Filtrar filas sin proveedor real
    tabla = tabla[tabla["Proveedor"].astype(str).str.strip().ne("")]

    # Filtro por proveedor específico
    if proveedor_unico:
        prov_opt = sorted(tabla["Proveedor"].dropna().astype(str).unique())
        proveedor_sel = st.selectbox("Selecciona proveedor:", prov_opt)
        tabla = tabla[tabla["Proveedor"] == proveedor_sel]

    # Tipificación numérica
    tabla["Stock"] = _to_num(tabla["Stock"]).round()
    tabla["V30D"]  = _to_num(tabla["V30D"]).round()
    tabla["V365"]  = _to_num(tabla["V365"]).round()

    # Filtros de operación
    if solo_stock_cero:
        tabla = tabla[tabla["Stock"].eq(0)]
    if solo_con_ventas_365:
        tabla = tabla[tabla["V365"] > 0]

    # Cálculos
    tabla["VtaDiaria"] = (tabla["V365"] / divisor_v365)
    tabla["VtaProm"]   = np.rint(tabla["VtaDiaria"] * dias).astype(int)

    v30   = tabla["V30D"]
    vprom = tabla["VtaProm"]

    # Max vectorizado:
    # - Si V30D == 0 → 0.5 * VtaProm
    # - Si V30D > 0 → intermedio = max(0.6*V30D + 0.4*VtaProm, V30D); Max = min(intermedio, 1.5*V30D)
    intermedio = np.maximum(0.6 * v30 + 0.4 * vprom, v30)
    max_calc   = np.where(v30.eq(0), 0.5 * vprom, np.minimum(intermedio, 1.5 * v30))
    tabla["Max"]    = np.rint(max_calc).astype(int)
    tabla["Compra"] = (tabla["Max"] - tabla["Stock"]).clip(lower=0).astype(int)

    # Selección y orden
    cols_final = ["Código", "Nombre", "Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]
    if "Código EAN" in tabla.columns:
        cols_final.insert(1, "Código EAN")
    if mostrar_proveedor:
        cols_final.insert(3, "Proveedor")

    tabla_final = (tabla[tabla["Compra"] > 0]
                   .sort_values("Nombre", na_position="last"))[cols_final]

    st.success("✅ Archivo procesado correctamente")
    st.dataframe(tabla_final, use_container_width=True, height=520)

    # Descarga Excel
    export = tabla_final.copy()
    for c in ["Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]:
        if c in export.columns:
            export[c] = pd.to_numeric(export[c], errors="coerce").fillna(0).astype(int)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="Compra del día")
        ws = writer.sheets["Compra del día"]
        ws.freeze_panes = "A2"

    st.download_button(
        "📄 Descargar Excel",
        data=output.getvalue(),
        file_name="Compra del día.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Resumen de alerta: V30D > VtaProm
    st.subheader("🔥 Top 10: V30D > VtaProm (orden alfabético)")
    calientes = export[export["V30D"] > export["VtaProm"]].sort_values("Nombre").head(10)
    if calientes.empty:
        st.info("✅ No hay productos con V30D > VtaProm.")
    else:
        st.dataframe(calientes[["Código", "Nombre", "V365", "VtaProm", "V30D"]], use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}")
