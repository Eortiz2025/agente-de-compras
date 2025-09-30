import pandas as pd
import numpy as np
import streamlit as st
import io

st.set_page_config(page_title="Agente de Compras", page_icon="💼", layout="wide")
st.title("💼 Agente de Compras")

# ----------------------------
# Utilidades
# ----------------------------
def _dedupe_cols(cols):
    """Evita columnas duplicadas añadiendo sufijos .1, .2... como hace pandas, pero estable."""
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out

def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _leer_erply_xls_html(file_obj):
    """
    Lee export de Erply (HTML dentro de .xls) y localiza la fila de encabezados real,
    que contiene 'Código' y 'Nombre'. Devuelve un DataFrame con encabezados correctos.
    """
    tablas = pd.read_html(file_obj, header=None)  # sin asumir header
    if not tablas:
        raise ValueError("No se detectaron tablas en el archivo.")
    df0 = tablas[0]
    # Buscar fila donde aparezca 'Código' y 'Nombre'
    header_idx = None
    for i in range(min(10, len(df0))):  # típicamente está en las primeras filas
        fila = df0.iloc[i].astype(str).str.strip()
        if ("Código" in set(fila)) and ("Nombre" in set(fila)):
            header_idx = i
            break
    if header_idx is None:
        # Fallback: intentar con header=1 como caso típico
        df = pd.read_html(file_obj, header=1)[0]
        return df

    # Usar esa fila como encabezados
    cols = df0.iloc[header_idx].astype(str).str.strip().tolist()
    cols = _dedupe_cols(cols)
    df = df0.iloc[header_idx+1:].copy()
    df.columns = cols
    df = df.dropna(how="all")  # quitar filas totalmente vacías
    return df

# ----------------------------
# Entrada de usuario
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
    # Importante: rebobinar el buffer antes de leer
    archivo.seek(0)
    tabla = _leer_erply_xls_html(archivo)

    # Normalización de posibles nombres esperados
    # Erply suele traer estas columnas (al menos):
    # "Código", "Código EAN", "Nombre", "Stock (total)", "Proveedor",
    # "Cantidad vendida" (periodo corto), "Ventas netas totales ($)" (periodo corto),
    # "Cantidad vendida.1" (últimos 365 días), "Ventas netas totales ($).1" (365)
    # Aseguramos que existan, y renombramos a claves consistentes.
    colmap = {}
    # Claves base
    for clave in ["Código", "Código EAN", "Nombre", "Stock (total)", "Proveedor"]:
        for c in tabla.columns:
            if str(c).strip().lower() == clave.lower():
                colmap[c] = clave
                break

    # Cantidades vendidas (dos periodos). Tomamos la primera como V30D y la segunda como V365.
    # Buscamos columnas que comiencen con "Cantidad vendida"
    cv_cols = [c for c in tabla.columns if str(c).strip().lower().startswith("cantidad vendida")]
    cv_cols = sorted(cv_cols, key=lambda x: (len(str(x)), str(x)))  # orden estable
    if len(cv_cols) >= 1:
        colmap[cv_cols[0]] = "V30D"
    if len(cv_cols) >= 2:
        colmap[cv_cols[1]] = "V365"

    # Renombrar
    tabla = tabla.rename(columns=colmap)

    # Validaciones mínimas
    requeridas = {"Código", "Nombre", "Proveedor", "Stock (total)", "V30D", "V365"}
    faltantes = [r for r in requeridas if r not in tabla.columns]
    if faltantes:
        st.error(f"❌ Columnas faltantes: {', '.join(faltantes)}. Revisa el export de Erply.")
        st.stop()

    # Filtro por proveedor (si corresponde)
    tabla = tabla[tabla["Proveedor"].astype(str).str.strip().ne("")]
    if proveedor_unico:
        prov_opt = sorted(tabla["Proveedor"].dropna().astype(str).unique())
        proveedor = st.selectbox("Selecciona el proveedor:", prov_opt)
        tabla = tabla[tabla["Proveedor"] == proveedor]

    # Tipificación numérica
    tabla["Stock"] = _to_num(tabla["Stock (total)"]).round()
    tabla["V30D"] = _to_num(tabla["V30D"]).round()
    tabla["V365"] = _to_num(tabla["V365"]).round()

    # Filtros operativos
    if solo_stock_cero:
        tabla = tabla[tabla["Stock"].eq(0)]
    if solo_con_ventas_365:
        tabla = tabla[tabla["V365"] > 0]

    # Cálculos
    tabla["VtaDiaria"] = (tabla["V365"] / divisor_v365)
    tabla["VtaProm"] = np.rint(tabla["VtaDiaria"] * dias).astype(int)

    v30 = tabla["V30D"]
    vprom = tabla["VtaProm"]

    # Max (vectorizado):
    # - Si V30D == 0 -> 0.5 * VtaProm
    # - Si no: intermedio = max(0.6*V30D + 0.4*VtaProm, V30D); Max = min(intermedio, 1.5*V30D)
    intermedio = np.maximum(0.6 * v30 + 0.4 * vprom, v30)
    max_calc = np.where(v30.eq(0), 0.5 * vprom, np.minimum(intermedio, 1.5 * v30))
    tabla["Max"] = np.rint(max_calc).astype(int)

    tabla["Compra"] = (tabla["Max"] - tabla["Stock"]).clip(lower=0).astype(int)

    # Selección de columnas finales
    base_cols = ["Código", "Código EAN", "Nombre", "Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]
    if mostrar_proveedor:
        final_cols = ["Código", "Código EAN", "Nombre", "Proveedor", "Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]
    else:
        final_cols = base_cols

    tabla_final = (tabla[tabla["Compra"] > 0]
                   .sort_values("Nombre", na_position="last"))[final_cols]

    st.success("✅ Archivo procesado correctamente")
    st.dataframe(tabla_final, use_container_width=True, height=500)

    # Descarga a Excel
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

    # 🔥 Productos calientes (demanda reciente > expectativa)
    st.subheader("🔥 Top 10: V30D > VtaProm (orden alfabético)")
    calientes = export[export["V30D"] > export["VtaProm"]].sort_values("Nombre").head(10)
    if calientes.empty:
        st.info("✅ No hay productos con V30D > VtaProm.")
    else:
        st.dataframe(calientes[["Código", "Nombre", "V365", "VtaProm", "V30D"]], use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}")
