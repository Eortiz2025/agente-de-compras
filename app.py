import pandas as pd
import numpy as np
import streamlit as st
import io
import re

st.set_page_config(page_title="Agente de Compras", page_icon="💼", layout="wide")
st.title("💼 Agente de Compras")

# -------------- Utilidades --------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def _dedupe(cols):
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

def _flatten_cols(mi_cols):
    """Aplana MultiIndex tomando preferentemente el segundo nivel ('Código', 'Nombre', etc.)."""
    flat = []
    for lvl0, lvl1 in mi_cols:
        c = (str(lvl1).strip() if lvl1 and "unnamed" not in str(lvl1).lower()
             else str(lvl0).strip())
        flat.append(c)
    return _dedupe(flat)

def _pick_sales_cols(mi_cols):
    """
    Regresa índices (posiciones) para V30D y V365 dentro del MultiIndex:
    - Busca columnas cuyo segundo nivel sea 'Cantidad vendida'.
    - Decide V30D por el rango más corto (p.ej. 08/09/2025–30/09/2025).
    - Decide V365 por el rango anual (p.ej. 30/09/2024–30/09/2025).
    """
    # Construir lista [(idx, lvl0_text, lvl1_text)]
    cand = []
    for i, (lvl0, lvl1) in enumerate(mi_cols):
        if str(lvl1).strip().lower() == "cantidad vendida":
            cand.append((i, str(lvl0), str(lvl1)))

    if not cand:
        return None, None

    # Heurística: si el lvl0 contiene un rango con 2024–2025 => V365
    # y si contiene solo fechas de 2025 y corta => V30D
    def _is_annual(s):
        s = s.replace(" ", "")
        return ("2024" in s and "2025" in s) or "365" in s

    v365_idx = None
    v30d_idx = None
    for i, lvl0, _ in cand:
        if _is_annual(lvl0):
            v365_idx = i

    # El V30D: el otro "Cantidad vendida" que no sea el anual
    for i, lvl0, _ in cand:
        if i != v365_idx:
            v30d_idx = i
            break

    # Fallback: si no detectó anual, toma el primero como V30D y segundo como V365
    if v365_idx is None and len(cand) >= 2:
        v30d_idx, v365_idx = cand[0][0], cand[1][0]
    elif v365_idx is None and len(cand) == 1:
        v30d_idx = cand[0][0]

    return v30d_idx, v365_idx

# -------------- Entradas --------------
archivo = st.file_uploader("🗂️ Sube el archivo exportado desde Erply (.xls)", type=["xls"])

colp = st.columns(3)
with colp[0]:
    dias = st.number_input("⏰ Días para VtaProm", min_value=1, step=1, value=30, format="%d")
with colp[1]:
    divisor_v365 = st.selectbox("📅 Divisor para V365", options=[365, 360, 342], index=0,
                                help="365 natural; 360 comercial; 342 hábiles.")
with colp[2]:
    proveedor_unico = st.checkbox("Filtrar por proveedor específico", value=False)

colf = st.columns(3)
with colf[0]:
    solo_stock_cero = st.checkbox("Solo Stock = 0", value=False)
with colf[1]:
    solo_con_ventas_365 = st.checkbox("Solo con ventas en 365 días (>0)", value=False)
with colf[2]:
    mostrar_proveedor = st.checkbox("Mostrar Proveedor", value=True)

if not archivo:
    st.info("Sube el archivo para continuar.")
    st.stop()

try:
    # ⚠️ Este archivo trae MultiIndex en columnas (dos filas de encabezado).
    archivo.seek(0)
    df = pd.read_html(archivo, header=[3, 4])[0]  # fila 3 = nivel 0, fila 4 = nivel 1

    if not isinstance(df.columns, pd.MultiIndex):
        st.error("El archivo no trae dos filas de encabezado como se esperaba.")
        st.stop()

    # Identificar columnas de ventas (V30D/V365) por el nivel 0 (texto con rangos)
    v30_idx, v365_idx = _pick_sales_cols(df.columns)
    if v30_idx is None:
        st.error("No se encontró columna de 'Cantidad vendida' (periodo corto).")
        st.stop()
    if v365_idx is None:
        st.warning("No se detectó claro el periodo anual; se continuará solo con V30D.")

    # Guardar nombres MultiIndex que usaremos antes de aplanar
    mi = df.columns
    col_codigo   = [i for i,(a,b) in enumerate(mi) if str(b).strip().lower()=="código"]
    col_ean      = [i for i,(a,b) in enumerate(mi) if str(b).strip().lower()=="código ean"]
    col_nombre   = [i for i,(a,b) in enumerate(mi) if str(b).strip().lower()=="nombre"]
    col_stock    = [i for i,(a,b) in enumerate(mi) if str(b).strip().lower()=="stock (total)"]
    col_prov     = [i for i,(a,b) in enumerate(mi) if str(b).strip().lower()=="proveedor"]

    # Aplanar columnas a un solo nivel legible
    flat_cols = _flatten_cols(df.columns)
    df.columns = flat_cols

    # Mapear a nombres estables
    colmap = {}
    # Campos base obligatorios
    if col_codigo:   colmap[df.columns[col_codigo[0]]] = "Código"
    if col_ean:      colmap[df.columns[col_ean[0]]]    = "Código EAN"
    if col_nombre:   colmap[df.columns[col_nombre[0]]] = "Nombre"
    if col_stock:    colmap[df.columns[col_stock[0]]]  = "Stock"
    if col_prov:     colmap[df.columns[col_prov[0]]]   = "Proveedor"
    # Ventas
    colmap[df.columns[v30_idx]] = "V30D"
    if v365_idx is not None:
        colmap[df.columns[v365_idx]] = "V365"

    df = df.rename(columns=colmap)

    # Validaciones mínimas
    requeridas = {"Código", "Nombre", "Stock", "Proveedor", "V30D"}
    falt = [r for r in requeridas if r not in df.columns]
    if falt:
        st.error("❌ Columnas faltantes: " + ", ".join(falt) +
                 "\nColumnas disponibles: " + ", ".join(df.columns))
        st.stop()

    # Limpiar proveedor y filtrar
    df = df[df["Proveedor"].astype(str).str.strip().ne("")]
    if proveedor_unico:
        provs = sorted(df["Proveedor"].dropna().astype(str).unique())
        sel = st.selectbox("Proveedor:", provs)
        df = df[df["Proveedor"] == sel]

    # Tipificación
    df["Stock"] = _to_num(df["Stock"]).round()
    df["V30D"]  = _to_num(df["V30D"]).round()
    if "V365" in df.columns:
        df["V365"] = _to_num(df["V365"]).round()
    else:
        df["V365"] = 0

    if solo_stock_cero:
        df = df[df["Stock"].eq(0)]
    if solo_con_ventas_365:
        df = df[df["V365"] > 0]

    # Cálculos
    df["VtaDiaria"] = df["V365"] / divisor_v365
    df["VtaProm"]   = np.rint(df["VtaDiaria"] * dias).astype(int)

    v30, vprom = df["V30D"], df["VtaProm"]
    intermedio = np.maximum(0.6*v30 + 0.4*vprom, v30)
    max_calc   = np.where(v30.eq(0), 0.5*vprom, np.minimum(intermedio, 1.5*v30))
    df["Max"]    = np.rint(max_calc).astype(int)
    df["Compra"] = (df["Max"] - df["Stock"]).clip(lower=0).astype(int)

    # Salida
    cols = ["Código", "Nombre", "Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]
    if "Código EAN" in df.columns:
        cols.insert(1, "Código EAN")
    if "Proveedor" in df.columns:
        if mostrar_proveedor and "Proveedor" not in cols:
            cols.insert(3, "Proveedor")
        elif not mostrar_proveedor and "Proveedor" in cols:
            cols.remove("Proveedor")

    final = (df[df["Compra"] > 0].copy()
             .sort_values("Nombre", na_position="last"))[cols]

    st.success("✅ Archivo procesado correctamente")
    st.dataframe(final, use_container_width=True, height=520)

    # Excel
    exp = final.copy()
    for c in ["Stock", "V365", "VtaProm", "V30D", "Max", "Compra"]:
        if c in exp.columns:
            exp[c] = pd.to_numeric(exp[c], errors="coerce").fillna(0).astype(int)

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        exp.to_excel(w, index=False, sheet_name="Compra del día")
        w.sheets["Compra del día"].freeze_panes = "A2"

    st.download_button("📄 Descargar Excel",
                       data=out.getvalue(),
                       file_name="Compra del día.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Alerta
    st.subheader("🔥 Top 10: V30D > VtaProm (orden alfabético)")
    hot = exp[exp["V30D"] > exp["VtaProm"]].sort_values("Nombre").head(10)
    if hot.empty:
        st.info("✅ No hay productos con V30D > VtaProm.")
    else:
        st.dataframe(hot[["Código", "Nombre", "V365", "VtaProm", "V30D"]], use_container_width=True)

except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}")
