import pandas as pd
import streamlit as st
import io

st.set_page_config(page_title="Agente de Compras", page_icon="💼")
st.title("💼 Agente de Compras")

# Subida del archivo
archivo = st.file_uploader("🗂️ Sube el archivo exportado desde Erply (.xls)", type=["xls"])

# Preguntar número de días
dias = st.text_input("⏰ ¿Cuántos días deseas calcular para VtaProm? (Escribe un número)")

# Validar que sea un número
if not dias.strip().isdigit():
    st.warning("⚠️ Por favor escribe un número válido de días para continuar.")
    st.stop()

dias = int(dias)  # Convertir a número

if archivo:
    try:
        # Leer archivo
        tabla = pd.read_html(archivo, skiprows=3)[0].iloc[:, 1:]

        # Encabezados esperados
        columnas_deseadas = [
            "Código", "Código EAN", "Nombre",
            "Stock (total)", "Stock (apartado)", "Stock (disponible)",
            "Proveedor", "Cantidad vendida", "Ventas netas totales ($)",
            "Cantidad vendida (2)", "Ventas netas totales ($) (2)"
        ]

        if len(tabla.columns) >= len(columnas_deseadas):
            tabla.columns = columnas_deseadas[:len(tabla.columns)]
        else:
            st.error("❌ El archivo no tiene suficientes columnas.")
            st.stop()

        # Eliminar columnas innecesarias
        columnas_a_eliminar = [
            "Ventas netas totales ($)", "Stock (apartado)", "Stock (disponible)",
            "Ventas netas totales ($) (2)"
        ]
        tabla = tabla.drop(columns=columnas_a_eliminar)

        # Renombrar columnas
        tabla = tabla.rename(columns={
            "Stock (total)": "Stock",
            "Cantidad vendida": "V365",
            "Cantidad vendida (2)": "V30D"
        })

        # Filtrar productos sin proveedor
        tabla = tabla[tabla["Proveedor"].notna()]
        tabla = tabla[tabla["Proveedor"].astype(str).str.strip() != ""]

        # Convertir columnas numéricas
        tabla["V365"] = pd.to_numeric(tabla["V365"], errors="coerce").round()
        tabla["V30D"] = pd.to_numeric(tabla["V30D"], errors="coerce").round()
        tabla["Stock"] = pd.to_numeric(tabla["Stock"], errors="coerce").round()

        # Calcular VtaProm y demás
        tabla["VtaDiaria"] = (tabla["V365"] / 342).round(2)
        tabla["VtaProm"] = (tabla["VtaDiaria"] * dias).round()
        tabla["Max"] = tabla[["VtaProm", "V30D"]].max(axis=1).round()
        tabla["Compra"] = (tabla["Max"] - tabla["Stock"]).round()

        # Eliminar columnas internas
        tabla = tabla.drop(columns=["VtaDiaria", "Proveedor"])

        # Filtrar productos a comprar
        tabla = tabla[tabla["Compra"] > 0].sort_values("Nombre")

        # Reordenar columnas
        columnas_finales = [
            "Código", "Código EAN", "Nombre", "Stock",
            "V365", "VtaProm", "V30D", "Max", "Compra"
        ]
        tabla = tabla[columnas_finales]

        st.success("✅ Archivo procesado correctamente")
        st.dataframe(tabla)

        # Crear archivo Excel en memoria y congelar primera fila
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            tabla.to_excel(writer, index=False, sheet_name='Compra del día')
            workbook = writer.book
            worksheet = writer.sheets['Compra del día']
            worksheet.freeze_panes = worksheet['A2']

        processed_data = output.getvalue()

        # Botón de descarga
        st.download_button(
            label="📄 Descargar Excel",
            data=processed_data,
            file_name="Compra del día.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
