import pandas as pd
import streamlit as st
import io

st.set_page_config(page_title="Agente de Compras", page_icon="💼")
st.title("💼 Agente de Compras")

# Subida del archivo
archivo = st.file_uploader("🗂️ Sube el archivo exportado desde Erply (.xls)", type=["xls"])

# Preguntar número de días
dias = st.text_input("⏰ ¿Cuántos días deseas calcular para VtaProm? (Escribe un número)")

# Validar que sea un número entero positivo
if not dias.strip().isdigit() or int(dias) <= 0:
    st.warning("⚠️ Por favor escribe un número válido de días (mayor que 0) para continuar.")
    st.stop()

dias = int(dias)

if archivo:
    try:
        # ✅ CAMBIO: lectura corregida para no perder productos como "1KG"
        tabla = pd.read_html(archivo, header=3)[0]
        if tabla.columns[0] in ("", "Unnamed: 0", "No", "Moneda"):
            tabla = tabla.iloc[:, 1:]

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

        # 👉 Preguntar si desea calcular solo un proveedor
        calcular_proveedor = st.checkbox("¿Deseas calcular sólo un proveedor?", value=False)

        if calcular_proveedor:
            lista_proveedores = tabla["Proveedor"].dropna().unique()
            proveedor_seleccionado = st.selectbox("Selecciona el proveedor a calcular:", sorted(lista_proveedores))
            tabla = tabla[tabla["Proveedor"] == proveedor_seleccionado]

        # Convertir columnas numéricas
        tabla["V365"] = pd.to_numeric(tabla["V365"], errors="coerce").round()
        tabla["V30D"] = pd.to_numeric(tabla["V30D"], errors="coerce").round()
        tabla["Stock"] = pd.to_numeric(tabla["Stock"], errors="coerce").round()

        # ✅ CÁLCULO ACTUALIZADO
        tabla["VtaDiaria"] = (tabla["V365"] / 342).round(2)
        tabla["VtaProm"] = (tabla["VtaDiaria"] * dias).round()

        # Si V30D = 0, Max = 0.5 * VtaProm; si no, usar lógica avanzada
        max_calculado = []
        for i, row in tabla.iterrows():
            if row["V30D"] == 0:
                max_val = 0.5 * row["VtaProm"]
            else:
                intermedio = max(0.6 * row["V30D"] + 0.4 * row["VtaProm"], row["V30D"])
                max_val = min(intermedio, row["V30D"] * 1.5)
            max_calculado.append(round(max_val))

        tabla["Max"] = max_calculado
        tabla["Compra"] = (tabla["Max"] - tabla["Stock"]).clip(lower=0).round()

        # Eliminar columna VtaDiaria
        tabla = tabla.drop(columns=["VtaDiaria"])

        # Filtrar productos a comprar
        tabla = tabla[tabla["Compra"] > 0].sort_values("Nombre")

        # Preguntar si mostrar columna proveedor
        mostrar_proveedor = st.checkbox("¿Mostrar Proveedor?", value=False)

        # Reordenar columnas según si incluye proveedor
        if mostrar_proveedor:
            columnas_finales = [
                "Código", "Código EAN", "Nombre", "Proveedor", "Stock",
                "V365", "VtaProm", "V30D", "Max", "Compra"
            ]
        else:
            tabla = tabla.drop(columns=["Proveedor"])
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

        # --- SOLO UNA SECCIÓN: Top 10 mostrando Código, Nombre, V365, VtaProm, V30D ---
        st.subheader("🔥 Top 10 Productos donde V30D supera a VtaProm (Orden alfabético)")

        productos_calientes = tabla[tabla["V30D"] > tabla["VtaProm"]]

        if not productos_calientes.empty:
            productos_calientes = productos_calientes.sort_values("Nombre", ascending=True)
            top_productos = productos_calientes.head(10)
            columnas_a_mostrar = ["Código", "Nombre", "V365", "VtaProm", "V30D"]
            st.dataframe(top_productos[columnas_a_mostrar])
        else:
            st.info("✅ No hay productos con V30D mayores que VtaProm en este momento.")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
