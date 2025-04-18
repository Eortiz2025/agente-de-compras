import pandas as pd
import streamlit as st
import io
import matplotlib.pyplot as plt
import numpy as np

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

        # Calcular VtaProm y demás
        tabla["VtaDiaria"] = (tabla["V365"] / 342).round(2)
        tabla["VtaProm"] = (tabla["VtaDiaria"] * dias).round()
        tabla["Max"] = tabla[["VtaProm", "V30D"]].max(axis=1).round()
        tabla["Compra"] = (tabla["Max"] - tabla["Stock"]).round()

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

        # --- GRAFICAS EN PESTAÑAS ---
        tab1, tab2 = st.tabs(["🔥 Top por Volumen", "📈 Top por Crecimiento %"])

        with tab1:
            productos_calientes = tabla[tabla["V30D"] > tabla["VtaProm"]]

            if not productos_calientes.empty:
                st.subheader("🔥 Productos con ventas de 30 días superiores al promedio")
                productos_calientes["Diferencia"] = productos_calientes["V30D"] - productos_calientes["VtaProm"]
                top_productos = productos_calientes.sort_values("Diferencia", ascending=False).head(10)

                nombres = top_productos["Nombre"]
                vtaprom = top_productos["VtaProm"]
                v30d = top_productos["V30D"]

                x = np.arange(len(nombres))
                width = 0.35

                fig, ax = plt.subplots(figsize=(14, 7))
                barras1 = ax.bar(x - width/2, vtaprom, width, label='VtaProm', color='#1f77b4')
                barras2 = ax.bar(x + width/2, v30d, width, label='V30D', color='#2ca02c')

                for barra in barras1 + barras2:
                    height = barra.get_height()
                    ax.annotate('{}'.format(int(height)),
                                xy=(barra.get_x() + barra.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

                ax.set_ylabel('Unidades', fontsize=12)
                ax.set_xlabel('Productos', fontsize=12)
                ax.set_title('🔥 Comparativo VtaProm vs V30D (Top 10 donde V30D > VtaProm)', fontsize=16)
                ax.set_xticks(x)
                ax.set_xticklabels(nombres, rotation=45, ha='right', fontsize=10)
                ax.legend(fontsize=10)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)

                st.pyplot(fig)
            else:
                st.info("✅ No hay productos con V30D mayores que VtaProm en este momento.")

        with tab2:
            productos_crecimiento = tabla[tabla["V30D"] > tabla["VtaProm"]]

            if not productos_crecimiento.empty:
                st.subheader("📈 Top 10 Productos con mayor crecimiento porcentual en V30D respecto a VtaProm")
                productos_crecimiento["Crecimiento %"] = ((productos_crecimiento["V30D"] - productos_crecimiento["VtaProm"]) / productos_crecimiento["VtaProm"]) * 100
                productos_crecimiento["Crecimiento %"] = productos_crecimiento["Crecimiento %"].replace([np.inf, -np.inf], np.nan).fillna(0)

                top_crecimiento = productos_crecimiento.sort_values("Crecimiento %", ascending=False).head(10)

                nombres = top_crecimiento["Nombre"]
                crecimiento = top_crecimiento["Crecimiento %"]

                x = np.arange(len(nombres))

                fig, ax = plt.subplots(figsize=(14, 7))
                barras = ax.bar(x, crecimiento, color='#ff7f0e')

                for barra in barras:
                    height = barra.get_height()
                    ax.annotate(f'{height:.1f}%', 
                                xy=(barra.get_x() + barra.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

                ax.set_ylabel('Crecimiento %', fontsize=12)
                ax.set_xlabel('Productos', fontsize=12)
                ax.set_title('📈 Crecimiento porcentual V30D vs VtaProm (Top 10)', fontsize=16)
                ax.set_xticks(x)
                ax.set_xticklabels(nombres, rotation=45, ha='right', fontsize=10)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)

                st.pyplot(fig)
            else:
                st.info("✅ No hay productos con V30D mayores que VtaProm en este momento.")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
