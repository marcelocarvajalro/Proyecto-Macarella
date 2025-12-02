import pandas as pd
from prophet import Prophet
import json
import os
import streamlit as st
from datetime import datetime
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Configuración de la página debe ser la primera llamada a Streamlit
st.set_page_config(page_title="Sistema de Inteligencia de Demanda", layout="wide", page_icon="📈")

class DemandForecastSystem:
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        self.datasets_file = os.path.join(data_dir, "datasets_metadata.json")
        self.ensure_data_directory()
        self.datasets = self.load_datasets_metadata()

    def ensure_data_directory(self):
        """Asegura que el directorio de datos exista"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_datasets_metadata(self) -> Dict:
        """Carga los metadatos de los datasets"""
        if os.path.exists(self.datasets_file):
            with open(self.datasets_file, 'r') as f:
                return json.load(f)
        return {}

    def save_datasets_metadata(self):
        """Guarda los metadatos de los datasets"""
        with open(self.datasets_file, 'w') as f:
            json.dump(self.datasets, f, indent=4)

    def process_complex_excel(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Procesa un archivo Excel con múltiples hojas (una por mes).
        Retorna: (DataFrame procesado, Mensaje de error/éxito)
        """
        try:
            # Obtener nombres de todas las hojas
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            
            all_data_frames = []
            
            for sheet in sheet_names:
                try:
                    # 1. Detección de Fecha (Row 1)
                    header_df = pd.read_excel(file_path, sheet_name=sheet, header=None, nrows=1)
                    if header_df.empty: continue
                    
                    raw_date_str = str(header_df.iloc[0, 0]).strip().upper()
                    
                    # Mapeo de meses
                    spanish_months = {
                        'ENERO': 'JANUARY', 'FEBRERO': 'FEBRUARY', 'MARZO': 'MARCH',
                        'ABRIL': 'APRIL', 'MAYO': 'MAY', 'JUNIO': 'JUNE',
                        'JULIO': 'JULY', 'AGOSTO': 'AUGUST', 'SEPTIEMBRE': 'SEPTEMBER',
                        'OCTUBRE': 'OCTOBER', 'NOVIEMBRE': 'NOVEMBER', 'DICIEMBRE': 'DECEMBER'
                    }
                    
                    for es, en in spanish_months.items():
                        if es in raw_date_str:
                            raw_date_str = raw_date_str.replace(es, en)
                            break
                    
                    try:
                        report_date = pd.to_datetime(raw_date_str)
                    except:
                        # Si falla la fecha en una hoja, la saltamos o logueamos error
                        continue

                    # 2. Estructura Matricial (Row 2 headers)
                    df_raw = pd.read_excel(file_path, sheet_name=sheet, header=1)

                    if "PRINT" not in df_raw.columns:
                        continue

                    # 3. Limpieza y Transformación
                    value_vars = [c for c in df_raw.columns if c != "PRINT" and "TOTAL" not in str(c).upper()]
                    
                    df_melted = df_raw.melt(
                        id_vars=["PRINT"],
                        value_vars=value_vars,
                        var_name="Channel",
                        value_name="Quantity"
                    )

                    df_melted["Quantity"] = df_melted["Quantity"].replace({'-': 0, 'N/A': 0, 'n/a': 0})
                    df_melted["Quantity"] = pd.to_numeric(df_melted["Quantity"], errors='coerce').fillna(0)
                    df_melted["Date"] = report_date
                    df_final_sheet = df_melted.rename(columns={"PRINT": "Product"})
                    
                    # Filtros de seguridad
                    df_final_sheet = df_final_sheet[~df_final_sheet["Product"].astype(str).str.upper().str.contains("TOTAL", na=False)]
                    df_final_sheet = df_final_sheet[~df_final_sheet["Channel"].astype(str).str.upper().str.contains("TOTAL", na=False)]
                    
                    all_data_frames.append(df_final_sheet)
                    
                except Exception as e:
                    print(f"Error procesando hoja {sheet}: {e}")
                    continue

            if not all_data_frames:
                return None, "No se pudieron extraer datos válidos de ninguna hoja."

            # Unir todo
            full_df = pd.concat(all_data_frames, ignore_index=True)
            full_df = full_df[["Date", "Product", "Channel", "Quantity"]]
            
            return full_df, "Success"

        except Exception as e:
            return None, f"Error general en ETL: {str(e)}"

    def add_dataset(self, name: str, description: str, file_path: str, frequency: str):
        """Agrega un nuevo dataset al sistema usando el ETL complejo"""
        dataset_id = f"dataset_{len(self.datasets) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Procesar ETL
        df, message = self.process_complex_excel(file_path)
        
        if df is None:
            return False, message

        # Guardar una copia del dataset
        dataset_filename = f"{dataset_id}.parquet"
        dataset_path = os.path.join(self.data_dir, dataset_filename)

        # Guardar como parquet
        df.to_parquet(dataset_path)

        # Guardar metadatos
        self.datasets[dataset_id] = {
            "name": name,
            "description": description,
            "filename": dataset_filename,
            "date_column": "Date",
            "demand_column": "Quantity", # Default, aunque ahora es más granular
            "frequency": frequency,
            "created_at": datetime.now().isoformat(),
            "rows": len(df),
            "date_range": {
                "start": df["Date"].min().isoformat(),
                "end": df["Date"].max().isoformat()
            },
            "products_count": df["Product"].nunique(),
            "channels_count": df["Channel"].nunique()
        }

        self.save_datasets_metadata()
        return True, dataset_id

    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Obtiene un dataset por su ID"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} no encontrado")

        dataset_info = self.datasets[dataset_id]
        dataset_path = os.path.join(self.data_dir, dataset_info["filename"])

        df = pd.read_parquet(dataset_path)
        return df

    def delete_dataset(self, dataset_id: str):
        """Elimina un dataset del sistema"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} no encontrado")

        dataset_info = self.datasets[dataset_id]
        dataset_path = os.path.join(self.data_dir, dataset_info["filename"])

        if os.path.exists(dataset_path):
            os.remove(dataset_path)

        del self.datasets[dataset_id]
        self.save_datasets_metadata()

    def get_descriptive_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Genera estadísticas descriptivas para el dashboard"""
        df = self.get_dataset(dataset_id)
        
        stats = {
            "total_sales": df["Quantity"].sum(),
            "total_records": len(df),
            "date_range": (df["Date"].min(), df["Date"].max()),
            "top_products": df.groupby("Product")["Quantity"].sum().sort_values(ascending=False).head(10),
            "sales_by_channel": df.groupby("Channel")["Quantity"].sum().sort_values(ascending=False),
            "heatmap_data": df.pivot_table(index="Product", columns="Channel", values="Quantity", aggfunc="sum").fillna(0)
        }
        
        # Mejor producto y canal
        stats["best_product"] = stats["top_products"].index[0] if not stats["top_products"].empty else "N/A"
        stats["best_channel"] = stats["sales_by_channel"].index[0] if not stats["sales_by_channel"].empty else "N/A"
        
        return stats

    def generate_forecast(self, dataset_id: str, periods: int, 
                          target_product: str = "All", target_channel: str = "All",
                          include_history: bool = True,
                          country_holidays: str = None,
                          seasonality_mode: str = 'additive') -> Dict[str, Any]:
        """Genera un pronóstico avanzado"""
        df = self.get_dataset(dataset_id)
        dataset_info = self.datasets[dataset_id]

        # Filtrar datos según selección
        mask = pd.Series(True, index=df.index)
        if target_product != "All":
            mask &= (df["Product"] == target_product)
        if target_channel != "All":
            mask &= (df["Channel"] == target_channel)
            
        filtered_df = df[mask]
        
        # Agrupar por fecha para tener la serie temporal
        ts_df = filtered_df.groupby("Date")["Quantity"].sum().reset_index()
        ts_df.columns = ["ds", "y"]

        # Resampling para llenar huecos
        ts_df = ts_df.set_index("ds").resample(dataset_info["frequency"]).sum().reset_index()

        # VALIDACIÓN: Prophet necesita al menos 2 puntos de datos
        if len(ts_df) < 2:
            dates_str = ", ".join(ts_df['ds'].dt.strftime('%Y-%m-%d').astype(str).tolist())
            raise ValueError(f"Datos insuficientes para pronosticar. Se requieren al menos 2 fechas históricas, pero solo se encontraron: {dates_str}. \n\nConsejo: Sube un archivo con más historial o asegúrate de no filtrar demasiado los datos.")

        # Configurar Prophet
        model = Prophet(seasonality_mode=seasonality_mode)
        
        if country_holidays:
            try:
                model.add_country_holidays(country_name=country_holidays)
            except Exception as e:
                st.warning(f"No se pudieron cargar festivos para {country_holidays}: {e}")

        model.fit(ts_df)

        future = model.make_future_dataframe(
            periods=periods,
            freq=dataset_info["frequency"],
            include_history=include_history
        )

        forecast = model.predict(future)

        return {
            "model": model,
            "forecast": forecast,
            "original_data": ts_df,
            "dataset_info": dataset_info,
            "filters": {"product": target_product, "channel": target_channel}
        }

    def get_forecast_metrics(self, forecast_result: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de calidad"""
        merged = forecast_result["forecast"].merge(
            forecast_result["original_data"],
            on="ds",
            how="left"
        )
        merged = merged[merged["y"].notna()]

        if len(merged) == 0:
            return {}

        # Evitar división por cero en MAPE
        y_true = merged["y"]
        y_pred = merged["yhat"]
        
        # Métricas robustas
        mae = abs(y_true - y_pred).mean()
        rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
        
        # MAPE modificado (si y=0, ignorar o usar epsilon)
        mask = y_true != 0
        if mask.any():
            mape = (abs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100
        else:
            mape = 0.0

        return {
            "MAPE": mape,
            "MAE": mae,
            "RMSE": rmse,
            "R-squared": 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
        }

# --- Interfaz de Usuario ---

def main():
    st.title("🚀 Suite de Inteligencia de Demanda")
    st.markdown("Sistema avanzado de ETL, Análisis y Pronóstico de Ventas")

    if "forecast_system" not in st.session_state:
        st.session_state.forecast_system = DemandForecastSystem()

    system = st.session_state.forecast_system

    # Sidebar mejorado
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
        st.header("Navegación")
        menu_option = st.radio(
            "",
            ["📁 Carga & ETL", "📊 Análisis Descriptivo", "🔮 Motor de Pronósticos", "📋 Resultados"]
        )
        st.divider()
        st.info("v2.0 - Enterprise Edition")

    # --- Módulo 1: Carga & ETL ---
    if menu_option == "📁 Carga & ETL":
        st.header("Ingestión de Datos Complejos")
        st.markdown("""
        Sube tus reportes mensuales en formato matricial. El sistema detectará automáticamente:
        - Fecha del reporte (Fila 1)
        - Productos y Canales (Matriz)
        - Limpieza de datos nulos
        """)

        tab1, tab2 = st.tabs(["📥 Nueva Carga", "🗄️ Datasets Disponibles"])

        with tab1:
            uploaded_file = st.file_uploader("Subir Excel Matricial", type=["xlsx", "xls"])
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                col1, col2 = st.columns(2)
                with col1:
                    dataset_name = st.text_input("Nombre del Dataset", placeholder="Ej: Ventas Q3 2024")
                with col2:
                    freq_options = ["D", "W", "M", "Q", "Y"]
                    frequency = st.selectbox("Frecuencia de los datos", options=freq_options, index=2) # Default Monthly

                dataset_desc = st.text_area("Descripción / Notas")

                if st.button("🚀 Procesar e Ingestar", type="primary"):
                    if not dataset_name:
                        st.error("Por favor asigna un nombre al dataset.")
                    else:
                        with st.spinner("Ejecutando ETL complejo..."):
                            success, result = system.add_dataset(
                                dataset_name, dataset_desc, temp_path, frequency
                            )
                            
                            if success:
                                st.balloons()
                                st.success(f"¡Dataset '{dataset_name}' procesado exitosamente! ID: {result}")
                                try:
                                    os.unlink(temp_path)
                                except PermissionError:
                                    pass # Windows a veces mantiene el archivo bloqueado brevemente
                            else:
                                st.error(f"Error en ETL: {result}")

        with tab2:
            if not system.datasets:
                st.info("No hay datasets. Sube uno en la pestaña anterior.")
            else:
                for d_id, info in system.datasets.items():
                    with st.expander(f"📂 {info['name']} ({info['created_at'][:10]})"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Filas (Transaccional)", info['rows'])
                        c2.metric("Productos", info.get('products_count', 'N/A'))
                        c3.metric("Canales", info.get('channels_count', 'N/A'))
                        
                        if st.button("🗑️ Eliminar", key=d_id):
                            system.delete_dataset(d_id)
                            st.rerun()

    # --- Módulo 2: Análisis Descriptivo ---
    elif menu_option == "📊 Análisis Descriptivo":
        st.header("Dashboard de Inteligencia de Negocio")
        
        if not system.datasets:
            st.warning("Carga datos primero.")
        else:
            d_names = {info["name"]: id for id, info in system.datasets.items()}
            sel_name = st.selectbox("Seleccionar Dataset", list(d_names.keys()))
            
            if sel_name:
                d_id = d_names[sel_name]
                stats = system.get_descriptive_stats(d_id)
                
                # KPIs
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Ventas Totales", f"{stats['total_sales']:,.0f}")
                k2.metric("Mejor Producto", stats['best_product'])
                k3.metric("Mejor Canal", stats['best_channel'])
                k4.metric("Registros", stats['total_records'])
                
                st.divider()
                
                # Gráficos
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("🏆 Top 10 Productos")
                    fig_prod = px.bar(
                        x=stats['top_products'].values,
                        y=stats['top_products'].index,
                        orientation='h',
                        labels={'x': 'Ventas', 'y': 'Producto'},
                        color=stats['top_products'].values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_prod, use_container_width=True)
                    
                with c2:
                    st.subheader("🏪 Ventas por Canal")
                    fig_chan = px.pie(
                        values=stats['sales_by_channel'].values,
                        names=stats['sales_by_channel'].index,
                        hole=0.4
                    )
                    st.plotly_chart(fig_chan, use_container_width=True)
                
                st.subheader("🔥 Mapa de Calor: Producto vs Canal")
                fig_heat = px.imshow(
                    stats['heatmap_data'],
                    labels=dict(x="Canal", y="Producto", color="Ventas"),
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    # --- Módulo 3: Motor de Pronósticos ---
    elif menu_option == "🔮 Motor de Pronósticos":
        st.header("Configuración de Pronóstico Avanzado")
        
        if not system.datasets:
            st.warning("Carga datos primero.")
        else:
            d_names = {info["name"]: id for id, info in system.datasets.items()}
            sel_name = st.selectbox("Seleccionar Dataset Base", list(d_names.keys()))
            
            if sel_name:
                d_id = d_names[sel_name]
                df = system.get_dataset(d_id)
                
                # Filtros de segmentación
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    prods = ["All"] + sorted(df["Product"].unique().tolist())
                    sel_prod = st.selectbox("Filtrar por Producto", prods)
                with col_f2:
                    chans = ["All"] + sorted(df["Channel"].unique().tolist())
                    sel_chan = st.selectbox("Filtrar por Canal", chans)
                
                st.divider()
                
                # Parámetros del Modelo
                st.subheader("Parámetros del Modelo Prophet")
                p1, p2, p3 = st.columns(3)
                
                with p1:
                    periods = st.number_input("Horizonte (Periodos)", 1, 60, 12)
                with p2:
                    seasonality = st.selectbox("Estacionalidad", ["additive", "multiplicative"])
                with p3:
                    holidays = st.selectbox("Días Festivos", [None, "CR", "MX", "US", "ES", "BR"])

                if st.button("✨ Generar Pronóstico", type="primary"):
                    with st.spinner("Entrenando modelo y proyectando futuro..."):
                        try:
                            res = system.generate_forecast(
                                d_id, periods, sel_prod, sel_chan, 
                                True, holidays, seasonality
                            )
                            st.session_state.current_forecast = res
                            st.success("¡Pronóstico completado!")
                            
                            # Redirigir visualmente a resultados (opcional, o mostrar aquí)
                            st.info("Ve a la pestaña 'Resultados' para ver el detalle.")
                            
                        except Exception as e:
                            st.error(f"Error en el modelo: {e}")

    # --- Módulo 4: Resultados ---
    elif menu_option == "📋 Resultados":
        st.header("Resultados del Pronóstico")
        
        if "current_forecast" not in st.session_state:
            st.info("Genera un pronóstico primero.")
        else:
            res = st.session_state.current_forecast
            metrics = system.get_forecast_metrics(res)
            
            # Filtros aplicados
            st.caption(f"Filtros: Producto={res['filters']['product']} | Canal={res['filters']['channel']}")
            
            # Métricas
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAPE (Error %)", f"{metrics.get('MAPE', 0):.2f}%", delta_color="inverse")
            m2.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
            m3.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
            m4.metric("R² (Ajuste)", f"{metrics.get('R-squared', 0):.4f}")
            
            # Gráfico Principal
            fig = go.Figure()
            
            # Histórico
            fig.add_trace(go.Scatter(
                x=res["original_data"]["ds"], 
                y=res["original_data"]["y"],
                name="Datos Reales",
                mode='lines+markers',
                line=dict(color='#1f77b4')
            ))
            
            # Predicción
            fig.add_trace(go.Scatter(
                x=res["forecast"]["ds"], 
                y=res["forecast"]["yhat"],
                name="Pronóstico",
                line=dict(color='#ff7f0e', dash='dash')
            ))
            
            # Intervalo de confianza
            fig.add_trace(go.Scatter(
                x=res["forecast"]["ds"].tolist() + res["forecast"]["ds"].tolist()[::-1],
                y=res["forecast"]["yhat_upper"].tolist() + res["forecast"]["yhat_lower"].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name="Intervalo Confianza"
            ))
            
            fig.update_layout(
                title="Proyección de Demanda",
                xaxis_title="Fecha",
                yaxis_title="Cantidad",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Componentes del modelo
            with st.expander("Ver Componentes del Modelo (Tendencia, Estacionalidad)"):
                fig2 = res["model"].plot_components(res["forecast"])
                st.pyplot(fig2)

            # Descarga
            csv = res["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)
            st.download_button(
                "📥 Descargar Pronóstico CSV",
                csv,
                "pronostico_demanda.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
