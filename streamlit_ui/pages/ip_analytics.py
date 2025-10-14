#pages/ip_analytics.py
import streamlit as st
from music_analytics_app.agents.ip_analytics.catalog_analyzer import CatalogAnalyzer
from music_analytics_app.agents.ip_analytics.export_power_analyzer import ExportPowerAnalyzer
from music_analytics_app.streamlit_ui.components.charts import display_chart

def show_ip_analytics():
    st.header("IP Value & Catalog Analytics")
    catalog_df = st.session_state.get("catalog_df")
    catalog_analyzer = CatalogAnalyzer()
    result = catalog_analyzer.growth_by_genre(catalog_df) if catalog_df is not None else None
    st.write("Catalog Growth by Genre/Age", result)
    chart_path = display_chart("catalog_growth_genre.png")
    if chart_path:
        st.image(chart_path, caption="Catalog Growth by Genre Chart")
