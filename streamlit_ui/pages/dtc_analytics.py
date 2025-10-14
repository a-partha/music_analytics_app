#pages/dtc_analytics.py
import streamlit as st
from music_analytics_app.agents.dtc_analytics.fan_engagement_analyzer import FanEngagementAnalyzer
from music_analytics_app.agents.dtc_analytics.platform_analyzer import PlatformAnalyzer
from music_analytics_app.streamlit_ui.components.charts import display_chart

def show_dtc_analytics():
    st.header("DTC & Fan Engagement Analytics")
    funnel_df = st.session_state.get("funnel_df")
    fan_analyzer = FanEngagementAnalyzer()
    result = fan_analyzer.analyze_funnel(funnel_df) if funnel_df is not None else None
    st.write("Fan Engagement Funnel", result)
    chart_path = display_chart("fan_funnel_chart.png")
    if chart_path:
        st.image(chart_path, caption="Fan Engagement Funnel Chart")

