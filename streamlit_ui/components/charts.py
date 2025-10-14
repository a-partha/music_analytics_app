#components/charts.py
import streamlit as st
import os

def display_chart(chart_name):
    chart_path = os.path.join("data/output", chart_name)
    if os.path.exists(chart_path):
        return chart_path
    st.warning(f"Chart not found: {chart_name}")
    return None
