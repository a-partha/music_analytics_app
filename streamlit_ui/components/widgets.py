#widgets.py
import streamlit as st

def file_upload_widget(label="Upload PDF Report"):
    return st.file_uploader(label, type=["pdf"])
