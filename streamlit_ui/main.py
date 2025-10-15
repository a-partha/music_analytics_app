# #main.py
# import streamlit as st
# from music_analytics_app.agents.dtc_analytics.fan_engagement_analyzer import FanEngagementAnalyzer
# from music_analytics_app.agents.dtc_analytics.platform_analyzer import PlatformAnalyzer
# from music_analytics_app.streamlit_ui.components.widgets import file_upload_widget
# from music_analytics_app.streamlit_ui.components.charts import display_chart

# st.set_page_config(page_title="Music Analytics Dashboard", layout="wide")

# file_upload_widget()


# st.sidebar.title("Navigation")

# pages = {
#     "Dashboard": "pages/dashboard.py",
#     "DTC Analytics": "pages/dtc_analytics.py",
#     "IP Analytics": "pages/ip_analytics.py"
# }

# selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# if selected_page == "Dashboard":
#     from music_analytics_app.streamlit_ui.pages.dashboard import show_dashboard
#     show_dashboard()
# elif selected_page == "DTC Analytics":
#     from music_analytics_app.streamlit_ui.pages.dtc_analytics import show_dtc_analytics
#     show_dtc_analytics()
# elif selected_page == "IP Analytics":
#     from music_analytics_app.streamlit_ui.pages.ip_analytics import show_ip_analytics
#     show_ip_analytics()

import streamlit as st
from music_analytics_app.streamlit_ui.components.widgets import file_upload_widget
from music_analytics_app.streamlit_ui.components.charts import display_chart

# ✅ MUST be the very first Streamlit command
st.set_page_config(page_title="Music Analytics Dashboard", layout="wide")

# File upload widget
uploaded_file = file_upload_widget()

# Process uploaded file
if uploaded_file is not None:
    # Save uploaded file temporarily
    import tempfile
    import os
    from music_analytics_app.agents.data_extraction.pdf_extractor import PDFExtractor
    from music_analytics_app.agents.data_extraction.data_validator import DataValidator
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Extract data
        with st.spinner("Extracting data from PDF..."):
            extractor = PDFExtractor(tmp_path)
            output_dir = "data/extracted"
            text_path, tables_path = extractor.save_extracted(output_dir)
            
            # Validate
            validator = DataValidator()
            text_valid = validator.validate_text(text_path)
            tables_valid = validator.validate_tables(tables_path)
            
            if text_valid and tables_valid:
                st.success("✅ PDF extracted and validated successfully!")
                
                # Store paths in session state for other pages
                st.session_state['text_path'] = text_path
                st.session_state['tables_path'] = tables_path
            else:
                st.error("⚠️ Data validation failed. Check the extracted files.")
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.write("Traceback:", e)
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.sidebar.title("Navigation")

pages = {
    "Dashboard": "pages/dashboard.py",
    "DTC Analytics": "pages/dtc_analytics.py",
    "IP Analytics": "pages/ip_analytics.py"
}

selected_page = st.sidebar.radio("Go to", list(pages.keys()))

if selected_page == "Dashboard":
    from music_analytics_app.streamlit_ui.pages.dashboard import show_dashboard
    show_dashboard()
elif selected_page == "DTC Analytics":
    from music_analytics_app.streamlit_ui.pages.dtc_analytics import show_dtc_analytics
    show_dtc_analytics()
elif selected_page == "IP Analytics":
    from music_analytics_app.streamlit_ui.pages.ip_analytics import show_ip_analytics
    show_ip_analytics()

