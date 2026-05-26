import os
import sys

if __name__ == "__main__":
    sys.exit(
        os.system(f'"{sys.executable}" -m streamlit run app/streamlit_app.py')
    )
