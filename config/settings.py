import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class Settings:
    # Core paths
    PDF_INPUT_DIR = os.getenv("PDF_INPUT_DIR", "data/input")
    PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", "data/output")
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    AGENT_CONFIG_PATH = os.getenv("AGENT_CONFIG_PATH", "config/agent_config.yaml")
    PREFECT_CONFIG_PATH = os.getenv("PREFECT_CONFIG_PATH", "config/prefect_config.yaml")

    # Streamlit
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_DEBUG = os.getenv("STREAMLIT_DEBUG", "1") == "1"

    # Google ADK (example keys)
    GOOGLE_ADK_PROJECT_ID = os.getenv("GOOGLE_ADK_PROJECT_ID", "")
    GOOGLE_ADK_CLIENT_SECRET = os.getenv("GOOGLE_ADK_CLIENT_SECRET", "")

    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "replace-with-a-long-random-string")

    # App metadata
    APP_TITLE = "Music Analytics Dashboard"
    VERSION = "1.0.0"
