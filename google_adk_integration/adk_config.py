import os
from config.settings import Settings

class GoogleADKConfig:
    """
    Loads Google ADK configuration from environment and settings.
    """
    def __init__(self):
        self.project_id = Settings.GOOGLE_ADK_PROJECT_ID
        self.client_secret = Settings.GOOGLE_ADK_CLIENT_SECRET
        self.config_path = Settings.AGENT_CONFIG_PATH

    def get_credentials(self):
        creds = {
            "project_id": self.project_id,
            "client_secret": self.client_secret,
            "config_path": self.config_path
        }
        return creds
