#agents/data_extraction/data_validator.py
import os
import pandas as pd
import logging

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_text(self, text_file):
        """Basic validation of extracted text."""
        if not os.path.exists(text_file):
            self.logger.error(f"Text file not found: {text_file}")
            return False
        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content.strip()) == 0:
            self.logger.warning("Text file is empty.")
            return False
        self.logger.info("Text validation passed.")
        return True

    def validate_tables(self, tables_file):
        """Basic validation of extracted tables."""
        if not os.path.exists(tables_file):
            self.logger.error(f"Tables file not found: {tables_file}")
            return False
        try:
            df = pd.read_csv(tables_file)
            if df.empty:
                self.logger.warning("Tables file has no data.")
                return False
        except Exception as e:
            self.logger.error(f"Error reading tables CSV: {e}")
            return False
        self.logger.info("Tables validation passed.")
        return True