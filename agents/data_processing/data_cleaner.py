#agents/data_processing/data_cleaner.py
import pandas as pd
import logging

class DataCleaner:
    """
    Cleans and preps raw analytics data for downstream processing.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_tables(self, table_files):
        """
        Cleans extracted tables for analytics.
        Args:
            table_files: List of CSV file paths or pandas DataFrames
        Returns:
            List of cleaned DataFrames
        """
        cleaned_dfs = []
        for source in table_files:
            try:
                # Handle both file paths and DataFrames
                if isinstance(source, str):
                    df = pd.read_csv(source)
                else:
                    df = source.copy()

                # Drop entirely empty columns/rows
                df.dropna(axis=0, how='all', inplace=True)
                df.dropna(axis=1, how='all', inplace=True)
                # Strip whitespace from column headers and values
                df.columns = df.columns.str.strip()
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                # Replace empty strings with NaN for consistency
                df.replace('', pd.NA, inplace=True)

                cleaned_dfs.append(df)
                self.logger.info(f"Cleaned table loaded: {df.shape}")
            except Exception as e:
                self.logger.error(f"Error cleaning table: {e}")

        return cleaned_dfs

    def clean_text(self, text):
        """
        Cleans and normalizes extracted PDF text.
        Args:
            text: Raw string
        Returns:
            Cleaned string
        """
        import re
        cleaned = re.sub(r'\s+', ' ', text).strip()
        # Remove non-printable characters
        cleaned = ''.join(c for c in cleaned if c.isprintable())
        self.logger.info(f"Text cleaned; length: {len(cleaned)}")
        return cleaned
