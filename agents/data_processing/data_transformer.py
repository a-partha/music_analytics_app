#agents/data_processing/data_transformer.py
import pandas as pd
import logging

class DataTransformer:
    """
    Transforms cleaned data into analytic-ready structures.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def tables_to_long_format(self, df):
        """
        Converts wide tables to long format usable by visualizations.
        Args:
            df: pandas DataFrame
        Returns:
            pandas DataFrame (long format)
        """
        try:
            long_df = df.melt()
            self.logger.info(f"Transformed to long format: {long_df.shape}")
            return long_df
        except Exception as e:
            self.logger.error(f"Error in wide-to-long transformation: {e}")
            return df

    def aggregate_metrics(self, df, groupby_column, agg_column, agg_func='sum'):
        """
        Produce aggregation for major report breakdowns.
        Args:
            df: pandas DataFrame
            groupby_column: grouping key
            agg_column: column to aggregate
            agg_func: aggregation function
        Returns:
            pandas DataFrame (aggregated)
        """
        try:
            agg = df.groupby(groupby_column)[agg_column].agg(agg_func).reset_index()
            self.logger.info(f"Aggregated metrics: {agg.shape}")
            return agg
        except Exception as e:
            self.logger.error(f"Error in aggregation: {e}")
            return pd.DataFrame()

    def text_to_sections(self, text, section_headers):
        """
        Splits cleaned text into report sections using headers.
        Args:
            text: Cleaned text string
            section_headers: List of strings (headers)
        Returns:
            Dict of section_name to content
        """
        sections = {}
        for i, header in enumerate(section_headers):
            start = text.find(header)
            if start == -1:
                continue
            end = text.find(section_headers[i+1], start) if i+1 < len(section_headers) else len(text)
            sections[header] = text[start:end].strip()
        self.logger.info(f"Extracted {len(sections)} sections from text.")
        return sections
