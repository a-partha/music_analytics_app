import pandas as pd
import logging

class ExportPowerAnalyzer:
    """
    Analyzes import/export streaming shares, power rankings, country splits.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def export_power_ranking(self, export_df: pd.DataFrame, top_n=10) -> pd.DataFrame:
        """
        Computes export power ranking by country/region.
        Args:
            export_df: DataFrame with ['country', 'export_power', 'top_artist']
            top_n: number of ranks to return
        Returns:
            DataFrame of top export countries/artists
        """
        try:
            ranking = export_df.sort_values('export_power', ascending=False).head(top_n)
            self.logger.info("Export power ranking calculated.")
            return ranking
        except Exception as e:
            self.logger.error(f"Export power ranking error: {e}")
            return pd.DataFrame()

    def country_streaming_share(self, share_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes streaming share breakdown by country.
        Args:
            share_df: DataFrame with ['country', 'streams']
        Returns:
            DataFrame with percentage shares per country
        """
        try:
            total = share_df['streams'].sum()
            share_df['percentage'] = (share_df['streams'] / total) * 100
            self.logger.info("Country streaming shares computed.")
            return share_df.sort_values('percentage', ascending=False)
        except Exception as e:
            self.logger.error(f"Country share error: {e}")
            return pd.DataFrame()
