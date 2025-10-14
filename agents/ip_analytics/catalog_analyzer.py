import pandas as pd
import logging

class CatalogAnalyzer:
    """
    Analyzes catalog growth, age, genre breakdowns, and top artists/works.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def growth_by_genre(self, catalog_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes growth rates for genre, subgenre, and catalog age.
        Args:
            catalog_df: DataFrame with ['genre', 'catalog_age', 'streams']
        Returns:
            DataFrame, growth by genre/subgenre/age
        """
        try:
            summary = catalog_df.groupby(['genre', 'catalog_age'])['streams'].sum().reset_index()
            self.logger.info("Catalog growth by genre/age computed.")
            return summary
        except Exception as e:
            self.logger.error(f"Growth analysis error: {e}")
            return pd.DataFrame()

    def luminate_index_leaderboard(self, index_df: pd.DataFrame, top_n=10) -> pd.DataFrame:
        """
        Ranks top artists/albums by Luminate Index metric.
        Args:
            index_df: DataFrame with ['artist', 'album', 'luminate_index']
            top_n: number of top entries to return
        Returns:
            DataFrame with leaderboard
        """
        try:
            leaderboard = index_df.sort_values('luminate_index', ascending=False).head(top_n)
            self.logger.info("Luminate Index leaderboard calculated.")
            return leaderboard
        except Exception as e:
            self.logger.error(f"Luminate Index error: {e}")
            return pd.DataFrame()
