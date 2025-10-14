import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

class IPVisualizer:
    """
    Visualizes IP/catalog analytics: growth bars, export country maps, top rankings, AI/transmedia overlays.
    """

    def __init__(self, output_dir="data/output"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_growth_by_genre(self, growth_df: pd.DataFrame):
        """
        Plots catalog growth by genre/age.
        Args:
            growth_df: DataFrame with ['genre', 'catalog_age', 'streams']
        Returns:
            file path to chart
        """
        plt.figure(figsize=(9, 5))
        sns.barplot(x="catalog_age", y="streams", hue="genre", data=growth_df)
        plt.title("Catalog Growth by Genre & Age")
        plt.ylabel("Streams")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "catalog_growth_genre.png")
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Saved catalog growth chart: {fig_path}")
        return fig_path

    def plot_export_power_ranking(self, ranking_df: pd.DataFrame):
        """
        Plots export power ranking of top countries/artists.
        Args:
            ranking_df: DataFrame ['country', 'export_power', 'top_artist']
        Returns:
            file path to chart
        """
        plt.figure(figsize=(10, 5))
        sns.barplot(x="country", y="export_power", data=ranking_df)
        plt.xticks(rotation=45)
        plt.title("Export Power Rankings")
        plt.ylabel("Score")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "export_power_chart.png")
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Saved export power chart: {fig_path}")
        return fig_path
