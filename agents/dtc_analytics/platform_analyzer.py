import pandas as pd
import logging
import matplotlib.pyplot as plt
import os


class PlatformAnalyzer:
    """
    Analyzes engagement across different platforms/channels.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = "data/output"
        os.makedirs(self.output_dir, exist_ok=True)

    def channel_overlay(self, df: pd.DataFrame, platforms: list) -> dict:
        """
        Returns over-indexing and platform engagement by genre/fan group.
        Args:
            df: DataFrame with columns ['platform', 'genre', 'fan_group', 'engagement_score']
            platforms: list of platform names
        Returns:
            Dict keyed by platform, genre overlays.
        """
        overlays = {}
        try:
            for platform in platforms:
                pf_df = df[df['platform'] == platform]
                by_genre = pf_df.groupby('genre')['engagement_score'].mean().sort_values(ascending=False)
                overlays[platform] = by_genre.to_dict()
            self.logger.info("Platform overlays calculated.")
        except Exception as e:
            self.logger.error(f"Channel overlay error: {e}")
        return overlays

    def tipping_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes fan tipping behaviors across platforms.
        Args:
            df: DataFrame with ['fan_id', 'platform', 'tip_amount']
        Returns:
            Aggregated tipping by platform
        """
        try:
            tips = df.groupby('platform')['tip_amount'].agg(['sum', 'mean', 'count']).reset_index()
            
            # Generate chart
            self._generate_tipping_chart(tips)
            
            self.logger.info("Tipping analysis complete.")
            return tips
        except Exception as e:
            self.logger.error(f"Tipping analysis error: {e}")
            return pd.DataFrame()

    def _generate_tipping_chart(self, tips_df):
        """Generate and save tipping analysis chart."""
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(tips_df['platform'], tips_df['sum'], color='lightgreen', edgecolor='darkgreen')
            plt.xlabel('Platform')
            plt.ylabel('Total Tips ($)')
            plt.title('Total Tipping by Platform')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_path = os.path.join(self.output_dir, "platform_tipping_chart.png")
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Tipping chart saved to {chart_path}")
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")