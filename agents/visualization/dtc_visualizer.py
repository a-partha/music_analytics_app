import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

class DTCVisualizer:
    """
    Visualizes DTC/Fan engagement analytics: funnel charts, segmentations, spend/merch/hours.
    """

    def __init__(self, output_dir="data/output"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_funnel(self, funnel_dict):
        """
        Plots funnel as a horizontal bar chart.
        Args:
            funnel_dict: {stage: value}
        Returns:
            file path to chart
        """
        stages = list(funnel_dict.keys())
        values = list(funnel_dict.values())
        plt.figure(figsize=(8, 4))
        sns.barplot(x=values, y=stages, palette="viridis")
        plt.title("Fan Engagement Funnel")
        plt.xlabel("Percentage")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "fan_funnel_chart.png")
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Saved funnel chart: {fig_path}")
        return fig_path

    def plot_spend_table(self, spend_df: pd.DataFrame):
        """
        Plots yearly spend/merch/travel/hours by fan profile.
        Args:
            spend_df: DataFrame with columns ['profile', 'music_hours', 'spend', 'merch', 'travel']
        Returns:
            file path to chart
        """
        plt.figure(figsize=(8, 5))
        spend_df.plot(kind="bar", x="profile", stacked=False)
        plt.title("Fan Profile Spending/Engagement")
        plt.ylabel("USD / Hours")
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "fan_spend_engagement.png")
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Saved spend/engagement chart: {fig_path}")
        return fig_path
