import pandas as pd
import logging
import matplotlib.pyplot as plt
import os


class FanEngagementAnalyzer:
    """
    Analyzes fan engagement funnel and segmentation metrics.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = "data/output"
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_funnel(self, funnel_df: pd.DataFrame) -> dict:
        """
        Calculates funnel conversion rates and engagement distributions.
        Args:
            funnel_df: DataFrame with funnel stages and percentages.
        Returns:
            dict of conversion metrics.
        """
        result = {}
        try:
            stages = funnel_df.columns.tolist()
            percentages = funnel_df.iloc[0].tolist()
            conversion = {}
            for i in range(1, len(percentages)):
                prev = percentages[i-1]
                curr = percentages[i]
                if prev:
                    conversion_rate = round((curr / prev)*100, 2)
                    conversion[f"{stages[i-1]}_to_{stages[i]}"] = conversion_rate
            result["conversion_rates"] = conversion
            result["stage_distribution"] = dict(zip(stages, percentages))
            
            # Generate chart
            self._generate_funnel_chart(stages, percentages)
            
            self.logger.info("Fan funnel analysis complete")
        except Exception as e:
            self.logger.error(f"Funnel analysis error: {e}")
        return result

    def _generate_funnel_chart(self, stages, percentages):
        """Generate and save funnel chart."""
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(stages, percentages, color='skyblue', edgecolor='navy')
            plt.xlabel('Funnel Stage')
            plt.ylabel('Percentage (%)')
            plt.title('Fan Engagement Funnel')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_path = os.path.join(self.output_dir, "fan_funnel_chart.png")
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Funnel chart saved to {chart_path}")
        except Exception as e:
            self.logger.error(f"Chart generation error: {e}")

    def segment_superfans(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Segments superfan users/fan groups based on threshold metric.
        Args:
            df: fan data DataFrame
            threshold: fraction/percent requirement
        Returns:
            DataFrame of superfan segments
        """
        try:
            segment = df[df['engagement_score'] >= threshold].copy()
            self.logger.info(f"Identified {len(segment)} superfans")
            return segment
        except Exception as e:
            self.logger.error(f"Superfan segmentation error: {e}")
            return pd.DataFrame()
