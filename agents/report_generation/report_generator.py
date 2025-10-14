import pandas as pd
import os
from fpdf import FPDF
import logging
from datetime import datetime

class ReportGenerator:
    """
    Generates business reports combining analytics, tables, and visualizations.
    """

    def __init__(self, output_dir="data/output"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pdf_report(self, summary: dict, dtc_tables: dict, ip_tables: dict, charts: list, filename=None):
        """
        Create a PDF with executive summary, tables, and chart images.
        Args:
            summary: Dict of textual findings/insights
            dtc_tables: Dict of DataFrames (DTC analytics)
            ip_tables: Dict of DataFrames (IP analytics)
            charts: List of image file paths (charts/plots)
            filename: Custom filename (optional)
        Returns:
            Path to generated PDF
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "Music Analytics Report", 0, 1, 'C')

        # Date
        pdf.cell(0, 10, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", 0, 1, 'C')
        pdf.ln(10)

        # Executive Summary
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, "Executive Summary", 0, 1)
        pdf.set_font("Arial", size=12)
        for k, v in summary.items():
            pdf.cell(0, 10, f"{k}:", 0, 1)
            pdf.multi_cell(0, 8, str(v))
            pdf.ln(2)

        # DTC Tables
        pdf.set_font("Arial", style="B", size=13)
        pdf.cell(0, 10, "DTC & Fan Engagement Analytics", 0, 1)
        pdf.set_font("Arial", size=10)
        for title, df in dtc_tables.items():
            pdf.cell(0, 8, title, 0, 1)
            table_str = df.to_string(index=False, justify='left')
            pdf.set_font("Arial", size=8)
            pdf.multi_cell(0, 5, table_str)
            pdf.ln(2)

        # IP Tables
        pdf.set_font("Arial", style="B", size=13)
        pdf.cell(0, 10, "IP Value & Catalog Analytics", 0, 1)
        pdf.set_font("Arial", size=10)
        for title, df in ip_tables.items():
            pdf.cell(0, 8, title, 0, 1)
            table_str = df.to_string(index=False, justify='left')
            pdf.set_font("Arial", size=8)
            pdf.multi_cell(0, 5, table_str)
            pdf.ln(2)

        # Charts
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, "Visualizations", 0, 1)
        for img_path in charts:
            if os.path.exists(img_path):
                pdf.image(img_path, w=170)
                pdf.ln(5)

        output_file = filename or f"music_analytics_report_{datetime.now():%Y%m%d_%H%M}.pdf"
        out_path = os.path.join(self.output_dir, output_file)
        pdf.output(out_path)
        self.logger.info(f"PDF report generated: {out_path}")
        return out_path

    def generate_summary_text(self, dtc_results: dict, ip_results: dict) -> dict:
        """
        Create summary sections based on analytics outputs.
        Args:
            dtc_results: dict
            ip_results: dict
        Returns:
            dict of summary paragraphs
        """
        summary = {}
        try:
            summary['Fan Engagement'] = f"Engagement conversion rates: {dtc_results.get('conversion_rates',{})}"
            summary['Superfan Insights'] = f"Superfan segment: {len(dtc_results.get('superfans',[]))} identified."
            summary['IP Value Highlights'] = f"Top ranked artists/albums: {ip_results.get('top_artists','N/A')}."
            summary['Export Power'] = f"Export power rankings: {ip_results.get('export_power','N/A')}."
            self.logger.info("Summary text generated.")
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
        return summary
