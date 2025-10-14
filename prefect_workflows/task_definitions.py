from prefect import task
from agents.data_extraction.pdf_extractor import PDFExtractor
from agents.data_extraction.data_validator import DataValidator
from agents.data_processing.data_cleaner import DataCleaner
from agents.data_processing.data_transformer import DataTransformer
from agents.dtc_analytics.fan_engagement_analyzer import FanEngagementAnalyzer
from agents.dtc_analytics.platform_analyzer import PlatformAnalyzer
from agents.ip_analytics.catalog_analyzer import CatalogAnalyzer
from agents.ip_analytics.export_power_analyzer import ExportPowerAnalyzer
from agents.report_generation.report_generator import ReportGenerator

@task
def extract_pdf(pdf_path):
    extractor = PDFExtractor(pdf_path)
    text = extractor.extract_text()
    tables = extractor.extract_tables()
    return text, tables

@task
def validate_data(text_path, tables_path):
    validator = DataValidator()
    valid_text = validator.validate_text(text_path)
    valid_tables = validator.validate_tables(tables_path)
    return valid_text and valid_tables

@task
def clean_and_transform_tables(table_files):
    cleaner = DataCleaner()
    transformer = DataTransformer()
    cleaned = cleaner.clean_tables(table_files)
    transformed = [transformer.tables_to_long_format(df) for df in cleaned]
    return transformed

@task
def fan_engagement_analysis(funnel_df):
    analyzer = FanEngagementAnalyzer()
    return analyzer.analyze_funnel(funnel_df)

@task
def platform_channel_overlay(df, platforms):
    analyzer = PlatformAnalyzer()
    return analyzer.channel_overlay(df, platforms)

@task
def catalog_growth_analysis(df):
    analyzer = CatalogAnalyzer()
    return analyzer.growth_by_genre(df)

@task
def export_power_ranking(df):
    analyzer = ExportPowerAnalyzer()
    return analyzer.export_power_ranking(df)

@task
def generate_report(summary, dtc_tables, ip_tables, charts):
    generator = ReportGenerator()
    return generator.generate_pdf_report(summary, dtc_tables, ip_tables, charts)
