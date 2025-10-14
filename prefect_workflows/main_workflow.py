from prefect import flow
from .task_definitions import (
    extract_pdf, validate_data, clean_and_transform_tables, 
    fan_engagement_analysis, platform_channel_overlay,
    catalog_growth_analysis, export_power_ranking,
    generate_report
)

@flow(name="Main Music Analytics Workflow")
def run_main_workflow(pdf_path, funnel_df, platform_df, catalog_df, export_df, charts):
    text, tables = extract_pdf(pdf_path)
    valid = validate_data(text, tables)
    cleaned_tables = clean_and_transform_tables(tables)
    dtc_result = fan_engagement_analysis(funnel_df)
    platform_overlay = platform_channel_overlay(platform_df, platforms=["Discord", "Reddit", "Twitch", "WhatsApp"])
    catalog_growth = catalog_growth_analysis(catalog_df)
    export_power = export_power_ranking(export_df)
    summary = {
        "dtc_result": dtc_result,
        "platform_overlay": platform_overlay,
        "catalog_growth": catalog_growth,
        "export_power": export_power,
    }
    report_path = generate_report(summary, {"Funnel Analysis": funnel_df}, {"Catalog Growth": catalog_df}, charts)
    return report_path
