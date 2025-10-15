from music_analytics_app.agents.data_extraction.pdf_extractor import PDFExtractor

pdf_path = "data/input/Luminate 2025 Midyear Music Report.pdf"
output_dir = "data/extracted"

extractor = PDFExtractor(pdf_path)
text_path, tables_path = extractor.save_extracted(output_dir)
print(f"Extracted text: {text_path}")
print(f"Extracted tables: {tables_path}")
