import sys
import os

# Add current directory to Python path so imports work
sys.path.insert(0, os.getcwd())

from agents.data_extraction.pdf_extractor import PDFExtractor

pdf_path = "data/input/Luminate 2025 Midyear Music Report.pdf"
output_dir = "data/extracted"

print("Starting extraction...")
extractor = PDFExtractor(pdf_path)
print("PDFExtractor created, calling save_extracted...")
text_path, tables_path = extractor.save_extracted(output_dir)
print(f"Extracted text: {text_path}")
print(f"Extracted tables: {tables_path}")
print("Done!")
