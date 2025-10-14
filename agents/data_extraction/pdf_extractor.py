#agents/data_extraction/pdf_extractor.py
import os
import pdfplumber
import logging

class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.logger = logging.getLogger(__name__)
        self.pages = []

    def extract_text(self):
        """Extract all text from the PDF."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.pages = [page.extract_text() for page in pdf.pages]
            self.logger.info(f"Extracted text from {len(self.pages)} pages.")
            return self.pages
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            raise

    def extract_tables(self):
        """Extract all tables from the PDF."""
        tables = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    tables_on_page = page.extract_tables()
                    tables.extend(tables_on_page)
            self.logger.info(f"Extracted {len(tables)} tables from PDF.")
            return tables
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            raise

    def save_extracted(self, output_dir):
        """Save extracted text and tables to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        text_pages = self.extract_text()
        tables = self.extract_tables()

        text_path = os.path.join(output_dir, "extracted_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            for page_num, text in enumerate(text_pages):
                f.write(f"--- PAGE {page_num+1} ---\n")
                f.write(text if text else "[NO TEXT]\n")
                f.write("\n\n")

        tables_path = os.path.join(output_dir, "extracted_tables.csv")
        import pandas as pd
        table_count = 0
        with open(tables_path, "w", encoding="utf-8") as tf:
            for table in tables:
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df.to_csv(tf, mode='a', index=False)
                    tf.write("\n\n")
                    table_count += 1
        self.logger.info(f"Saved text and {table_count} tables to {output_dir}.")
        return text_path, tables_path
