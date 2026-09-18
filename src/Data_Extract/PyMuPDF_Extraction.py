import pymupdf
import re
import json
import cv2
import numpy as np
from PIL import Image


class PDFExtractor:
    """General-purpose PyMuPDF extractor. Open once, call any method."""

    def __init__(self, pdf_path: str):
        self.doc = pymupdf.open(pdf_path)

    def get_metadata(self) -> dict:
        return self.doc.metadata

    def extract_text_with_bbox(self, page_num: int = 0) -> list[dict]:
        """
        Extract every text span with its bounding box using dict mode.
        Returns [{"text": ..., "bbox": [x0, y0, x1, y1]}, ...]
        """
        results = []
        for block in self.doc[page_num].get_text("dict").get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line.get("spans", []):
                    if span["text"].strip():
                        results.append({
                            "text": span["text"].strip(),
                            "bbox": [round(v) for v in span["bbox"]]
                        })
        return results

    def find_word_bbox(self, target: str, page_num: int = 0) -> list | None:
        """Return bbox of the first word matching target, or None."""
        for word in self.doc[page_num].get_text("words"):
            x0, y0, x1, y1, text, *_ = word
            if target.lower() in text.lower():
                return [round(x0), round(y0), round(x1), round(y1)]
        return None

    def extract_tables(self) -> list[dict]:
        """Extract all tables with cell text and bboxes across every page."""
        results = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            for i, table in enumerate(page.find_tables().tables):
                table_data, cell_bboxes = [], []
                for row_num in range(table.row_count):
                    row_data, row_bboxes = [], []
                    for col_num in range(table.col_count):
                        idx = row_num * table.col_count + col_num
                        if idx < len(table.cells) and table.cells[idx]:
                            bbox = table.cells[idx]
                            row_data.append(page.get_text("text", clip=bbox).strip())
                            row_bboxes.append(list(bbox))
                        else:
                            row_data.append("")
                            row_bboxes.append(None)
                    table_data.append(row_data)
                    cell_bboxes.append(row_bboxes)
                results.append({
                    "page": page_num,
                    "table_index": i,
                    "bbox": list(table.bbox),
                    "data": table_data,
                    "cell_bboxes": cell_bboxes
                })
        return results

    def find_phone_number(self, page_num: int = 0) -> dict | None:
        """Find a phone number on a page and return its text and bbox."""
        text = self.doc[page_num].get_text("text")
        match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
        if not match:
            return None
        phone = match.group()
        return {"text": phone, "bbox": self.find_word_bbox(phone.split()[0], page_num)}

    def render_page_image(self, page_num: int = 0, output_path: str = "page.png"):
        self.doc[page_num].get_pixmap().save(output_path)

    def close(self):
        self.doc.close()


if __name__ == "__main__":
    PDF = "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf"
    extractor = PDFExtractor(PDF)

    print("Metadata:", extractor.get_metadata())
    print("\nText + BBoxes (first 5):")
    for item in extractor.extract_text_with_bbox()[:5]:
        print(item)

    print("\nTables:")
    for table in extractor.extract_tables():
        for row in table["data"]:
            print(row)

    extractor.close()
