import pymupdf
import re
import json

PDF_PATH = "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# This PDF is a two-column form. PyMuPDF's get_text("text") linearizes it so
# all labels come first, then all values — they're never adjacent in the stream.
# Strategy: regex on value formats (e.g. "4.250 %") to find them, then
# get_text("words") to pin their position on the page.
# ─────────────────────────────────────────────────────────────────────────────


class LoanWorksheetExtractor:

    HEADER_PATTERNS = {
        "Applicant Name":    r"FEES WORKSHEET\n(.+)",
        "Date Prepared":     r"(\d{2}/\d{2}/\d{4})",
        "Loan Program":      r"(\d+ YEAR [A-Z\s\-]+Purchase)",
        "Total Loan Amount": r"\$\s+([\d,]+)\n",
        "Interest Rate":     r"(\d+\.\d+\s*%)",
        "Loan Term":         r"(\d+ / \d+ mths)",
    }

    FEE_SECTIONS = ["ORIGINATION CHARGES", "OTHER CHARGES"]

    def __init__(self, pdf_path: str = PDF_PATH):
        self.doc = pymupdf.open(pdf_path)
        self.page = self.doc[0]
        self._text = self.page.get_text("text")
        self._words = self.page.get_text("words")

    def _find_bbox(self, token: str) -> list | None:
        """Return bbox of the first word matching token."""
        for word in self._words:
            x0, y0, x1, y1, text, *_ = word
            if token.lower() in text.lower():
                return [round(x0), round(y0), round(x1), round(y1)]
        return None

    def extract_header_fields(self) -> list[dict]:
        """Extract key scalar fields from the form header via regex."""
        value_block = self._text[self._text.find("FEES WORKSHEET"):]
        results = []
        for label, pattern in self.HEADER_PATTERNS.items():
            match = re.search(pattern, value_block)
            if not match:
                results.append({"field": label, "text": "NOT FOUND", "bbox": None})
                continue
            value = match.group(1).strip()
            token = value.split()[0].replace(",", "").replace("$", "")
            results.append({"field": label, "text": value, "bbox": self._find_bbox(token)})
        return results

    def extract_fee_rows(self) -> list[dict]:
        """
        Extract fee line items from Origination and Other Charges sections.
        Fee rows follow: Fee Name → Paid To → Borrower → $ → Amount.
        The 'Borrower' anchor makes the pattern reliable across all rows.
        """
        results = []
        pattern = r"([A-Z][^\n$]+)\n[^\n]+\nBorrower\n\$\n([\d,.]+)"

        for section in self.FEE_SECTIONS:
            start = self._text.find(section)
            if start == -1:
                continue
            end = len(self._text)
            for other in self.FEE_SECTIONS:
                pos = self._text.find(other, start + 1)
                if pos != -1 and pos < end:
                    end = pos

            for match in re.finditer(pattern, self._text[start:end]):
                fee_name = match.group(1).strip()
                if fee_name in self.FEE_SECTIONS:
                    continue
                amount = match.group(2).strip()
                results.append({
                    "text": f"{fee_name} - ${amount}",
                    "bbox": self._find_bbox(fee_name.split()[0]),
                    "section": section
                })
        return results

    def run(self) -> list[dict]:
        """Run full extraction and return combined JSON-ready output."""
        combined = []
        for f in self.extract_header_fields():
            combined.append({"text": f"{f['field']}: {f['text']}", "bbox": f["bbox"]})
        for f in self.extract_fee_rows():
            combined.append({"text": f["text"], "bbox": f["bbox"]})
        return combined

    def close(self):
        self.doc.close()


if __name__ == "__main__":
    extractor = LoanWorksheetExtractor()
    results = extractor.run()
    print(json.dumps(results, indent=2))
    extractor.close()
