import pymupdf
import pytesseract
import cv2
import numpy as np
import re
import json
from PIL import Image

PDF_PATH = "src/data/LenderFeesWorksheetNew.pdf"
KEY_FIELDS = ["MORTGAGE", "NOTE", "LENDER", "PROPERTY ADDRESS", "DATE", "SIGNATURE"]

OCR_CORRECTIONS = [
    (r'\bL0AN\b',     'LOAN'),
    (r'\bM0RTGAGE\b', 'MORTGAGE'),
    (r'\b1NTEREST\b', 'INTEREST'),
]


class TesseractExtractor:
    """OCR pipeline: PDF page → image → preprocess → Tesseract → structured output."""

    def __init__(self, pdf_path: str = PDF_PATH, ocr_config: str = r'--oem 3 -l eng'):
        self.doc = pymupdf.open(pdf_path)
        self.ocr_config = ocr_config

    def render_page(self, page_num: int = 0) -> np.ndarray:
        """Render a PDF page to a grayscale numpy array."""
        pix = self.doc[page_num].get_pixmap()
        img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        """Sharpen and upscale the image to improve OCR accuracy."""
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        h, w = gray.shape
        return cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    def extract_text(self, gray: np.ndarray) -> str:
        """Run OCR and clean the output — collapse whitespace, fix common misreads."""
        raw = pytesseract.image_to_string(gray, config=self.ocr_config)
        text = " ".join(raw.split())
        for pattern, replacement in OCR_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r'[^a-zA-Z0-9\s,.%-]', '', text)
    
    def extract_layout_aware_text(self):
        """Comapare EasyOCR and PaddleOCR"""
        reader = easyocr.Reader(['ch_sim','en'])
        result = reader.readtext('image.jpg')

    def extract_fields(self, text: str) -> dict:
        """Pull structured key-value fields from cleaned OCR text using regex."""
        fields = {}
        loan_match = re.search(r"Loan Amount[:\s$]*([\d,]+)", text, re.IGNORECASE)
        if loan_match:
            fields["loan_amount"] = loan_match.group(1)
        return fields

    def get_word_bboxes(self, gray: np.ndarray, confidence_threshold: int = 40) -> list[dict]:
        """Return all words Tesseract detected, filtered by confidence score."""
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        results = []
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            if word and conf >= confidence_threshold:
                results.append({
                    "text": word,
                    "bbox": [data["left"][i], data["top"][i],
                             data["width"][i], data["height"][i]],
                    "confidence": conf
                })
        return results

    def draw_bboxes(self, gray: np.ndarray, word_data: list[dict],
                    highlight: list[str] | None = None) -> np.ndarray:
        """Draw bounding boxes on the image. Highlighted words appear in red."""
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        highlight_set = {f.upper() for f in (highlight or [])}
        for entry in word_data:
            x, y, w, h = entry["bbox"]
            color = (0, 0, 255) if entry["text"].upper() in highlight_set else (0, 255, 0)
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_bgr, entry["text"], (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def run(self, page_num: int = 0) -> dict:
        """Full pipeline: render → preprocess → OCR → extract fields + bboxes."""
        gray = self.render_page(page_num)
        gray = self.preprocess(gray)
        text = self.extract_text(gray)
        fields = self.extract_fields(text)
        words = self.get_word_bboxes(gray)
        return {"text": text, "fields": fields, "words": words}

    def close(self):
        self.doc.close()


if __name__ == "__main__":
    extractor = TesseractExtractor()
    result = extractor.run()
    print("Extracted Fields:", json.dumps(result["fields"], indent=2))
    print(f"Total words detected: {len(result['words'])}")

    gray = extractor.preprocess(extractor.render_page())
    annotated = extractor.draw_bboxes(gray, result["words"], highlight=KEY_FIELDS)
    Image.fromarray(annotated).show(title="Tesseract")

    extractor.close()
