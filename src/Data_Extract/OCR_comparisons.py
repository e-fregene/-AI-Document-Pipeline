import pymupdf
import cv2
import numpy as np
from PIL import Image
import easyocr
from paddleocr import PaddleOCR

PDF_PATH = "src/data/LenderFeesWorksheetNew.pdf"


def render_page(pdf_path: str, page_num: int = 0) -> np.ndarray:
    """Render a PDF page to an RGB numpy array shared by both engines."""
    doc = pymupdf.open(pdf_path)
    pix = doc[page_num].get_pixmap()
    doc.close()
    return np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))


def annotate_and_show(img_rgb: np.ndarray, results: list[dict], title: str):
    """Draw bounding boxes on a copy of the image and display it."""
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(img, r["text"][:20], (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show(title=title)



class EasyOCRExtractor:
    """EasyOCR engine wrapper. Reads a rendered image, returns words + bboxes."""

    def __init__(self, lang: list[str] = ["en"]):
        self.reader = easyocr.Reader(lang, verbose=False)

    def extract(self, img_rgb: np.ndarray, confidence_threshold: float = 0.4) -> list[dict]:
        """
        Returns [{"text": ..., "bbox": [x, y, w, h], "confidence": ...}]
        EasyOCR gives bounding box as 4 corner points — we convert to x,y,w,h.
        """
        raw = self.reader.readtext(img_rgb)
        results = []
        for (points, text, conf) in raw:
            if conf < confidence_threshold:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            results.append({"text": text, "bbox": [x, y, w, h], "confidence": round(conf, 3)})
        return results



class PaddleOCRExtractor:
    """PaddleOCR engine wrapper. Same interface as EasyOCRExtractor."""

    def __init__(self, lang: str = "en"):
        self.ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    def extract(self, img_rgb: np.ndarray, confidence_threshold: float = 0.4) -> list[dict]:
        """
        Returns [{"text": ..., "bbox": [x, y, w, h], "confidence": ...}]
        PaddleOCR gives box as 4 corner points — same conversion as EasyOCR.
        """
        raw = self.ocr.predict(img_rgb)
        results = []
        for page in (raw or []):
            texts  = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys  = page.get("rec_polys", [])
            for text, conf, points in zip(texts, scores, polys):
                if conf < confidence_threshold or not text.strip():
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - x), int(max(ys) - y)
                results.append({"text": text, "bbox": [x, y, w, h], "confidence": round(conf, 3)})
        return results



if __name__ == "__main__":
    img = render_page(PDF_PATH)

    print("Running EasyOCR...")
    easy = EasyOCRExtractor()
    easy_results = easy.extract(img)
    print(f"  EasyOCR  → {len(easy_results)} words detected")
    annotate_and_show(img, easy_results, title="EasyOCR")

    print("Running PaddleOCR...")
    paddle = PaddleOCRExtractor()
    paddle_results = paddle.extract(img)
    print(f"  PaddleOCR → {len(paddle_results)} words detected")
    annotate_and_show(img, paddle_results, title="PaddleOCR")
