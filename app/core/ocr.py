import re
from PIL import Image
from typing import List, Tuple, Optional
import traceback

# Optional: PaddleOCR
try:
    from paddleocr import PaddleOCR
    _paddle_available = True
except ImportError:
    PaddleOCR = None
    _paddle_available = False

# Optional: EasyOCR
try:
    import easyocr
    _easyocr_available = True
except ImportError:
    easyocr = None
    _easyocr_available = False

# Optional: PyTesseract
try:
    import pytesseract
    _pytesseract_available = True
except ImportError:
    pytesseract = None
    _pytesseract_available = False

_paddle_instance = None
_easyocr_instance = None

def get_ocr_text(image_path: str) -> str:
    """Run OCR and return full text string using available backends with fallback."""
    lines, full_text = get_ocr_lines(image_path)
    return full_text

def get_ocr_lines(image_path: str) -> Tuple[List[str], str]:
    """Run OCR and return list of lines and full text string."""
    global _paddle_instance, _easyocr_instance

    # 1. Try EasyOCR (usually faster/easier to install)
    if _easyocr_available:
        try:
            if _easyocr_instance is None:
                _easyocr_instance = easyocr.Reader(['en'])
            res = _easyocr_instance.readtext(image_path, detail=0, paragraph=True)
            lines = [l.strip() for l in res if l.strip()]
            return lines, "\n".join(lines)
        except Exception:
            print(f"[OCR] EasyOCR Error: {traceback.format_exc()}")

    # 2. Try PyTesseract
    if _pytesseract_available:
        try:
            img = Image.open(image_path)
            txt = pytesseract.image_to_string(img)
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            return lines, "\n".join(lines)
        except Exception:
            print(f"[OCR] PyTesseract Error: {traceback.format_exc()}")

    # 3. Try PaddleOCR (SROIE specialized)
    if _paddle_available:
        try:
            if _paddle_instance is None:
                _paddle_instance = PaddleOCR(use_angle_cls=True, lang="en")
            result = _paddle_instance.ocr(image_path)
            if result and result[0] is not None:
                lines = [item[1][0].strip() for item in result[0] if item[1][0].strip()]
                return lines, "\n".join(lines)
        except Exception:
            print(f"[OCR] PaddleOCR Error: {traceback.format_exc()}")

    return [], ""

def get_ocr_bboxes(image_path: str) -> Tuple[List[str], List[List[int]], int, int]:
    """Get words paired with their bounding boxes (used for LayoutLM)."""
    if not _easyocr_available:
        return [], [], 0, 0
    
    try:
        global _easyocr_instance
        if _easyocr_instance is None:
            _easyocr_instance = easyocr.Reader(['en'])
        
        result = _easyocr_instance.readtext(image_path)
        img = Image.open(image_path)
        width, height = img.size
        img.close()

        words = []
        bboxes = []
        for detection in result:
            bbox_points = detection[0]
            text = detection[1].strip()
            if not text: continue

            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x0, y0, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            line_words = [w for w in text.split() if w]
            bbox_width = max(x2 - x0, 1)
            cur_x = x0
            total_chars = sum(len(w) for w in line_words)
            if total_chars == 0: continue

            for w in line_words:
                word_x2 = cur_x + int(bbox_width * len(w) / total_chars)
                words.append(w)
                bboxes.append([cur_x, y0, word_x2, y2])
                cur_x = word_x2 + 2 # Minor spacing

        return words, bboxes, width, height
    except Exception:
        print(f"[OCR] BBox Detection Error: {traceback.format_exc()}")
        return [], [], 0, 0
