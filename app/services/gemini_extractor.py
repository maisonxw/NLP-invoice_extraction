import json
import re
from typing import Dict, Any
import google.generativeai as genai
from app.core.config import GEMINI_API_KEY
from app.core.ocr import get_ocr_text

# Prompt for Gemini
_PROMPT = """\
Extract the following fields from the invoice/receipt text:
- company: store or restaurant name
- address: full street address
- date: YYYY-MM-DD (format it properly)
- total: final amount as digits only (e.g. 12.50)

Return ONLY a valid JSON object. Correct OCR spelling errors if possible.

Receipt text:
{text}
"""

def extract_fields(image_path: str) -> Dict[str, Any]:
    """Extract fields using Gemini LLM on top of OCR text."""
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not found in environment settings."}
    
    ocr_text = get_ocr_text(image_path)
    if not ocr_text:
        return {"error": "OCR failed to return any text."}

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    try:
        response = model.generate_content(_PROMPT.format(text=ocr_text))
        raw_text = getattr(response, 'text', str(response))
        parsed = _parse_json(raw_text)
        parsed["method"] = "LLM (Gemini)"
        parsed["_debug_ocr_text"] = ocr_text[:1000]
        return parsed
    except Exception as e:
        return {
            "method": "LLM (Gemini)",
            "error": f"Gemini API Error: {str(e)}",
            "_debug_ocr_text": ocr_text[:1000]
        }

def _parse_json(raw: str) -> Dict[str, Any]:
    """Robustly extract JSON from text."""
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip("` \n")
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "company": "Not Found", "address": "Not Found",
        "date": "Not Found", "total": "Not Found"
    }
