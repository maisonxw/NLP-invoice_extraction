import re
from typing import Dict, Any, List
from app.core.ocr import get_ocr_lines

def extract_fields(image_path: str) -> Dict[str, Any]:
    """Extract fields using hand-crafted regex rules (SROIE baseline)."""
    lines, full_text = get_ocr_lines(image_path)
    
    if not lines:
        return {
            "method": "Rule-based",
            "company": "Not Found",
            "address": "Not Found",
            "date": "Not Found",
            "total": "Not Found",
            "error": "OCR failed to detect text."
        }

    # COMPANY: typically the first text line
    company = lines[0] if lines else "Not Found"

    # ADDRESS: scan early lines for address keywords
    address_kws = [
        "jalan", "street", "road", "avenue", "no.", "lot", "block",
        "floor", "level", "km", "taman", "kompleks", "plaza", "off",
        "lorong", "bandar", "mall", "centre", "center",
    ]
    address = "Not Found"
    for line in lines[1:12]:
        if any(kw in line.lower() for kw in address_kws):
            address = line
            break

    # DATE: standard pattern match
    date_match = re.search(
        r"\b(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"
        r"|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b",
        full_text,
    )

    # TOTAL: keyword followed by an amount
    total_match = re.search(
        r"(?:TOTAL|GRAND\s*TOTAL|AMOUNT\s*DUE|NET\s*TOTAL|JUMLAH\s*BESAR|JUMLAH)"
        r"[^\d\n]*([\d,]+\.?\d*)",
        full_text,
        re.IGNORECASE,
    )

    # Fallback: if keyword-based match fails, take the last numeric token
    total_val = "Not Found"
    if total_match:
        total_val = total_match.group(1)
    else:
        amounts = re.findall(r"[\d,]+\.?\d*", full_text)
        if amounts:
            total_val = amounts[-1]

    return {
        "method": "Rule-based",
        "company": company,
        "address": address,
        "date": date_match.group(1) if date_match else "Not Found",
        "total": total_val,
        "_debug_ocr_lines": lines[:50],
    }
