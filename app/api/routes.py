from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import tempfile
from pathlib import Path
from app.core.config import ALLOWED_EXTENSIONS
from app.services import rule_extractor, gemini_extractor, layoutlm_extractor

router = APIRouter()

def _save_upload(upload: UploadFile) -> Path:
    """Save the uploaded file to a temporary location."""
    suffix = Path(upload.filename).suffix.lower() if upload.filename else ".jpg"
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(upload.file, tmp)
    tmp.close()
    return Path(tmp.name)

@router.post("/predict/{method}")
async def predict_invoice(method: str, file: UploadFile = File(...)):
    """General endpoint for invoice extraction across different methods."""
    tmp_path = _save_upload(file)
    try:
        if method == "rule":
            result = rule_extractor.extract_fields(str(tmp_path))
        elif method == "llm":
            result = gemini_extractor.extract_fields(str(tmp_path))
        elif method == "layoutlm":
            result = layoutlm_extractor.extract_fields(str(tmp_path))
        else:
            raise HTTPException(status_code=400, detail=f"Invalid extraction method: {method}")
        
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)
