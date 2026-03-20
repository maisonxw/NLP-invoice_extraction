import torch
from transformers import AutoTokenizer, LayoutLMForTokenClassification
from typing import Dict, Any, List, Tuple
from app.core.config import LAYOUTLM_MODEL_PATH
from app.core.ocr import get_ocr_bboxes

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_available = False

def _load_model_state():
    """Initial model loading."""
    global _tokenizer, _model, _model_available
    if _model_available: return
    
    if LAYOUTLM_MODEL_PATH.exists():
        try:
            _tokenizer = AutoTokenizer.from_pretrained(str(LAYOUTLM_MODEL_PATH))
            _model = LayoutLMForTokenClassification.from_pretrained(str(LAYOUTLM_MODEL_PATH))
            _model.eval()
            _model.to(_device)
            _model_available = True
        except Exception as e:
            print(f"[LayoutLM] Load error: {e}")
            _model_available = False

def extract_fields(image_path: str) -> Dict[str, Any]:
    """Extract fields using fine-tuned LayoutLM model."""
    _load_model_state()
    if not _model_available:
        return {
            "method": "LayoutLM (Fine-tuned)",
            "error": f"LayoutLM model not found at {LAYOUTLM_MODEL_PATH}"
        }

    words, bboxes, width, height = get_ocr_bboxes(image_path)
    if not words:
        return {"error": "OCR failed (LayoutLM requirement)."}

    norm_bboxes = [_normalize_bbox(b, width, height) for b in bboxes]
    pred_labels, token_to_word = _run_inference(words, norm_bboxes)
    
    # Map back to words
    word_labels = ["O"] * len(words)
    for pred_label, word_idx in zip(pred_labels, token_to_word):
        if 0 <= word_idx < len(words) and word_labels[word_idx] == "O":
            word_labels[word_idx] = pred_label

    field_results = {"S-COMPANY": [], "S-ADDRESS": [], "S-DATE": [], "S-TOTAL": []}
    for word, label in zip(words, word_labels):
        if label in field_results:
            field_results[label].append(word)

    return {
        "method": "LayoutLM (Fine-tuned)",
        "company": " ".join(field_results["S-COMPANY"]) or "Not Found",
        "address": " ".join(field_results["S-ADDRESS"]) or "Not Found",
        "date": " ".join(field_results["S-DATE"]) or "Not Found",
        "total": " ".join(field_results["S-TOTAL"]) or "Not Found"
    }

def _normalize_bbox(bbox, width, height):
    """Normalize bbox to 0-1000 range for LayoutLM."""
    if width == 0 or height == 0: return [0, 0, 1000, 1000]
    return [
        max(0, min(int(1000 * bbox[0] / width), 1000)),
        max(0, min(int(1000 * bbox[1] / height), 1000)),
        max(0, min(int(1000 * bbox[2] / width), 1000)),
        max(0, min(int(1000 * bbox[3] / height), 1000))
    ]

def _run_inference(words: List[str], bboxes: List[List[int]], max_seq_len=512) -> Tuple[List[str], List[int]]:
    """Tokenize and run the LayoutLM model."""
    token_ids, token_bboxes, token_to_word_idx = [], [], []
    for word_idx, (word, bbox) in enumerate(zip(words, bboxes)):
        sub_tokens = _tokenizer.tokenize(word)
        sub_ids = _tokenizer.convert_tokens_to_ids(sub_tokens)
        token_ids.extend(sub_ids)
        token_bboxes.extend([bbox] * len(sub_ids))
        token_to_word_idx.extend([word_idx] * len(sub_ids))

    # Add special tokens
    max_tokens = max_seq_len - 2
    token_ids = [_tokenizer.cls_token_id] + token_ids[:max_tokens] + [_tokenizer.sep_token_id]
    token_bboxes = [[0, 0, 0, 0]] + token_bboxes[:max_tokens] + [[1000, 1000, 1000, 1000]]
    token_to_word_idx = [-1] + token_to_word_idx[:max_tokens] + [-1]

    # Padding
    pad_len = max_seq_len - len(token_ids)
    attention_mask = [1] * len(token_ids) + [0] * pad_len
    token_ids += [_tokenizer.pad_token_id] * pad_len
    token_bboxes += [[0, 0, 0, 0]] * pad_len
    
    input_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(_device)
    bbox_tensor = torch.tensor([token_bboxes], dtype=torch.long).to(_device)
    mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(_device)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids_tensor, bbox=bbox_tensor, attention_mask=mask_tensor)
        preds = torch.argmax(outputs.logits, dim=-1)[0]

    id2label = _model.config.id2label
    pred_labels = [id2label.get(int(p), "O") for p in preds[:len(token_to_word_idx)]]
    return pred_labels, token_to_word_idx
