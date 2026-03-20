# Invoice Intelligence

A robust document processing system for structured information extraction from receipts and invoices. This platform leverages multiple extraction strategies including rule-based heuristics, LLM-powered context awareness, and fine-tuned spatial models (LayoutLM).

## Features

- **Multi-Strategy Extraction**: Choose between Rule-based, LLM (Gemini), or Fine-tuned LayoutLM.
- **Unified OCR Layer**: Modular support for PaddleOCR, EasyOCR, and PyTesseract with automatic fallback.
- **Modern Web Interface**: Clean, premium UI for real-time document analysis.
- **RESTful API**: Clean FastAPI implementation for easy integration.

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   Create a `.env` file in the root directory (see `.env.example`):
   ```ini
   GEMINI_API_KEY=your_key_here
   ```

3. **Run the Application**
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure

- `app/`: Main application logic.
  - `api/`: FastAPI route definitions.
  - `core/`: OCR handlers and configuration.
  - `services/`: Specialized extraction engines.
  - `templates/`: Modern frontend components.
- `train/`: Research and model training notebooks.

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Jinja2
- **ML/AI**: Google Gemini (LLM), LayoutLM (HuggingFace Transformers), PyTorch
- **OCR**: PaddleOCR, EasyOCR, Tesseract

---
*Developed for efficient document digitization.*
