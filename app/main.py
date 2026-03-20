import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from app.api.routes import router

app = FastAPI(
    title="Invoice Intelligence API",
    description="Automated Extraction of Receipt & Invoice data using Rules, LLMs, and Spatial Models.",
    version="2.0.0"
)

# Setup templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Include our refactored API routes
app.include_router(router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main frontend application."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    # Start the server locally
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
