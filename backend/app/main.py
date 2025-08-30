import os
import asyncio
import logging  # <-- ADDED
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

# --- Logging Configuration ---  <-- ADDED
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if it exists
load_dotenv()

# Import the enhanced redaction functions
from .redaction import redact_text, Entity, RedactionResult, initialize_models, get_ner_status

app = FastAPI(
    title="Enhanced PII Guardian API",
    version="2.0.0",
    description="A robust PII detection and redaction service with advanced adjudication."
)

# --- Pydantic Models for API ---
class RedactRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The text to be redacted.")

class RedactResponse(BaseModel):
    redacted_text: str
    entities: List[Entity]
    map: Dict[str, str] = Field(..., description="A map from placeholders to their original values.")

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """
    Asynchronously initializes all models and configurations when the server starts.
    This ensures the API is ready to handle requests immediately without a cold-start delay.
    """
    logger.info("Server startup: Kicking off model initialization...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, initialize_models)
    logger.info("Model initialization complete. API is ready.")

@app.get("/health", summary="Check API Health")
async def health() -> Dict[str, str]:
    """Provides a simple health check endpoint."""
    return {"status": "ok"}

@app.post("/redact", response_model=RedactResponse, summary="Redact PII from Text")
async def redact(req: RedactRequest) -> RedactResponse:
    """
    Accepts text and returns a redacted version along with details of the entities found.
    """
    try:
        # Run the CPU-bound redaction logic in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        result: RedactionResult = await loop.run_in_executor(
            None, lambda: redact_text(req.text)
        )
        return RedactResponse(
            redacted_text=result.redacted_text,
            entities=result.entities,
            map=result.placeholder_to_original,
        )
    except Exception as e:
        logger.error(f"Redaction failed with error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during redaction.")

@app.get("/status", summary="Get NER Model Status")
async def ner_status() -> Dict[str, object]:
    """Returns the current status of loaded models and configurations."""
    return get_ner_status()