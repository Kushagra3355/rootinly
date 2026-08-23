"""
Rootinly AI - Simple Hairfall Stage Determiner Microservice.

Loads API Key and URL from .env file, queries Roboflow model, and returns
the stage, confidence percentage, and execution logs maintained within
the stage_determiner/logs directory.
"""

import argparse
import base64
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from stage_determiner.logger import (
        logger,
        StageExecutionLogger,
        LOGS_DIR,
        APP_LOG_FILE,
    )
except ImportError:
    from logger import (  # type: ignore
        logger,
        StageExecutionLogger,
        LOGS_DIR,
        APP_LOG_FILE,
    )

import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Configuration from .env
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL")
INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"


class StageResponse(BaseModel):
    stage: str
    confidence: float
    duration_ms: Optional[float] = None
    log_file: Optional[str] = None
    logs: Optional[List[Dict[str, str]]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle events and logs startup/shutdown."""
    logger.info("=======================================================")
    logger.info("Initializing Hairfall Stage Determiner Microservice...")
    logger.info(f"Isolated Logs Directory: {LOGS_DIR.resolve()}")
    logger.info(f"App Log File: {APP_LOG_FILE.resolve()}")
    if ROBOFLOW_API_URL and ROBOFLOW_MODEL_ID:
        logger.info(f"Roboflow Model Endpoint: {ROBOFLOW_API_URL}/{ROBOFLOW_MODEL_ID}")
    else:
        logger.warning("Roboflow environment variables not fully configured.")
    logger.info("=======================================================")
    yield
    logger.info("Shutting down Hairfall Stage Determiner Microservice...")


app = FastAPI(
    title="Hairfall Stage Determiner",
    description="Microservice for hairfall stage classification with isolated logging in stage_determiner/logs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the minimal stage determiner web page."""
    logger.info("GET / - Serving stage determiner web interface.")
    if INDEX_HTML_PATH.exists():
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    logger.error(f"index.html not found at {INDEX_HTML_PATH}")
    return HTMLResponse("<h2>index.html not found</h2>", status_code=404)


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    logger.info("GET /api/v1/health - Service health verified.")
    return {"status": "healthy", "logs_directory": str(LOGS_DIR.resolve())}


@app.get("/api/v1/logs")
async def list_logs():
    """Lists the log files maintained inside stage_determiner/logs."""
    try:
        log_files = sorted(
            [f.name for f in LOGS_DIR.glob("*.log")],
            reverse=True,
        )
        return {
            "status": "success",
            "logs_directory": str(LOGS_DIR.resolve()),
            "total_log_files": len(log_files),
            "log_files": log_files,
        }
    except Exception as e:
        logger.error(f"Failed to list log files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-stage", response_model=StageResponse)
@app.post("/predict", response_model=StageResponse)
async def predict_stage(file: UploadFile = File(...)):
    """Receives image and returns stage, confidence percentage, and execution logs."""
    exec_logger = StageExecutionLogger()
    filename = file.filename or "unknown_image.jpg"
    content_type = file.content_type or "unknown"
    exec_logger.log(f"Received stage prediction request for file: '{filename}' ({content_type})")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            exec_logger.log("Uploaded file is empty (0 bytes).", level="ERROR")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        exec_logger.log(f"Image read successfully ({len(image_bytes)} bytes). Encoding to Base64...")
        img_b64 = base64.b64encode(image_bytes).decode("ascii")

        url = f"{ROBOFLOW_API_URL}/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}"
        exec_logger.log(f"Dispatching inference request to Roboflow model '{ROBOFLOW_MODEL_ID}'...")

        res = requests.post(
            url,
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )

        if res.status_code == 200:
            data = res.json()
            stage = data.get("top", "Level 1")
            raw_conf = data.get("confidence", 0.0)
            conf_pct = round(float(raw_conf) * 100, 2)
            duration_ms = exec_logger.get_total_duration_ms()

            exec_logger.log(
                f"Roboflow inference successful: Stage='{stage}', Confidence={conf_pct}% (Duration: {duration_ms:.1f}ms)",
                level="SUCCESS",
            )

            return StageResponse(
                stage=str(stage),
                confidence=conf_pct,
                duration_ms=duration_ms,
                log_file=exec_logger.get_log_filepath(),
                logs=exec_logger.get_logs(),
            )
        else:
            err_msg = f"Roboflow API error (HTTP {res.status_code}): {res.text}"
            exec_logger.log(err_msg, level="ERROR")
            raise HTTPException(
                status_code=res.status_code,
                detail=err_msg,
            )

    except HTTPException:
        raise
    except Exception as e:
        exec_logger.log(f"Unhandled error during prediction: {str(e)}", level="ERROR")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Hairfall Stage Determiner Microservice")
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    logger.info(f"Starting Stage Determiner server at http://localhost:{args.port}")
    uvicorn.run("stage_determiner.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
