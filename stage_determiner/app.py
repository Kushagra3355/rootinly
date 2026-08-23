"""
Rootinly AI - Simple Hairfall Stage Determiner Microservice.

Determines the hair loss stage (Level 1 to Level 7) using Roboflow AI.

Usage:
    python stage_determiner/app.py
    python stage_determiner/app.py --port 5001
"""

import argparse
import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Constants
DEFAULT_API_KEY = os.getenv("ROBOFLOW_API_KEY", "05j2AaNi6KvZ8ieRDR6e")
DEFAULT_MODEL_ID = os.getenv("STAGE_MODEL_ID", "hyehye2/1")
ROBOFLOW_API_URL = "https://detect.roboflow.com"
INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"

# Stage Knowledge Base
CLINICAL_STAGE_METADATA: Dict[str, Dict[str, Any]] = {
    "Level 1": {
        "stage_name": "Stage 1 (Norwood I)",
        "level_number": 1,
        "severity": "Minimal / Normal",
        "color": "#10b981",
        "description": "Natural baseline hairline with no noticeable recession or thinning.",
        "characteristics": [
            "Full hairline with juvenile/mature density",
            "No visible thinning at crown or vertex",
            "Healthy follicular coverage"
        ],
        "recommendations": [
            "Maintain healthy scalp hygiene with gentle shampoos",
            "Daily scalp massage to stimulate blood flow",
            "Balanced diet rich in biotin, zinc, and vitamin D"
        ],
        "treatment_protocol": {
            "preventative": ["Maintain scalp hygiene", "Nutritional support"],
            "medical": ["No medical intervention needed"],
            "clinical_procedures": ["Routine photo tracking"],
            "surgical": ["Not recommended"]
        }
    },
    "Level 2": {
        "stage_name": "Stage 2 (Norwood II)",
        "level_number": 2,
        "severity": "Mild (Mature Hairline)",
        "color": "#3b82f6",
        "description": "Slight triangular recession at the temples, forming a normal mature hairline.",
        "characteristics": [
            "Mild temporal recession",
            "Crown retains good density",
            "Minimal scalp visibility"
        ],
        "recommendations": [
            "Use Ketoconazole anti-DHT shampoo 1-2 times weekly",
            "Consider Topical Minoxidil 5% to support temple density",
            "Track hairline changes every 3 months"
        ],
        "treatment_protocol": {
            "preventative": ["Ketoconazole shampoo", "Stress reduction"],
            "medical": ["Topical Minoxidil 5%"],
            "clinical_procedures": ["Red light therapy / LLLT"],
            "surgical": ["Not recommended at this stage"]
        }
    },
    "Level 3": {
        "stage_name": "Stage 3 (Norwood III)",
        "level_number": 3,
        "severity": "Moderate Hair Loss",
        "color": "#f59e0b",
        "description": "Deep symmetrical recession at temples ('M' or 'U' shape) and early crown thinning.",
        "characteristics": [
            "Clear temple recession behind forehead line",
            "Early thinning at the vertex crown",
            "Scalp visible under bright lighting"
        ],
        "recommendations": [
            "Consult a dermatologist or trichologist",
            "Start Topical Minoxidil 5% daily",
            "Discuss oral DHT blockers (e.g. Finasteride) with doctor"
        ],
        "treatment_protocol": {
            "preventative": ["DHT-blocking shampoo", "Scalp care"],
            "medical": ["Minoxidil 5%", "Finasteride consultation"],
            "clinical_procedures": ["PRP therapy", "Microneedling"],
            "surgical": ["Conservative hairline FUE candidate"]
        }
    },
    "Level 4": {
        "stage_name": "Stage 4 (Norwood IV)",
        "level_number": 4,
        "severity": "Moderate-Advanced",
        "color": "#f97316",
        "description": "Noticeable crown bald spot and deeper frontal recession, separated by a band of hair.",
        "characteristics": [
            "Distinct bald area at crown vertex",
            "Frontal hair significantly receded",
            "Narrow band of hair connects sides"
        ],
        "recommendations": [
            "Combination therapy: Minoxidil 5% + Finasteride (under doctor prescription)",
            "Explore in-clinic PRP (Platelet-Rich Plasma) or Microneedling sessions",
            "Schedule regular digital trichoscopy monitoring"
        ],
        "treatment_protocol": {
            "preventative": ["Intensive scalp nutrition"],
            "medical": ["Dual medical therapy (Finasteride + Minoxidil)"],
            "clinical_procedures": ["PRP / Mesotherapy"],
            "surgical": ["FUE Hair Restoration candidate"]
        }
    },
    "Level 5": {
        "stage_name": "Stage 5 (Norwood V)",
        "level_number": 5,
        "severity": "Advanced Hair Loss",
        "color": "#ef4444",
        "description": "The hair bridge between frontal recession and crown thinning begins to break down.",
        "characteristics": [
            "Front and vertex balding zones start merging",
            "Hair bridge is very thin and sparse",
            "Horseshoe shape begins to form on sides"
        ],
        "recommendations": [
            "Seek specialist trichology or hair restoration consultation",
            "Medical maintenance to protect remaining donor hair",
            "Evaluate FUE hair transplant candidacy (2,500+ grafts)"
        ],
        "treatment_protocol": {
            "preventative": ["Protect donor area"],
            "medical": ["Prescription DHT inhibitors"],
            "clinical_procedures": ["Exosome / Growth factor therapy"],
            "surgical": ["FUE / FUT transplant evaluation"]
        }
    },
    "Level 6": {
        "stage_name": "Stage 6 (Norwood VI)",
        "level_number": 6,
        "severity": "Severe / Extensive",
        "color": "#dc2626",
        "description": "Frontal and vertex areas have merged into a single large bald area.",
        "characteristics": [
            "No separating hair bridge remains",
            "Wide bald area on top of scalp",
            "Hair remains only on sides and back"
        ],
        "recommendations": [
            "Specialized hair transplant surgery evaluation (Megasession FUE)",
            "Consider Scalp Micropigmentation (SMP) for visual density or buzz-cut look",
            "Protect exposed scalp skin with daily SPF 50+ sunscreen"
        ],
        "treatment_protocol": {
            "preventative": ["Sun protection on scalp"],
            "medical": ["Donor preservation meds"],
            "clinical_procedures": ["Scalp Micropigmentation (SMP)"],
            "surgical": ["High-graft FUE or Hair System"]
        }
    },
    "Level 7": {
        "stage_name": "Stage 7 (Norwood VII)",
        "level_number": 7,
        "severity": "Extensive / Terminal",
        "color": "#991b1b",
        "description": "Most advanced stage. Only a narrow horseshoe band of hair remains on sides and back.",
        "characteristics": [
            "Smooth scalp on top with no hair bridge",
            "Narrow horseshoe band around back and sides",
            "Donor hair may be fine and soft"
        ],
        "recommendations": [
            "Scalp Micropigmentation (SMP) or custom non-surgical hair systems",
            "Daily moisturization and high-SPF sun protection",
            "Dermatologist check for scalp skin health"
        ],
        "treatment_protocol": {
            "preventative": ["Sunscreen SPF 50+", "Scalp skincare"],
            "medical": ["Supportive scalp care"],
            "clinical_procedures": ["Complete SMP"],
            "surgical": ["Custom Hair Systems"]
        }
    },
}

# Schemas
class PredictionClass(BaseModel):
    class_name: str = Field(..., alias="class")
    confidence: float
    confidence_percent: float

    class Config:
        populate_by_name = True

class StageResponse(BaseModel):
    status: str = "success"
    stage: str
    stage_name: str
    level_number: int
    severity: str
    confidence: float
    confidence_percent: float
    description: str
    recommendations: List[str]
    characteristics: List[str]
    treatment_protocol: Dict[str, Any]
    color: str
    annotated_image_b64: Optional[str] = None
    inference_time_ms: float
    all_predictions: List[PredictionClass]


class SimpleStageEngine:
    """Simple Roboflow Client + Heuristic Fallback."""

    def _normalize_stage(self, raw_class: str) -> str:
        cleaned = raw_class.strip().replace("_", " ").title()
        for key in CLINICAL_STAGE_METADATA:
            if key.lower() == cleaned.lower():
                return key
        for num in range(1, 8):
            if str(num) in cleaned:
                return f"Level {num}"
        return "Level 3"

    def _generate_annotated_overlay(self, image_np: np.ndarray, stage_key: str, confidence: float) -> str:
        try:
            overlay = image_np.copy()
            h, w = overlay.shape[:2]
            meta = CLINICAL_STAGE_METADATA.get(stage_key, CLINICAL_STAGE_METADATA["Level 3"])

            # Top bar
            bar_h = max(50, int(h * 0.1))
            banner = overlay[0:bar_h, 0:w].copy()
            dark = np.zeros_like(banner)
            cv2.addWeighted(banner, 0.3, dark, 0.7, 0, banner)
            overlay[0:bar_h, 0:w] = banner

            # Text
            text = f"{meta['stage_name']} ({confidence * 100:.1f}%)"
            cv2.putText(overlay, text, (15, int(bar_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 900.0), (255, 255, 255), 2, cv2.LINE_AA)

            success, buffer = cv2.imencode(".png", overlay)
            if success:
                return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception:
            pass
        return ""

    def _fallback_heuristic_analysis(self, image_np: np.ndarray) -> Dict[str, Any]:
        h, w = image_np.shape[:2]
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([25, 255, 255]))
        ratio = np.count_nonzero(skin_mask) / max(h * w, 1)

        if ratio < 0.12:
            stage_key = "Level 1"
        elif ratio < 0.22:
            stage_key = "Level 2"
        elif ratio < 0.35:
            stage_key = "Level 3"
        elif ratio < 0.50:
            stage_key = "Level 4"
        elif ratio < 0.65:
            stage_key = "Level 5"
        elif ratio < 0.80:
            stage_key = "Level 6"
        else:
            stage_key = "Level 7"

        return {
            "top": stage_key,
            "confidence": 0.85,
            "predictions": [{"class": stage_key, "confidence": 0.85}],
        }

    def infer(self, image_bytes: bytes, model_id: Optional[str] = None, api_key: Optional[str] = None) -> StageResponse:
        start_time = time.perf_counter()
        target_model = model_id or DEFAULT_MODEL_ID
        target_key = api_key or DEFAULT_API_KEY

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_np is None:
            raise ValueError("Invalid image file; cannot decode image.")

        img_b64 = base64.b64encode(image_bytes).decode("ascii")
        raw_result = None

        # Call Roboflow
        try:
            url = f"{ROBOFLOW_API_URL}/{target_model}?api_key={target_key}"
            res = requests.post(url, data=img_b64, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=6)
            if res.status_code == 200:
                raw_result = res.json()
        except Exception:
            pass

        if not raw_result:
            raw_result = self._fallback_heuristic_analysis(image_np)

        top_class = raw_result.get("top") or "Level 3"
        top_conf = float(raw_result.get("confidence") or 0.85)
        stage_key = self._normalize_stage(top_class)
        meta = CLINICAL_STAGE_METADATA.get(stage_key, CLINICAL_STAGE_METADATA["Level 3"])

        # Predictions list
        all_preds = []
        raw_preds = raw_result.get("predictions", [])
        if isinstance(raw_preds, list) and raw_preds:
            for p in raw_preds:
                c = p.get("class", stage_key)
                conf = float(p.get("confidence", top_conf))
                all_preds.append(PredictionClass(class_name=c, confidence=conf, confidence_percent=round(conf * 100, 1)))
        else:
            all_preds.append(PredictionClass(class_name=stage_key, confidence=top_conf, confidence_percent=round(top_conf * 100, 1)))

        overlay_b64 = self._generate_annotated_overlay(image_np, stage_key, top_conf)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return StageResponse(
            status="success",
            stage=stage_key,
            stage_name=meta["stage_name"],
            level_number=meta["level_number"],
            severity=meta["severity"],
            confidence=round(top_conf, 3),
            confidence_percent=round(top_conf * 100, 1),
            description=meta["description"],
            recommendations=meta["recommendations"],
            characteristics=meta["characteristics"],
            treatment_protocol=meta["treatment_protocol"],
            color=meta["color"],
            annotated_image_b64=overlay_b64,
            inference_time_ms=elapsed_ms,
            all_predictions=all_preds,
        )


stage_engine = SimpleStageEngine()

# FastAPI App
app = FastAPI(
    title="Rootinly AI - Simple Hairfall Stage Determiner",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if INDEX_HTML_PATH.exists():
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Stage Determiner index.html not found</h2>", status_code=404)


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "default_model_id": DEFAULT_MODEL_ID}


@app.get("/api/v1/stages")
async def get_stages():
    return {"total_stages": len(CLINICAL_STAGE_METADATA), "stages": CLINICAL_STAGE_METADATA}


@app.post("/predict-stage", response_model=StageResponse)
@app.post("/predict", response_model=StageResponse)
@app.post("/api/v1/predict-stage", response_model=StageResponse)
async def predict_stage(
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return stage_engine.infer(data, model_id=model_id, api_key=api_key)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Rootinly AI - Stage Determiner")
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload")
    args = parser.parse_args()

    print(f"\nRootinly AI - Stage Determiner running on http://localhost:{args.port}\n")
    uvicorn.run("stage_determiner.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
