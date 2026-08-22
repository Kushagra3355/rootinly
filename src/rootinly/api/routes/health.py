"""Health check and system inspection routes."""
from datetime import datetime
from fastapi import APIRouter, Depends
from src.rootinly.api.dependencies import get_segmentor
from src.rootinly.config import settings
from src.rootinly.core.segmentor import CrownSegmentor
from src.rootinly.schemas.response import HealthResponse

router = APIRouter(tags=["Health & System"])

@router.get("/health", response_model=HealthResponse, summary="System Health Check")
async def health_check(segmentor: CrownSegmentor = Depends(get_segmentor)):
    """
    Returns service health status, version, and model readiness.
    """
    return HealthResponse(
        status="healthy" if segmentor.is_loaded else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=segmentor.is_loaded,
        model_path=str(segmentor.model_path),
        timestamp=datetime.now().isoformat(),
    )
