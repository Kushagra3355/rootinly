"""Health check and system inspection routes."""
from datetime import datetime
from fastapi import APIRouter, Depends
from src.rootinly.api.dependencies import get_segmentor, get_stage_determiner
from src.rootinly.config import settings
from src.rootinly.core.segmentor import CrownSegmentor
from src.rootinly.core.stage_determiner import StageDeterminerService
from src.rootinly.schemas.response import HealthResponse

router = APIRouter(tags=["Health & System"])

@router.get("/health", response_model=HealthResponse, summary="System Health Check")
async def health_check(
    segmentor: CrownSegmentor = Depends(get_segmentor),
    stage_service: StageDeterminerService = Depends(get_stage_determiner),
):
    """
    Returns unified service health status, version, active modules, and AI readiness.
    """
    is_healthy = segmentor.is_loaded and stage_service.is_loaded
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=segmentor.is_loaded,
        model_path=str(segmentor.model_path),
        stage_model_loaded=stage_service.is_loaded,
        stage_model_path=str(stage_service.model_path),
        roboflow_configured=stage_service.is_loaded,
        active_modules=["crown_comparison", "stage_determiner"],
        timestamp=datetime.now().isoformat(),
    )

