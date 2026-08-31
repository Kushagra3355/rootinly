"""Hairfall stage determination API routes."""
import asyncio
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from src.rootinly.api.dependencies import get_stage_determiner
from src.rootinly.core.stage_determiner import StageDeterminerService
from src.rootinly.schemas.response import ErrorResponse
from src.rootinly.schemas.stage import StageResponse

router = APIRouter(tags=["Hairfall Stage Determiner"])


@router.post(
    "/predict-stage",
    response_model=StageResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Classify Hairfall Stage from Scalp Photo",
)
@router.post(
    "/predict",
    response_model=StageResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    include_in_schema=False,
)
@router.post(
    "/api/v1/stage/predict",
    response_model=StageResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    include_in_schema=False,
)
async def predict_hairfall_stage(
    file: UploadFile = File(..., description="Scalp image file (JPG, PNG) for stage determination"),
    service: StageDeterminerService = Depends(get_stage_determiner),
):
    """
    Receives a single scalp photograph and predicts the clinical hairfall stage (Norwood Scale Stage 1-7).
    Returns the classified stage, confidence percentage, duration, and execution logs.
    """
    filename = file.filename or "uploaded_scalp.jpg"
    content_type = file.content_type or "image/jpeg"

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Non-blocking async thread execution
        response = await asyncio.to_thread(
            service.predict,
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
        )
        return response

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage prediction failed: {str(e)}")


@router.get(
    "/api/v1/stage/health",
    summary="Hairfall Stage Determiner Health Check",
)
async def stage_health(
    service: StageDeterminerService = Depends(get_stage_determiner),
):
    """Checks model readiness and directory paths for stage determination."""
    return {
        "status": "healthy" if service.is_loaded else "unconfigured",
        "service": "YOLOv8 Norwood Hairfall Stage Classifier",
        "configured": service.is_loaded,
        "model_loaded": service.is_loaded,
        "model_path": str(service.model_path),
        "logs_directory": str(service.logs_dir.resolve()),
    }
