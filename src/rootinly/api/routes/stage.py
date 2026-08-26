"""Hairfall stage determination API routes."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from src.rootinly.api.dependencies import get_stage_determiner
from src.rootinly.core.stage_determiner import StageDeterminerService
from src.rootinly.schemas.response import ErrorResponse
from src.rootinly.schemas.stage import StageLogsListResponse, StageResponse

router = APIRouter(tags=["Hairfall Stage Classification"])


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
    Receives a single scalp photograph and predicts the clinical hairfall stage (Level 1-5).
    Returns the classified stage, confidence percentage, duration, clinical insights, and execution logs.
    """
    filename = file.filename or "uploaded_scalp.jpg"
    content_type = file.content_type or "image/jpeg"

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        response = service.predict(
            image_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
        )
        return response

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except TimeoutError as te:
        raise HTTPException(status_code=504, detail=str(te))
    except RuntimeError as re:
        # Check if it was an upstream error with status code
        err_str = str(re)
        if "HTTP 403" in err_str or "Forbidden" in err_str:
            raise HTTPException(status_code=403, detail=err_str)
        elif "HTTP 404" in err_str:
            raise HTTPException(status_code=404, detail=err_str)
        raise HTTPException(status_code=500, detail=err_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage prediction failed: {str(e)}")


@router.get(
    "/api/v1/stage/logs",
    response_model=StageLogsListResponse,
    summary="List Stage Determination Log Files",
)
@router.get(
    "/api/v1/logs",
    response_model=StageLogsListResponse,
    summary="List Stage Log Files (Legacy Alias)",
    include_in_schema=False,
)
async def list_stage_logs(
    service: StageDeterminerService = Depends(get_stage_determiner),
):
    """Lists all timestamped log files recorded during stage classification."""
    try:
        return service.list_logs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list log files: {str(e)}")


@router.get(
    "/api/v1/stage/health",
    summary="Stage Classification Health Check",
)
async def stage_health(
    service: StageDeterminerService = Depends(get_stage_determiner),
):
    """Checks configuration readiness and directory paths for stage determination."""
    return {
        "status": "healthy" if service.is_configured else "unconfigured",
        "service": "Roboflow Hairfall Stage Classifier",
        "configured": service.is_configured,
        "model_id": service.model_id,
        "api_url": service.api_url,
        "logs_directory": str(service.logs_dir.resolve()),
    }
