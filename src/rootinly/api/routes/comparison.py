"""Crown hair comparison and frontend routes."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from src.rootinly.api.dependencies import get_pipeline
from src.rootinly.config import settings
from src.rootinly.core.pipeline import CrownComparisonPipeline
from src.rootinly.schemas.response import ComparisonResponse, ErrorResponse

router = APIRouter(tags=["Crown Hair Comparison"])

@router.get("/", summary="Serve Web Interface")
async def serve_frontend():
    """Serves the single-page web UI for image comparison."""
    index_file = settings.get_index_html_path()
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "index.html frontend not found."},
    )

@router.post(
    "/compare-crowns",
    response_model=ComparisonResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Compare Baseline and Follow-up Crown Photos",
)
async def compare_crown_photos(
    last_visit_image: UploadFile = File(..., description="Baseline visit crown photograph"),
    today_visit_image: UploadFile = File(..., description="Follow-up visit crown photograph"),
    pipeline: CrownComparisonPipeline = Depends(get_pipeline),
):
    """
    Analyzes baseline and follow-up crown photos to compute:
    - Hair density percentage
    - Scalp visibility percentage
    - Percentage deltas
    - Visual color-coded overlay masks
    - Timestamped execution logs
    """
    try:
        bytes_last = await last_visit_image.read()
        bytes_today = await today_visit_image.read()

        response = pipeline.process(
            last_bytes=bytes_last,
            today_bytes=bytes_today,
            last_filename=last_visit_image.filename or "baseline.jpg",
            today_filename=today_visit_image.filename or "follow_up.jpg",
        )
        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison processing failed: {str(e)}")
