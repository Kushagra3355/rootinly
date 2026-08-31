"""API routes for collecting and storing user validation feedback."""
import json
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from src.rootinly.core.firebase_service import firebase_service
from src.rootinly.logger import logger

router = APIRouter(tags=["Feedback"])


@router.post(
    "/api/v1/feedback/crown",
    summary="Submit Crown Hair Growth Comparison Feedback",
)
@router.post(
    "/feedback-crown",
    summary="Submit Crown Hair Growth Comparison Feedback (Alias)",
    include_in_schema=False,
)
async def submit_crown_feedback(
    is_prev_masking_correct: bool = Form(..., description="Is baseline image masking correct (Yes/No)"),
    prev_masking_pct: float = Form(..., ge=0, le=100, description="Percentage correctness of baseline masking (0-100)"),
    is_curr_masking_correct: bool = Form(..., description="Is follow-up image masking correct (Yes/No)"),
    curr_masking_pct: float = Form(..., ge=0, le=100, description="Percentage correctness of follow-up masking (0-100)"),
    is_prev_classification_correct: bool = Form(..., description="Is baseline hair/scalp classification correct (Yes/No)"),
    prev_classification_pct: float = Form(..., ge=0, le=100, description="Percentage correctness of baseline hair/scalp classification (0-100)"),
    is_curr_classification_correct: bool = Form(..., description="Is follow-up hair/scalp classification correct (Yes/No)"),
    curr_classification_pct: float = Form(..., ge=0, le=100, description="Percentage correctness of follow-up hair/scalp classification (0-100)"),
    is_result_valid: bool = Form(..., description="Is overall comparison result valid (Yes/No)"),
    overall_validity_pct: float = Form(..., ge=0, le=100, description="Percentage correctness of overall result (0-100)"),
    metrics: str = Form(default="{}", description="JSON string containing visit metrics and percentage change"),
    notes: str = Form(default="", description="Optional feedback notes or clinical comments"),
    last_visit_image: UploadFile = File(None, description="Original baseline crown image"),
    today_visit_image: UploadFile = File(None, description="Original follow-up crown image"),
):
    """
    Receives both Yes/No validation decisions and percentage accuracy scores (0-100),
    then saves the records to Firebase Storage and Cloud Firestore.
    """
    try:
        baseline_bytes = await last_visit_image.read() if last_visit_image else None
        today_bytes = await today_visit_image.read() if today_visit_image else None
        
        parsed_metrics = {}
        if metrics:
            try:
                parsed_metrics = json.loads(metrics)
            except Exception as pe:
                logger.warning(f"Failed to parse feedback metrics JSON: {pe}")

        feedback_data = {
            "is_prev_masking_correct": is_prev_masking_correct,
            "prev_masking_pct": prev_masking_pct,
            "is_curr_masking_correct": is_curr_masking_correct,
            "curr_masking_pct": curr_masking_pct,
            "is_prev_classification_correct": is_prev_classification_correct,
            "prev_classification_pct": prev_classification_pct,
            "is_curr_classification_correct": is_curr_classification_correct,
            "curr_classification_pct": curr_classification_pct,
            "is_result_valid": is_result_valid,
            "overall_validity_pct": overall_validity_pct,
            "metrics": parsed_metrics,
            "notes": notes,
        }

        feedback_id = firebase_service.save_crown_feedback(
            feedback_data=feedback_data,
            baseline_bytes=baseline_bytes,
            today_bytes=today_bytes,
        )



        return {
            "status": "success",
            "message": "Feedback and images recorded successfully in Firebase.",
            "feedback_id": feedback_id,
        }

    except RuntimeError as re:
        logger.warning(f"Firebase feedback submission skipped: {re}")
        # Return 200 with notice or 503 so frontend can give user feedback
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"Error submitting crown feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")
