"""API response schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field
from src.rootinly.schemas.analysis import (
    VisitMetrics,
    PercentageChange,
    Visualizations,
    LogEntry,
)

class ComparisonResponse(BaseModel):
    status: str = Field("success", description="Status of the comparison request")
    pipeline: str = Field("yolov8_segmentation", description="Pipeline identifier")
    response_time_ms: float = Field(..., description="Total execution duration in milliseconds")
    response_time_formatted: str = Field(..., description="Human-readable response time string")
    log_file: str = Field(..., description="Relative path to execution log file")
    timestamp: str = Field(..., description="ISO 8601 timestamp of analysis")
    previous_visit: VisitMetrics = Field(..., description="Baseline visit metrics")
    today_visit: VisitMetrics = Field(..., description="Follow-up visit metrics")
    percentage_change: PercentageChange = Field(..., description="Relative percentage change deltas")
    visualizations: Visualizations = Field(..., description="Base64 encoded overlay visualizations")
    logs: List[LogEntry] = Field(default_factory=list, description="Detailed execution log entries")

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Health status of the service")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Indicates if YOLO model is loaded into memory")
    model_path: str = Field(..., description="Path to active model weights file")
    timestamp: str = Field(..., description="Current ISO 8601 timestamp")

class ErrorResponse(BaseModel):
    status: str = Field("error", description="Error status indicator")
    message: str = Field(..., description="High-level error message")
    detail: Optional[str] = Field(None, description="Detailed error description")
