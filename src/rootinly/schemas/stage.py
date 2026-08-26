"""Hairfall stage classification schema definitions."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StageLogEntry(BaseModel):
    """Schema for individual stage execution log entries."""
    timestamp: str = Field(..., description="Timestamp of the log step (HH:MM:SS)")
    message: str = Field(..., description="Log message text")
    level: str = Field("INFO", description="Log severity level (INFO, WARNING, ERROR, SUCCESS)")


class StageResponse(BaseModel):
    """Unified response model for hairfall stage determination."""
    status: str = Field("success", description="Prediction status")
    stage: str = Field(..., description="Classified hairfall stage (e.g. Level 1, Level 2, Stage 2)")
    confidence: float = Field(..., description="Confidence percentage of the stage prediction (0-100)")
    duration_ms: Optional[float] = Field(None, description="Inference and processing duration in milliseconds")
    duration_formatted: Optional[str] = Field(None, description="Human-readable duration string")
    log_file: Optional[str] = Field(None, description="Relative path to execution log file")
    logs: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Step-by-step diagnostic log entries")
    severity: Optional[str] = Field(None, description="Clinical severity classification (e.g. Minimal, Mild, Moderate, Advanced)")
    description: Optional[str] = Field(None, description="Clinical summary and visual pattern description")
    recommendation: Optional[str] = Field(None, description="Clinical care and follow-up guidance")


class StageLogsListResponse(BaseModel):
    """Response model for listing stage determiner log files."""
    status: str = Field("success", description="Status indicator")
    logs_directory: str = Field(..., description="Absolute or relative logs directory path")
    total_log_files: int = Field(..., description="Total count of available log files")
    log_files: List[str] = Field(default_factory=list, description="List of log file names")
