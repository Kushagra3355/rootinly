"""Analysis and metric schemas."""
from pydantic import BaseModel, Field

class VisitMetrics(BaseModel):
    hair_density_percent: float = Field(..., description="Hair density percentage in valid ROI")
    scalp_visibility_percent: float = Field(..., description="Scalp visibility percentage in valid ROI")
    roi_total_pixels: int = Field(..., description="Total pixels in the segmented head ROI")
    hair_pixels: int = Field(..., description="Calculated hair pixels")
    scalp_pixels: int = Field(..., description="Calculated scalp/skin pixels")

class PercentageChange(BaseModel):
    hair_density_change_pct: float = Field(..., description="Percentage change in hair density")
    scalp_visibility_change_pct: float = Field(..., description="Percentage change in scalp visibility")

class Visualizations(BaseModel):
    previous_visit_overlay_b64: str = Field(..., description="Data URL Base64 PNG overlay for baseline visit")
    today_visit_overlay_b64: str = Field(..., description="Data URL Base64 PNG overlay for follow-up visit")

class LogEntry(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the log entry (HH:MM:SS)")
    message: str = Field(..., description="Log message text")
    level: str = Field("INFO", description="Log level (INFO, WARNING, ERROR, SUCCESS)")
