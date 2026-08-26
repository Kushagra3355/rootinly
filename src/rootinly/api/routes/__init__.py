"""API route handlers package."""
from src.rootinly.api.routes.comparison import router as comparison_router
from src.rootinly.api.routes.stage import router as stage_router
from src.rootinly.api.routes.health import router as health_router

__all__ = ["comparison_router", "stage_router", "health_router"]

