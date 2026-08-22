"""API route handlers package."""
from src.rootinly.api.routes.comparison import router as comparison_router
from src.rootinly.api.routes.health import router as health_router

__all__ = ["comparison_router", "health_router"]
