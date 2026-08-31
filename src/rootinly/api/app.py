"""FastAPI application factory and middleware setup for unified Rootinly AI platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.rootinly.api.dependencies import get_segmentor, get_stage_determiner
from src.rootinly.api.routes.comparison import router as comparison_router
from src.rootinly.api.routes.stage import router as stage_router
from src.rootinly.api.routes.feedback import router as feedback_router
from src.rootinly.api.routes.health import router as health_router

from src.rootinly.config import settings
from src.rootinly.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle events (startup and shutdown)."""
    logger.info("=======================================================")
    logger.info(f"Initializing {settings.APP_NAME} v{settings.APP_VERSION}...")
    
    # 1. Pre-load YOLO segmentation model for Module 1 (Crown Comparison)
    segmentor = get_segmentor()
    if segmentor.is_loaded:
        logger.info(f"[Module 1: Crown Comparison] YOLOv8 loaded from: {segmentor.model_path}")
    else:
        logger.warning("[Module 1: Crown Comparison] YOLOv8 weights not found or failed to load.")

    # 2. Pre-load YOLO Norwood classification model for Module 2 (Hairfall Stage Determiner)
    stage_service = get_stage_determiner()
    if stage_service.is_loaded:
        logger.info(
            f"[Module 2: Stage Determiner] YOLOv8 Norwood Classifier loaded from: {stage_service.model_path}"
        )
    else:
        logger.warning("[Module 2: Stage Determiner] YOLOv8 Norwood Stage model weights not found or failed to load.")

    logger.info("Serving 2 Modules: (1) Crown Comparison, (2) Hairfall Stage Determiner")
    logger.info("=======================================================")
        
    yield
    
    logger.info("Shutting down Rootinly API server...")

TAGS_METADATA = [
    {
        "name": "Web",
        "description": "Single-page web application interface delivery.",
    },
    {
        "name": "Crown Hair Comparison",
        "description": "Module 1: Crown view hair density and scalp comparison engine with health checks.",
    },
    {
        "name": "Hairfall Stage Determiner",
        "description": "Module 2: Scalp photograph Norwood hairfall stage classification with health checks.",
    },
    {
        "name": "Health & System",
        "description": "Unified system health checks and model inspection.",
    },
]

def create_app() -> FastAPI:
    """Creates and configures a production-ready FastAPI application serving both modules."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static assets mounting
    if settings.STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

    # API Routes for both modules & health
    app.include_router(comparison_router)
    app.include_router(stage_router)
    app.include_router(feedback_router)
    app.include_router(health_router, prefix="/api/v1")

    return app


app = create_app()

