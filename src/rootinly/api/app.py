"""FastAPI application factory and middleware setup for unified Rootinly AI platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.rootinly.api.dependencies import get_segmentor, get_stage_determiner
from src.rootinly.api.routes.comparison import router as comparison_router
from src.rootinly.api.routes.stage import router as stage_router
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

    # 2. Verify Roboflow configuration for Module 2 (Hairfall Stage Determiner)
    stage_service = get_stage_determiner()
    if stage_service.is_configured:
        logger.info(
            f"[Module 2: Stage Determiner] Roboflow Classifier configured: "
            f"model='{stage_service.model_id}' at '{stage_service.api_url}'"
        )
    else:
        logger.warning("[Module 2: Stage Determiner] Roboflow environment variables incomplete.")

    logger.info("Serving 2 Modules: (1) Crown Comparison, (2) Hairfall Stage Determiner")
    logger.info("=======================================================")
        
    yield
    
    logger.info("Shutting down Rootinly API server...")

def create_app() -> FastAPI:
    """Creates and configures a production-ready FastAPI application serving both modules."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
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
    app.include_router(health_router, prefix="/api/v1")

    return app

app = create_app()

