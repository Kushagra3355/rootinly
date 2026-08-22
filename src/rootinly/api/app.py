"""FastAPI application factory and middleware setup."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.rootinly.api.dependencies import get_segmentor
from src.rootinly.api.routes.comparison import router as comparison_router
from src.rootinly.api.routes.health import router as health_router
from src.rootinly.config import settings
from src.rootinly.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle events (startup and shutdown)."""
    logger.info(f"Initializing {settings.APP_NAME} v{settings.APP_VERSION}...")
    
    # Pre-load YOLO segmentation model on startup to keep inference latency minimal
    segmentor = get_segmentor()
    if segmentor.is_loaded:
        logger.info(f"YOLOv8 Model loaded successfully into memory from: {segmentor.model_path}")
    else:
        logger.warning("YOLOv8 Model failed to load during startup; check weights location.")
        
    yield
    
    logger.info("Shutting down Rootinly API server...")

def create_app() -> FastAPI:
    """Creates and configures a production-ready FastAPI application."""
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

    # API Routes
    app.include_router(comparison_router)
    app.include_router(health_router, prefix="/api/v1")

    return app

app = create_app()
