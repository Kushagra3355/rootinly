import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Settings:
    """Production configuration settings for Rootinly AI unified service."""
    # Application Metadata
    APP_NAME: str = "Rootinly AI - Hair & Scalp Analysis Platform"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Unified platform for Crown View Hair Growth Comparison and Hairfall Stage Classification."
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))

    # Directory Paths
    BASE_DIR: Path = BASE_DIR
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.pt")))
    )
    FALLBACK_MODEL_PATH: Path = BASE_DIR / "best.pt"
    STATIC_DIR: Path = field(
        default_factory=lambda: Path(os.getenv("STATIC_DIR", str(BASE_DIR / "static")))
    )
    LOGS_DIR: Path = field(
        default_factory=lambda: Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))
    )
    STAGE_LOGS_DIR: Path = field(
        default_factory=lambda: Path(os.getenv("STAGE_LOGS_DIR", str(BASE_DIR / "stage_determiner" / "logs")))
    )

    # CORS
    CORS_ORIGINS: List[str] = field(
        default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
    )

    # YOLO Segmentation Parameters
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))

    # YOLO Stage Classification Parameters (Norwood Scale)
    STAGE_MODEL_PATH: Path = field(
        default_factory=lambda: Path(os.getenv("STAGE_MODEL_PATH", str(BASE_DIR / "models" / "best_norwood.pt")))
    )
    FALLBACK_STAGE_MODEL_PATH: Path = BASE_DIR / "best_norwood.pt"

    # HSV Scalp/Skin Detection Thresholds
    LOWER_SKIN_HSV: Tuple[int, int, int] = (0, 10, 60)
    UPPER_SKIN_HSV: Tuple[int, int, int] = (30, 255, 255)

    # Visualization Overlays (BGR Colors for OpenCV)
    OVERLAY_HAIR_COLOR: Tuple[int, int, int] = (0, 220, 100)      # Green (Hair)
    OVERLAY_SCALP_COLOR: Tuple[int, int, int] = (0, 90, 255)      # Orange (Scalp)
    OVERLAY_CONTOUR_COLOR: Tuple[int, int, int] = (255, 255, 0)   # Cyan (Contour)
    OVERLAY_ALPHA: float = 0.45
    OVERLAY_BETA: float = 0.55

    def get_active_model_path(self) -> Path:
        """Returns the active segmentation model weight path."""
        return self.MODEL_PATH

    def get_stage_model_path(self) -> Path:
        """Returns the stage classification model weight path."""
        return self.STAGE_MODEL_PATH

    def get_index_html_path(self) -> Path:
        """Returns the index.html path from static directory."""
        return self.STATIC_DIR / "index.html"

settings = Settings()

