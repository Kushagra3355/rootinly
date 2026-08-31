"""YOLOv8 Segmentation module for crown/head contour extraction."""
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
from src.rootinly.config import settings
from src.rootinly.logger import logger

class CrownSegmentor:
    """
    Manages custom YOLOv8 instance segmentation model for precise hairline and crown extraction.
    """
    def __init__(self, model_path: Optional[Path] = None, confidence: Optional[float] = None):
        self.model_path = model_path or settings.get_active_model_path()
        self.confidence = confidence if confidence is not None else settings.YOLO_CONFIDENCE
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO model into memory."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model path '{self.model_path}' does not exist. Attempting fallback...")
                if settings.FALLBACK_MODEL_PATH.exists():
                    self.model_path = settings.FALLBACK_MODEL_PATH
                else:
                    raise FileNotFoundError(
                        f"Model weights not found at primary '{self.model_path}' or fallback '{settings.FALLBACK_MODEL_PATH}'"
                    )

            self.model = YOLO(str(self.model_path))
            logger.info(f"Successfully loaded YOLOv8 segmentation model from '{self.model_path}'.")
        except Exception as e:
            logger.error(f"Failed to load YOLO segmentation model: {e}")
            self.model = None

    @property
    def is_loaded(self) -> bool:
        """Checks if the YOLO model is successfully loaded."""
        return self.model is not None

    def extract_mask(self, image: np.ndarray, conf: Optional[float] = None) -> np.ndarray:
        """
        Extracts the head/crown binary mask using the YOLO segmentation model.
        
        Args:
            image: OpenCV BGR image (np.ndarray).
            conf: Optional confidence threshold override.
            
        Returns:
            np.ndarray: Binary mask of shape (H, W) where 1 = Crown ROI, 0 = Background.

        Raises:
            RuntimeError: If YOLO segmentation model is not loaded.
            ValueError: If no head is detected in the image.
        """
        h, w = image.shape[:2]

        if self.model is None:
            logger.error("YOLO segmentation model is not loaded.")
            raise RuntimeError("YOLO segmentation model is not loaded.")

        conf_thresh = conf if conf is not None else self.confidence
        results = self.model.predict(source=image, conf=conf_thresh, verbose=False)

        if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
            # Extract raw mask data tensor for the primary detected head segment
            mask_data = results[0].masks.data[0].cpu().numpy()
            
            # YOLO internally resizes images to 640x640; resize mask back to source image dimensions
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Convert probabilities to binary mask (1 for crown/hair, 0 for background)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            if int(np.sum(binary_mask)) == 0:
                logger.warning("No head mask detected by YOLO.")
                raise ValueError("No head detected")

            return binary_mask
        else:
            logger.warning("No head mask detected by YOLO.")
            raise ValueError("No head detected")
