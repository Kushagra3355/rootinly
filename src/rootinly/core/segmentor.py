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

    def extract_masks_batch(
        self, images: list, conf: Optional[float] = None
    ) -> list:
        """
        Extracts binary head/crown masks for multiple images in a single batched YOLO forward pass.
        
        Args:
            images: List of OpenCV BGR images (np.ndarray).
            conf: Optional confidence threshold override.
            
        Returns:
            list: List of binary masks (np.ndarray) of shape (H, W) where 1 = Crown ROI, 0 = Background.

        Raises:
            RuntimeError: If YOLO segmentation model is not loaded.
            ValueError: If no head is detected in one of the images.
        """
        if not images:
            return []

        if self.model is None:
            logger.error("YOLO segmentation model is not loaded.")
            raise RuntimeError("YOLO segmentation model is not loaded.")

        conf_thresh = conf if conf is not None else self.confidence
        # Batched forward pass on all images simultaneously
        results = self.model.predict(source=images, conf=conf_thresh, verbose=False)

        binary_masks = []
        for i, (img, res) in enumerate(zip(images, results)):
            h, w = img.shape[:2]
            if res.masks is not None and len(res.masks.data) > 0:
                mask_data = res.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8)

                if int(np.sum(binary_mask)) == 0:
                    logger.warning(f"No head mask detected by YOLO for image index {i}.")
                    raise ValueError("No head detected")

                binary_masks.append(binary_mask)
            else:
                logger.warning(f"No head mask detected by YOLO for image index {i}.")
                raise ValueError("No head detected")

        return binary_masks

    def extract_mask(self, image: np.ndarray, conf: Optional[float] = None) -> np.ndarray:
        """
        Extracts the head/crown binary mask for a single image using the YOLO segmentation model.
        
        Args:
            image: OpenCV BGR image (np.ndarray).
            conf: Optional confidence threshold override.
            
        Returns:
            np.ndarray: Binary mask of shape (H, W) where 1 = Crown ROI, 0 = Background.

        Raises:
            RuntimeError: If YOLO segmentation model is not loaded.
            ValueError: If no head is detected in the image.
        """
        return self.extract_masks_batch([image], conf=conf)[0]
