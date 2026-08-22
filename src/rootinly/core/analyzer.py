"""Follicular density and scalp visibility analysis module."""
from typing import Optional, Tuple
import cv2
import numpy as np
from src.rootinly.config import settings
from src.rootinly.schemas.analysis import PercentageChange, VisitMetrics

class CrownAnalyzer:
    """
    Analyzes hair density vs scalp exposure using HSV color segmentation within an organic head mask,
    and renders color-coded overlays with contour boundaries.
    """
    def __init__(
        self,
        lower_skin_hsv: Optional[Tuple[int, int, int]] = None,
        upper_skin_hsv: Optional[Tuple[int, int, int]] = None,
        hair_color: Optional[Tuple[int, int, int]] = None,
        scalp_color: Optional[Tuple[int, int, int]] = None,
        contour_color: Optional[Tuple[int, int, int]] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        self.lower_skin = np.array(lower_skin_hsv or settings.LOWER_SKIN_HSV, dtype=np.uint8)
        self.upper_skin = np.array(upper_skin_hsv or settings.UPPER_SKIN_HSV, dtype=np.uint8)
        self.hair_color = hair_color or settings.OVERLAY_HAIR_COLOR
        self.scalp_color = scalp_color or settings.OVERLAY_SCALP_COLOR
        self.contour_color = contour_color or settings.OVERLAY_CONTOUR_COLOR
        self.alpha = alpha if alpha is not None else settings.OVERLAY_ALPHA
        self.beta = beta if beta is not None else settings.OVERLAY_BETA

    def calculate_metrics_and_overlay(
        self, image: np.ndarray, head_mask: np.ndarray
    ) -> Tuple[VisitMetrics, np.ndarray]:
        """
        Calculates hair density and scalp visibility percentages within the head mask,
        and generates a blended visualization overlay.
        
        Args:
            image: OpenCV BGR image.
            head_mask: Binary head mask array (1 = valid ROI, 0 = background).
            
        Returns:
            Tuple[VisitMetrics, np.ndarray]: Calculated metrics object and blended overlay image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        valid_roi = (head_mask == 1)
        total_roi_pixels = int(np.sum(valid_roi))

        if total_roi_pixels == 0:
            total_roi_pixels = 1

        scalp_pixels = int(np.sum(valid_roi & (skin_mask > 0)))
        hair_pixels = max(0, total_roi_pixels - scalp_pixels)

        hair_density_pct = round((hair_pixels / total_roi_pixels) * 100.0, 2)
        scalp_visibility_pct = round((scalp_pixels / total_roi_pixels) * 100.0, 2)

        # Generate visual overlay
        overlay = image.copy()
        overlay[valid_roi & (skin_mask == 0)] = self.hair_color
        overlay[valid_roi & (skin_mask > 0)] = self.scalp_color

        contours, _ = cv2.findContours(
            head_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(overlay, contours, -1, self.contour_color, 2)

        blended = cv2.addWeighted(overlay, self.alpha, image, self.beta, 0)

        metrics = VisitMetrics(
            hair_density_percent=hair_density_pct,
            scalp_visibility_percent=scalp_visibility_pct,
            roi_total_pixels=total_roi_pixels,
            hair_pixels=hair_pixels,
            scalp_pixels=scalp_pixels,
        )
        return metrics, blended

    @staticmethod
    def calculate_delta(current: float, previous: float) -> float:
        """
        Calculates percentage change between follow-up and baseline values.
        
        Formula: ((current - previous) / previous) * 100
        """
        if previous == 0:
            return 0.0
        return round(((current - previous) / previous) * 100.0, 2)

    def compute_deltas(
        self, today_metrics: VisitMetrics, previous_metrics: VisitMetrics
    ) -> PercentageChange:
        """
        Computes percentage change deltas between today and baseline visits.
        """
        return PercentageChange(
            hair_density_change_pct=self.calculate_delta(
                today_metrics.hair_density_percent, previous_metrics.hair_density_percent
            ),
            scalp_visibility_change_pct=self.calculate_delta(
                today_metrics.scalp_visibility_percent, previous_metrics.scalp_visibility_percent
            ),
        )
