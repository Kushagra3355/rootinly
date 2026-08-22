"""End-to-end Crown Comparison pipeline orchestration."""
from datetime import datetime
from typing import Optional
from src.rootinly.core.analyzer import CrownAnalyzer
from src.rootinly.core.preprocessor import (
    align_images,
    decode_image_bytes,
    encode_image_to_base64,
)
from src.rootinly.core.segmentor import CrownSegmentor
from src.rootinly.logger import ExecutionLogger
from src.rootinly.schemas.analysis import Visualizations
from src.rootinly.schemas.response import ComparisonResponse

class CrownComparisonPipeline:
    """
    Orchestrates the entire AI hair comparison pipeline:
    1. Decodes and validates uploaded images.
    2. Aligns dimensions for consistent ROI comparison.
    3. Runs YOLOv8 segmentation to extract organic hairline/crown masks.
    4. Computes follicular hair density and scalp visibility.
    5. Computes clinical delta percentages.
    6. Generates blended visual heatmaps with color overlays and contours.
    7. Collects timestamped execution logs and formats response.
    """
    def __init__(
        self,
        segmentor: Optional[CrownSegmentor] = None,
        analyzer: Optional[CrownAnalyzer] = None,
    ):
        self.segmentor = segmentor or CrownSegmentor()
        self.analyzer = analyzer or CrownAnalyzer()

    def process(
        self,
        last_bytes: bytes,
        today_bytes: bytes,
        last_filename: str = "baseline.jpg",
        today_filename: str = "follow_up.jpg",
    ) -> ComparisonResponse:
        """
        Executes the comparison pipeline on two image byte streams.
        """
        exec_logger = ExecutionLogger()
        exec_logger.log(
            f"Photos received: Baseline='{last_filename}', Follow-up='{today_filename}'"
        )

        # 1. Decode images
        img_last = decode_image_bytes(last_bytes)
        img_today = decode_image_bytes(today_bytes)

        # 2. Align images
        img_last, img_today = align_images(img_last, img_today)
        h_ref, w_ref = img_last.shape[:2]
        exec_logger.log(f"Aligned images to resolution {w_ref}x{h_ref}")

        # 3. AI Segmentation Mask Extraction
        mask_last = self.segmentor.extract_mask(img_last)
        mask_today = self.segmentor.extract_mask(img_today)

        # 4. Follicular Metrics & Overlays
        metrics_last, overlay_last = self.analyzer.calculate_metrics_and_overlay(img_last, mask_last)
        metrics_today, overlay_today = self.analyzer.calculate_metrics_and_overlay(img_today, mask_today)

        exec_logger.log(
            f"Baseline: Hair = {metrics_last.hair_density_percent}%, Scalp = {metrics_last.scalp_visibility_percent}%"
        )
        exec_logger.log(
            f"Follow-up: Hair = {metrics_today.hair_density_percent}%, Scalp = {metrics_today.scalp_visibility_percent}%"
        )

        # 5. Deltas
        deltas = self.analyzer.compute_deltas(metrics_today, metrics_last)
        density_sign = "+" if deltas.hair_density_change_pct > 0 else ""
        scalp_sign = "+" if deltas.scalp_visibility_change_pct > 0 else ""
        exec_logger.log(
            f"Result: Hair Density {density_sign}{deltas.hair_density_change_pct}%, Scalp Visibility {scalp_sign}{deltas.scalp_visibility_change_pct}%"
        )

        # 6. Encode visualizations to Base64
        b64_last = encode_image_to_base64(overlay_last)
        b64_today = encode_image_to_base64(overlay_today)

        total_time_ms = exec_logger.get_total_duration_ms()
        exec_logger.log(f"AI Analysis completed in {total_time_ms:.1f} ms", level="SUCCESS")

        return ComparisonResponse(
            status="success",
            pipeline="yolov8_segmentation",
            response_time_ms=total_time_ms,
            response_time_formatted=f"{total_time_ms:.1f} ms",
            log_file=exec_logger.get_log_filepath(),
            timestamp=datetime.now().isoformat(),
            previous_visit=metrics_last,
            today_visit=metrics_today,
            percentage_change=deltas,
            visualizations=Visualizations(
                previous_visit_overlay_b64=f"data:image/png;base64,{b64_last}",
                today_visit_overlay_b64=f"data:image/png;base64,{b64_today}",
            ),
            logs=exec_logger.get_logs(),
        )
