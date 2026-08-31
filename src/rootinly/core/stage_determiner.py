"""
Hairfall Stage Determiner Core Service.

Encapsulates local YOLOv8 Norwood Scale classification inference,
per-request execution logging, and log management.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from src.rootinly.config import settings
from src.rootinly.logger import logger
from src.rootinly.schemas.stage import StageResponse, StageLogsListResponse


class StageExecutionLogger:
    """
    Collects per-request execution logs for stage determination,
    writes them to a timestamped log file inside the stage logs directory,
    and returns structured log entries for API responses.
    """

    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir or settings.STAGE_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs: List[Dict[str, str]] = []
        self.start_time = time.perf_counter()

        timestamp_file_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"stage_predict_{timestamp_file_str}"
        candidate_path = self.logs_dir / f"{base_name}.log"
        if candidate_path.exists():
            base_name = f"{base_name}_{datetime.now().strftime('%f')}"
            candidate_path = self.logs_dir / f"{base_name}.log"

        self.log_filename = f"{base_name}.log"
        self.log_file_path = candidate_path

        try:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write(
                    f"=== Hairfall Stage Determination Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"
                )
        except Exception as e:
            logger.error(f"Failed to initialize stage request log file: {e}")

    def log(self, message: str, level: str = "INFO") -> None:
        """Records a log message with timestamp, level, and writes to log file."""
        time_str = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"timestamp": time_str, "message": message, "level": level})

        if level == "ERROR":
            logger.error(f"[StageDeterminer] {message}")
        elif level == "WARNING":
            logger.warning(f"[StageDeterminer] {message}")
        elif level == "DEBUG":
            logger.debug(f"[StageDeterminer] {message}")
        elif level == "SUCCESS":
            logger.info(f"[StageDeterminer] [SUCCESS] {message}")
        else:
            logger.info(f"[StageDeterminer] {message}")

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] [{level}] {message}\n")
        except Exception as e:
            logger.error(f"Failed to write to stage log file: {e}")

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns in-memory structured log entries."""
        return self.logs

    def get_log_filepath(self) -> str:
        """Returns relative path to the log file within stage_determiner/logs."""
        try:
            rel_path = self.log_file_path.relative_to(settings.BASE_DIR)
            return str(rel_path).replace("\\", "/")
        except Exception:
            return f"stage_determiner/logs/{self.log_filename}"

    def get_total_duration_ms(self) -> float:
        """Calculates elapsed time in milliseconds."""
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)


class StageDeterminerService:
    """
    Service class managing Hairfall Stage Determination.
    Loads and runs local YOLOv8 Norwood Scale classification model ('best_norwood.pt'),
    and provides structured execution logs.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        logs_dir: Optional[Path] = None,
    ):
        self.model_path = model_path or settings.get_stage_model_path()
        self.logs_dir = logs_dir or settings.STAGE_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO stage classification model into memory."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Stage model path '{self.model_path}' does not exist. Attempting fallback...")
                if settings.FALLBACK_STAGE_MODEL_PATH.exists():
                    self.model_path = settings.FALLBACK_STAGE_MODEL_PATH
                else:
                    raise FileNotFoundError(
                        f"Stage classification weights not found at primary '{self.model_path}' or fallback '{settings.FALLBACK_STAGE_MODEL_PATH}'"
                    )

            self.model = YOLO(str(self.model_path))
            logger.info(f"Successfully loaded YOLOv8 Norwood stage classification model from '{self.model_path}'.")
        except Exception as e:
            logger.error(f"Failed to load YOLO stage classification model: {e}")
            self.model = None

    @property
    def is_loaded(self) -> bool:
        """Checks if the YOLO stage classification model is loaded."""
        return self.model is not None

    @property
    def is_configured(self) -> bool:
        """Backward-compatible readiness indicator."""
        return self.is_loaded

    def predict(
        self,
        image_bytes: bytes,
        filename: str = "scalp_image.jpg",
        content_type: str = "image/jpeg",
    ) -> StageResponse:
        """
        Processes image bytes, runs local YOLOv8 Norwood classification inference,
        and returns structured StageResponse with execution logs.
        """
        exec_logger = StageExecutionLogger(logs_dir=self.logs_dir)
        exec_logger.log(
            f"Received stage prediction request for file: '{filename}' ({content_type})"
        )

        if not image_bytes:
            exec_logger.log("Uploaded file is empty (0 bytes).", level="ERROR")
            raise ValueError("Uploaded file is empty.")

        exec_logger.log(
            f"Image read successfully ({len(image_bytes)} bytes). Decoding image buffer..."
        )

        if self.model is None:
            exec_logger.log("YOLO Norwood stage classification model is not loaded.", level="ERROR")
            raise RuntimeError("YOLO Norwood stage classification model is not loaded.")

        # Decode image bytes to OpenCV BGR image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            exec_logger.log("Failed to decode image from uploaded bytes.", level="ERROR")
            raise ValueError("Failed to decode image. Ensure valid JPG/PNG format.")

        h, w = img.shape[:2]
        exec_logger.log(f"Decoded image dimensions: {w}x{h} (channels: 3).")
        exec_logger.log(f"Running inference using local YOLO model '{self.model_path.name}'...")

        try:
            results = self.model.predict(source=img, verbose=False)

            if not results or results[0].probs is None:
                raise RuntimeError("YOLO model returned empty classification output.")

            probs = results[0].probs
            top1_idx = int(probs.top1)
            raw_class = str(self.model.names.get(top1_idx, top1_idx))
            conf_val = float(probs.top1conf)
            conf_pct = round(conf_val * 100, 2)

            # Format stage name
            if raw_class.isdigit():
                stage_name = f"Stage {raw_class}"
            elif not raw_class.lower().startswith("stage") and not raw_class.lower().startswith("level"):
                stage_name = f"Stage {raw_class}"
            else:
                stage_name = raw_class

            duration_ms = exec_logger.get_total_duration_ms()

            exec_logger.log(
                f"YOLO Norwood inference successful: Stage='{stage_name}' (Class ID: {raw_class}), Confidence={conf_pct}% (Duration: {duration_ms:.1f}ms)",
                level="SUCCESS",
            )

            return StageResponse(
                status="success",
                stage=stage_name,
                confidence=conf_pct,
                duration_ms=duration_ms,
                duration_formatted=f"{duration_ms:.1f} ms",
                log_file=exec_logger.get_log_filepath(),
                logs=exec_logger.get_logs(),
            )

        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            err_msg = f"Unhandled error during stage prediction: {str(e)}"
            exec_logger.log(err_msg, level="ERROR")
            raise RuntimeError(err_msg) from e

    def list_logs(self) -> StageLogsListResponse:
        """Returns list of timestamped log files in the stage logs directory."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_files = sorted(
            [f.name for f in self.logs_dir.glob("*.log")],
            reverse=True,
        )
        return StageLogsListResponse(
            status="success",
            logs_directory=str(self.logs_dir.resolve()),
            total_log_files=len(log_files),
            log_files=log_files,
        )
