"""
Hairfall Stage Determiner Core Service.

Encapsulates Roboflow classification inference, clinical stage metadata mapping,
per-request execution logging, and log management.
"""

import base64
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import requests

from src.rootinly.config import settings
from src.rootinly.logger import logger
from src.rootinly.schemas.stage import StageResponse, StageLogsListResponse


STAGE_METADATA = {
    "level 1": {
        "severity": "Minimal / Normal",
        "description": "No significant hair loss or recession detected. Follicular density is within a healthy baseline.",
        "recommendation": "Maintain regular scalp health routine and repeat check-up every 3 to 6 months for monitoring.",
    },
    "stage 1": {
        "severity": "Minimal / Normal",
        "description": "No significant hair loss or recession detected. Follicular density is within a healthy baseline.",
        "recommendation": "Maintain regular scalp health routine and repeat check-up every 3 to 6 months for monitoring.",
    },
    "level 2": {
        "severity": "Mild Thinning",
        "description": "Mild anterior hairline maturation or early-stage thinning at the crown vertex.",
        "recommendation": "Early preventative care advised: gentle scalp stimulation, nutrient enrichment, and quarterly follow-up.",
    },
    "stage 2": {
        "severity": "Mild Thinning",
        "description": "Mild anterior hairline maturation or early-stage thinning at the crown vertex.",
        "recommendation": "Early preventative care advised: gentle scalp stimulation, nutrient enrichment, and quarterly follow-up.",
    },
    "level 3": {
        "severity": "Moderate Thinning",
        "description": "Noticeable crown vertex thinning and/or receding frontal hairline with active miniaturization.",
        "recommendation": "Targeted clinical therapy recommended: DHT-blocking topical care, trichologist evaluation, and monthly tracking.",
    },
    "stage 3": {
        "severity": "Moderate Thinning",
        "description": "Noticeable crown vertex thinning and/or receding frontal hairline with active miniaturization.",
        "recommendation": "Targeted clinical therapy recommended: DHT-blocking topical care, trichologist evaluation, and monthly tracking.",
    },
    "level 4": {
        "severity": "Advanced Thinning",
        "description": "Substantial follicular thinning across vertex and anterior zones with evident scalp exposure.",
        "recommendation": "Comprehensive clinical protocol advised: advanced medical topical/oral therapies and bi-weekly photo comparisons.",
    },
    "stage 4": {
        "severity": "Advanced Thinning",
        "description": "Substantial follicular thinning across vertex and anterior zones with evident scalp exposure.",
        "recommendation": "Comprehensive clinical protocol advised: advanced medical topical/oral therapies and bi-weekly photo comparisons.",
    },
    "level 5": {
        "severity": "Severe Loss",
        "description": "Extensive scalp visibility across crown and frontal zones with sparse remaining terminal hair.",
        "recommendation": "Specialist trichology consultation recommended for intensive restorative protocols and clinical evaluation.",
    },
    "stage 5": {
        "severity": "Severe Loss",
        "description": "Extensive scalp visibility across crown and frontal zones with sparse remaining terminal hair.",
        "recommendation": "Specialist trichology consultation recommended for intensive restorative protocols and clinical evaluation.",
    },
}


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
    Queries Roboflow classification endpoint, handles timeouts and errors,
    and enriches raw model output with clinical insights and structured logs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: Optional[float] = None,
        logs_dir: Optional[Path] = None,
    ):
        self.api_key = api_key or settings.ROBOFLOW_API_KEY
        self.model_id = model_id or settings.ROBOFLOW_MODEL_ID
        self.api_url = (api_url or settings.ROBOFLOW_API_URL).rstrip("/")
        self.timeout = timeout or settings.ROBOFLOW_TIMEOUT_SECONDS
        self.logs_dir = logs_dir or settings.STAGE_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_configured(self) -> bool:
        """Checks if all necessary Roboflow credentials are configured."""
        return bool(self.api_key and self.model_id and self.api_url)

    @property
    def endpoint_url(self) -> str:
        """Returns the full endpoint URL for inference."""
        return f"{self.api_url}/{self.model_id}?api_key={self.api_key}"

    def predict(
        self,
        image_bytes: bytes,
        filename: str = "scalp_image.jpg",
        content_type: str = "image/jpeg",
    ) -> StageResponse:
        """
        Processes image bytes, queries Roboflow classification model,
        and returns structured StageResponse with clinical metadata and execution logs.
        """
        exec_logger = StageExecutionLogger(logs_dir=self.logs_dir)
        exec_logger.log(
            f"Received stage prediction request for file: '{filename}' ({content_type})"
        )

        if not image_bytes:
            exec_logger.log("Uploaded file is empty (0 bytes).", level="ERROR")
            raise ValueError("Uploaded file is empty.")

        exec_logger.log(
            f"Image read successfully ({len(image_bytes)} bytes). Encoding to Base64..."
        )
        img_b64 = base64.b64encode(image_bytes).decode("ascii")

        exec_logger.log(
            f"Dispatching inference request to Roboflow model '{self.model_id}' at '{self.api_url}'..."
        )

        try:
            response = requests.post(
                self.endpoint_url,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                data = response.json()
                raw_stage = data.get("top", "Level 1")
                raw_conf = data.get("confidence", 0.0)
                
                # Confidence may be 0.0-1.0 or 0-100
                conf_val = float(raw_conf)
                if conf_val <= 1.0:
                    conf_pct = round(conf_val * 100, 2)
                else:
                    conf_pct = round(conf_val, 2)

                duration_ms = exec_logger.get_total_duration_ms()
                stage_key = str(raw_stage).strip().lower()
                meta = STAGE_METADATA.get(stage_key, {
                    "severity": "Clinical Assessment Required",
                    "description": f"Classified as {raw_stage}. Follicular pattern identified.",
                    "recommendation": "Review with clinical specialist and track changes over time.",
                })

                exec_logger.log(
                    f"Roboflow inference successful: Stage='{raw_stage}', Confidence={conf_pct}% (Duration: {duration_ms:.1f}ms)",
                    level="SUCCESS",
                )

                return StageResponse(
                    status="success",
                    stage=str(raw_stage),
                    confidence=conf_pct,
                    duration_ms=duration_ms,
                    duration_formatted=f"{duration_ms:.1f} ms",
                    log_file=exec_logger.get_log_filepath(),
                    logs=exec_logger.get_logs(),
                    severity=meta["severity"],
                    description=meta["description"],
                    recommendation=meta["recommendation"],
                )
            else:
                err_msg = f"Roboflow API error (HTTP {response.status_code}): {response.text}"
                exec_logger.log(err_msg, level="ERROR")
                raise RuntimeError(err_msg)

        except requests.Timeout as e:
            err_msg = f"Roboflow inference request timed out after {self.timeout}s."
            exec_logger.log(err_msg, level="ERROR")
            raise TimeoutError(err_msg) from e
        except (ValueError, RuntimeError, TimeoutError):
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
