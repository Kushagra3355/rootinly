"""
Stage Determiner Logging Module.

Handles all logging operations for the Hairfall Stage Determiner microservice,
ensuring all server and request execution logs are maintained exclusively inside
the stage_determiner/logs directory.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base directory for stage_determiner and its isolated logs folder
STAGE_DETERMINER_DIR = Path(__file__).resolve().parent
LOGS_DIR = STAGE_DETERMINER_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Main persistent log file for the stage determiner service
APP_LOG_FILE = LOGS_DIR / "stage_determiner.log"


def setup_stage_determiner_logging(level: str = "INFO") -> logging.Logger:
    """
    Configures console and file logger dedicated to the stage determiner service.
    Writes all service logs into stage_determiner/logs/stage_determiner.log.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger_instance = logging.getLogger("StageDeterminer")
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger_instance.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger_instance.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger_instance.addHandler(console_handler)

        # Persistent Rotating File Handler in stage_determiner/logs
        file_handler = RotatingFileHandler(
            APP_LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger_instance.addHandler(file_handler)

    return logger_instance


# Module-level logger instance
logger = setup_stage_determiner_logging()


class StageExecutionLogger:
    """
    Collects per-request execution logs for stage determination,
    writes them to a timestamped log file inside stage_determiner/logs,
    and returns structured log entries for API responses.
    """

    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir or LOGS_DIR
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
            logger.error(f"Failed to initialize request log file: {e}")

    def log(self, message: str, level: str = "INFO") -> None:
        """Records a log message with timestamp, level, and writes to log file."""
        time_str = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"timestamp": time_str, "message": message, "level": level})

        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "DEBUG":
            logger.debug(message)
        else:
            logger.info(message)

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] [{level}] {message}\n")
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns in-memory structured log entries."""
        return self.logs

    def get_log_filepath(self) -> str:
        """Returns relative path to the log file within stage_determiner/logs."""
        return f"stage_determiner/logs/{self.log_filename}"

    def get_total_duration_ms(self) -> float:
        """Calculates elapsed time in milliseconds."""
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)
