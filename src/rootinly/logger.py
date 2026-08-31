import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.rootinly.config import settings

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configures console and file logger for the application."""
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("RootinlyAI")

logger = setup_logging()

class ExecutionLogger:
    """
    Collects execution logs per request, writes them to a timestamped file,
    and returns them structured for API responses.
    """
    def __init__(self, logs_dir: Optional[Path] = None):
        self.logs_dir = logs_dir or settings.GROWTH_COMP_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs: List[Dict[str, str]] = []
        self.start_time = time.perf_counter()
        
        timestamp_file_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"{timestamp_file_str}.log"
        self.log_file_path = self.logs_dir / self.log_filename

        try:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write(f"=== Crown Hair Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
        except Exception as e:
            logger.error(f"Failed to initialize log file: {e}")

    def log(self, message: str, level: str = "INFO") -> None:
        """Records a log message with level and timestamp."""
        time_str = datetime.now().strftime("%H:%M:%S")
        self.logs.append({"timestamp": time_str, "message": message, "level": level})

        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{time_str}] [{level}] {message}\n")
        except Exception:
            pass

    def get_logs(self) -> List[Dict[str, str]]:
        """Returns the in-memory log entries."""
        return self.logs

    def get_log_filepath(self) -> str:
        """Returns the relative path to the log file within logs/growth_comparison."""
        try:
            rel_path = self.log_file_path.relative_to(settings.BASE_DIR)
            return str(rel_path).replace("\\", "/")
        except Exception:
            return f"logs/growth_comparison/{self.log_filename}"

    def get_total_duration_ms(self) -> float:
        """Calculates elapsed time in milliseconds."""
        return round((time.perf_counter() - self.start_time) * 1000.0, 2)
