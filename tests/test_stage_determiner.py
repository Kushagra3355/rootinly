"""Unit and integration tests for Hairfall Stage Determiner with isolated logging."""
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.rootinly.api.app import app
from src.rootinly.config import settings
from src.rootinly.core.stage_determiner import (
    StageExecutionLogger,
    StageDeterminerService,
)

LOGS_DIR = settings.STAGE_LOGS_DIR



class TestHairfallStageDeterminer(unittest.TestCase):
    """Test suite for stage determiner and its isolated logging functionality."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_logs_directory_isolated(self):
        """Verify that logs directory is strictly located inside logs/stage_determiner."""
        self.assertTrue(LOGS_DIR.exists())
        self.assertEqual(LOGS_DIR.parent.name, "logs")
        self.assertEqual(LOGS_DIR.name, "stage_determiner")

    def test_serve_frontend(self):
        """Test GET / serves the clean HTML interface without frontend execution logs."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("Hairfall Stage Determiner", response.text)
        self.assertIn("resStage", response.text)
        self.assertIn("resConfidence", response.text)
        self.assertNotIn("logsWrapper", response.text)

    def test_health_check(self):
        """Test GET /api/v1/stage/health returns healthy status and logs directory path."""
        response = self.client.get("/api/v1/stage/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("logs_directory", data)
        self.assertTrue(data["logs_directory"].endswith(str(Path("logs/stage_determiner"))))


    def test_empty_file_upload(self):
        """Test POST /predict-stage with empty bytes fails with 400 and logs error."""
        response = self.client.post(
            "/predict-stage",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Uploaded file is empty.")

    def test_predict_stage_endpoint_success(self):
        """Test POST /predict-stage returns stage, confidence, and records logs in background."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", img)
        img_bytes = buffer.tobytes()

        response = self.client.post(
            "/predict-stage",
            files={"file": ("test_crown.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check prediction results
        self.assertIn("stage", data)
        self.assertTrue(data["stage"].startswith("Stage") or data["stage"].startswith("Level") or data["stage"].isdigit())
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 100.0)

        # Check logs returned in response
        self.assertIn("logs", data)
        self.assertIsInstance(data["logs"], list)
        self.assertGreater(len(data["logs"]), 0)

        # Check log file path
        self.assertIn("log_file", data)
        self.assertTrue(data["log_file"].startswith("logs/stage_determiner/"))

        # Verify the actual file was created in logs/stage_determiner
        log_filename = Path(data["log_file"]).name
        created_log_path = LOGS_DIR / log_filename
        self.assertTrue(created_log_path.exists())

        # Verify log file contents
        content = created_log_path.read_text(encoding="utf-8")
        self.assertIn("Hairfall Stage Determination Log", content)

    def test_predict_stage_endpoint_corrupt_file(self):
        """Test POST /predict-stage handles corrupt image file bytes."""
        response = self.client.post(
            "/predict-stage",
            files={"file": ("corrupt.jpg", b"invalid_non_image_binary_data", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Failed to decode image", response.json()["detail"])

    def test_stage_execution_logger_unit(self):
        """Unit test StageExecutionLogger functionality."""
        exec_logger = StageExecutionLogger()
        exec_logger.log("Starting test execution step", level="INFO")
        exec_logger.log("Warning encountered", level="WARNING")
        exec_logger.log("Error simulated", level="ERROR")

        logs = exec_logger.get_logs()
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]["message"], "Starting test execution step")
        self.assertEqual(logs[0]["level"], "INFO")
        self.assertEqual(logs[1]["level"], "WARNING")
        self.assertEqual(logs[2]["level"], "ERROR")

        duration = exec_logger.get_total_duration_ms()
        self.assertGreaterEqual(duration, 0.0)

        filepath = exec_logger.get_log_filepath()
        self.assertTrue(filepath.startswith("logs/stage_determiner/"))

        actual_file = LOGS_DIR / Path(filepath).name
        self.assertTrue(actual_file.exists())
        file_content = actual_file.read_text(encoding="utf-8")
        self.assertIn("[INFO] Starting test execution step", file_content)
        self.assertIn("[WARNING] Warning encountered", file_content)
        self.assertIn("[ERROR] Error simulated", file_content)


if __name__ == "__main__":
    unittest.main()
