"""Unit and integration tests for Hairfall Stage Determiner with isolated logging."""
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from fastapi.testclient import TestClient

from stage_determiner.app import app
from stage_determiner.logger import (
    LOGS_DIR,
    APP_LOG_FILE,
    StageExecutionLogger,
    setup_stage_determiner_logging,
)


class TestHairfallStageDeterminer(unittest.TestCase):
    """Test suite for stage determiner and its isolated logging functionality."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_logs_directory_isolated(self):
        """Verify that logs directory is strictly located inside stage_determiner/logs."""
        self.assertTrue(LOGS_DIR.exists())
        self.assertEqual(LOGS_DIR.parent.name, "stage_determiner")
        self.assertEqual(LOGS_DIR.name, "logs")

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
        """Test GET /api/v1/health returns healthy status and logs directory path."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("logs_directory", data)
        self.assertTrue(data["logs_directory"].endswith(str(Path("stage_determiner/logs"))))

    def test_list_logs_endpoint(self):
        """Test GET /api/v1/logs lists logs inside stage_determiner/logs."""
        response = self.client.get("/api/v1/logs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("logs_directory", data)
        self.assertIn("log_files", data)
        self.assertIsInstance(data["log_files"], list)

    def test_empty_file_upload(self):
        """Test POST /predict-stage with empty bytes fails with 400 and logs error."""
        response = self.client.post(
            "/predict-stage",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Uploaded file is empty.")

    @patch("stage_determiner.app.requests.post")
    def test_predict_stage_endpoint_success(self, mock_post):
        """Test POST /predict-stage returns stage, confidence, and records logs in background."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "top": "Stage 2",
            "confidence": 0.942,
        }
        mock_post.return_value = mock_response

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
        self.assertEqual(data["stage"], "Stage 2")
        self.assertEqual(data["confidence"], 94.2)

        # Check logs returned in response
        self.assertIn("logs", data)
        self.assertIsInstance(data["logs"], list)
        self.assertGreater(len(data["logs"]), 0)

        # Check log file path
        self.assertIn("log_file", data)
        self.assertTrue(data["log_file"].startswith("stage_determiner/logs/"))

        # Verify the actual file was created in stage_determiner/logs
        log_filename = Path(data["log_file"]).name
        created_log_path = LOGS_DIR / log_filename
        self.assertTrue(created_log_path.exists())

        # Verify log file contents
        content = created_log_path.read_text(encoding="utf-8")
        self.assertIn("Hairfall Stage Determination Log", content)
        self.assertIn("Stage='Stage 2'", content)

    @patch("stage_determiner.app.requests.post")
    def test_predict_stage_endpoint_roboflow_error(self, mock_post):
        """Test POST /predict-stage handles Roboflow API errors with logging."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden: Invalid API Key"
        mock_post.return_value = mock_response

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", img)
        img_bytes = buffer.tobytes()

        response = self.client.post(
            "/predict-stage",
            files={"file": ("test_crown.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Roboflow API error", response.json()["detail"])

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
        self.assertTrue(filepath.startswith("stage_determiner/logs/"))

        actual_file = LOGS_DIR / Path(filepath).name
        self.assertTrue(actual_file.exists())
        file_content = actual_file.read_text(encoding="utf-8")
        self.assertIn("[INFO] Starting test execution step", file_content)
        self.assertIn("[WARNING] Warning encountered", file_content)
        self.assertIn("[ERROR] Error simulated", file_content)


if __name__ == "__main__":
    unittest.main()
