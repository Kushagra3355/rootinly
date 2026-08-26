"""Integration tests for unified API endpoints serving both modules."""
import io
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from src.rootinly.api.app import app

class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Create a small dummy valid image buffer
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        self.img_bytes = encoded.tobytes()

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("version", data)
        self.assertIn("model_loaded", data)
        self.assertIn("roboflow_configured", data)
        self.assertIn("active_modules", data)
        self.assertIn("crown_comparison", data["active_modules"])
        self.assertIn("stage_determiner", data["active_modules"])

    def test_frontend_serves_both_modules(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("Crown Hair Comparison", response.text)
        self.assertIn("Hairfall Stage Determiner", response.text)
        self.assertIn("compare-crowns", response.text)
        self.assertIn("predict-stage", response.text)

    def test_compare_crowns_endpoint_success(self):
        files = {
            "last_visit_image": ("baseline.jpg", io.BytesIO(self.img_bytes), "image/jpeg"),
            "today_visit_image": ("today.jpg", io.BytesIO(self.img_bytes), "image/jpeg"),
        }
        response = self.client.post("/compare-crowns", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("previous_visit", data)
        self.assertIn("today_visit", data)
        self.assertIn("percentage_change", data)
        self.assertIn("visualizations", data)

    def test_compare_crowns_missing_file(self):
        files = {
            "last_visit_image": ("baseline.jpg", io.BytesIO(self.img_bytes), "image/jpeg"),
        }
        response = self.client.post("/compare-crowns", files=files)
        self.assertEqual(response.status_code, 422)

    @patch("src.rootinly.core.stage_determiner.requests.post")
    def test_predict_stage_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "top": "Level 2",
            "confidence": 0.885,
        }
        mock_post.return_value = mock_response

        files = {
            "file": ("scalp.jpg", io.BytesIO(self.img_bytes), "image/jpeg"),
        }
        response = self.client.post("/predict-stage", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["stage"], "Level 2")
        self.assertEqual(data["confidence"], 88.5)
        self.assertIn("severity", data)
        self.assertIn("description", data)
        self.assertIn("recommendation", data)
        self.assertIn("logs", data)

    def test_predict_stage_empty_file(self):
        files = {
            "file": ("empty.jpg", io.BytesIO(b""), "image/jpeg"),
        }
        response = self.client.post("/predict-stage", files=files)
        self.assertEqual(response.status_code, 400)

    @patch("src.rootinly.core.stage_determiner.requests.post")
    def test_predict_stage_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden: Invalid API Key"
        mock_post.return_value = mock_response

        files = {
            "file": ("scalp.jpg", io.BytesIO(self.img_bytes), "image/jpeg"),
        }
        response = self.client.post("/predict-stage", files=files)
        self.assertEqual(response.status_code, 403)

    def test_stage_logs_endpoint(self):
        response = self.client.get("/api/v1/stage/logs")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("log_files", data)

    def test_stage_health_endpoint(self):
        response = self.client.get("/api/v1/stage/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("configured", data)

if __name__ == "__main__":
    unittest.main()

