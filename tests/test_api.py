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

    def test_predict_stage_success(self):
        files = {
            "file": ("scalp.jpg", io.BytesIO(self.img_bytes), "image/jpeg"),
        }
        response = self.client.post("/predict-stage", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("stage", data)
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertIn("logs", data)

    def test_predict_stage_empty_file(self):
        files = {
            "file": ("empty.jpg", io.BytesIO(b""), "image/jpeg"),
        }
        response = self.client.post("/predict-stage", files=files)
        self.assertEqual(response.status_code, 400)

    def test_predict_stage_invalid_file(self):
        files = {
            "file": ("bad.jpg", io.BytesIO(b"not_an_image_data"), "image/jpeg"),
        }
        response = self.client.post("/predict-stage", files=files)
        self.assertEqual(response.status_code, 400)


    def test_crown_health_endpoint(self):
        response = self.client.get("/api/v1/crown/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)
        self.assertIn("service", data)

    def test_stage_health_endpoint(self):
        response = self.client.get("/api/v1/stage/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)
        self.assertIn("service", data)

    @patch("src.rootinly.api.routes.feedback.firebase_service.save_crown_feedback")
    def test_feedback_crown_with_patient_info(self, mock_save):
        mock_save.return_value = "test-feedback-uuid-123"
        form_data = {
            "patient_name": "John Doe",
            "time_since_treatment": "3 months",
            "is_prev_masking_correct": "true",
            "prev_masking_pct": "95.0",
            "is_curr_masking_correct": "true",
            "curr_masking_pct": "90.0",
            "is_prev_classification_correct": "true",
            "prev_classification_pct": "85.0",
            "is_curr_classification_correct": "true",
            "curr_classification_pct": "88.0",
            "is_result_valid": "true",
            "overall_validity_pct": "92.0",
            "notes": "Patient shows improvement in vertex density.",
        }
        response = self.client.post("/api/v1/feedback/crown", data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feedback_id"], "test-feedback-uuid-123")
        mock_save.assert_called_once()
        saved_data = mock_save.call_args[1]["feedback_data"]
        self.assertEqual(saved_data["patient_name"], "John Doe")
        self.assertEqual(saved_data["time_since_treatment"], "3 months")

    @patch("src.rootinly.api.routes.feedback.firebase_service.save_crown_feedback")
    def test_feedback_crown_alias_with_patient_info(self, mock_save):
        mock_save.return_value = "test-feedback-alias-456"
        form_data = {
            "patient_name": "Jane Smith",
            "time_since_treatment": "6 months",
            "is_prev_masking_correct": "true",
            "prev_masking_pct": "100.0",
            "is_curr_masking_correct": "true",
            "curr_masking_pct": "100.0",
            "is_prev_classification_correct": "true",
            "prev_classification_pct": "100.0",
            "is_curr_classification_correct": "true",
            "curr_classification_pct": "100.0",
            "is_result_valid": "true",
            "overall_validity_pct": "100.0",
        }
        response = self.client.post("/feedback-crown", data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["feedback_id"], "test-feedback-alias-456")
        saved_data = mock_save.call_args[1]["feedback_data"]
        self.assertEqual(saved_data["patient_name"], "Jane Smith")
        self.assertEqual(saved_data["time_since_treatment"], "6 months")

if __name__ == "__main__":
    unittest.main()
