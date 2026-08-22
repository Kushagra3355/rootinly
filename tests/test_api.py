"""Integration tests for API endpoints."""
import unittest
import io
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

    def test_frontend_endpoint(self):
        response = self.client.get("/")
        self.assertIn(response.status_code, [200, 404])

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

if __name__ == "__main__":
    unittest.main()
