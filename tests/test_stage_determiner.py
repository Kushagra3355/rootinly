"""Unit and integration tests for Hairfall Stage Determiner microservice."""
import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from stage_determiner.app import app, CLINICAL_STAGE_METADATA, stage_engine

class TestHairfallStageDeterminer(unittest.TestCase):
    """Test suite for simple stage determiner endpoints and logic."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_serve_frontend(self):
        """Test GET / serves the HTML single-page app."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("Hairfall Stage Determiner", response.text)

    def test_health_check(self):
        """Test GET /api/v1/health returns healthy status."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("default_model_id"), "hyehye2/1")

    def test_stages_knowledge_base(self):
        """Test GET /api/v1/stages returns all 7 clinical stages."""
        response = self.client.get("/api/v1/stages")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("total_stages"), 7)
        self.assertIn("Level 1", data.get("stages", {}))
        self.assertIn("Level 7", data.get("stages", {}))

    def test_heuristic_fallback_analysis(self):
        """Test heuristic stage analyzer on synthetic images."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        fallback = stage_engine._fallback_heuristic_analysis(img)
        self.assertIn("top", fallback)
        self.assertIn("predictions", fallback)
        self.assertGreaterEqual(len(fallback["predictions"]), 1)

    def test_overlay_generation(self):
        """Test OpenCV clinical badge overlay generation."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay_b64 = stage_engine._generate_annotated_overlay(img, "Level 3", 0.85)
        self.assertTrue(overlay_b64.startswith("data:image/png;base64,"))

    def test_predict_stage_endpoint(self):
        """Test POST /predict-stage with generated image bytes."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", img)
        img_bytes = buffer.tobytes()

        response = self.client.post(
            "/predict-stage",
            files={"file": ("scalp.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("Level", data.get("stage", ""))
        self.assertIn("recommendations", data)
        self.assertIn("all_predictions", data)
        self.assertGreaterEqual(len(data.get("all_predictions", [])), 1)

    def test_empty_file_upload(self):
        """Test POST /predict-stage with empty bytes fails with 400."""
        response = self.client.post(
            "/predict-stage",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
