"""Unit tests for image preprocessing and utilities."""
import unittest
import numpy as np
import cv2
from src.rootinly.core.preprocessor import (
    decode_image_bytes,
    align_images,
    encode_image_to_base64,
)

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 100x100 BGR test image
        self.test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_img[:, :] = [120, 150, 180]
        _, self.img_encoded = cv2.imencode(".png", self.test_img)
        self.img_bytes = self.img_encoded.tobytes()

    def test_decode_image_bytes_valid(self):
        decoded = decode_image_bytes(self.img_bytes)
        self.assertIsInstance(decoded, np.ndarray)
        self.assertEqual(decoded.shape, (100, 100, 3))

    def test_decode_image_bytes_empty(self):
        with self.assertRaises(ValueError):
            decode_image_bytes(b"")

    def test_decode_image_bytes_corrupt(self):
        with self.assertRaises(ValueError):
            decode_image_bytes(b"not_an_image_binary_data_12345")

    def test_align_images(self):
        img_ref = np.zeros((200, 300, 3), dtype=np.uint8)
        img_target = np.zeros((100, 100, 3), dtype=np.uint8)
        
        ref_out, target_out = align_images(img_ref, img_target)
        self.assertEqual(ref_out.shape, (200, 300, 3))
        self.assertEqual(target_out.shape, (200, 300, 3))

    def test_encode_image_to_base64(self):
        b64_str = encode_image_to_base64(self.test_img)
        self.assertIsInstance(b64_str, str)
        self.assertTrue(len(b64_str) > 0)

if __name__ == "__main__":
    unittest.main()
