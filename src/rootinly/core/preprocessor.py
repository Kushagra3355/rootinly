"""Image preprocessing, alignment, and encoding utilities."""
import base64
from typing import Tuple
import cv2
import numpy as np

def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decodes raw image bytes into an OpenCV BGR numpy image array.
    
    Args:
        image_bytes: Raw bytes from uploaded file.
        
    Returns:
        np.ndarray: OpenCV BGR image.
        
    Raises:
        ValueError: If bytes are empty or image cannot be decoded.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Image bytes cannot be empty.")
    
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image. Unsupported or corrupted image format.")
    
    return image

def align_images(ref_img: np.ndarray, target_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatially aligns target image to reference image dimensions for consistent ROI comparison.
    
    Args:
        ref_img: Reference image (baseline visit).
        target_img: Target image to align (follow-up visit).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ref_img, target_img_aligned).
    """
    h_ref, w_ref = ref_img.shape[:2]
    target_aligned = cv2.resize(target_img, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)
    return ref_img, target_aligned

def encode_image_to_base64(image: np.ndarray, format_ext: str = ".png") -> str:
    """
    Encodes an OpenCV image to a base64 UTF-8 string for frontend delivery.
    
    Args:
        image: OpenCV BGR image array.
        format_ext: Image format extension (e.g., '.png', '.jpg').
        
    Returns:
        str: Base64-encoded string.
        
    Raises:
        ValueError: If encoding fails.
    """
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        raise ValueError(f"Failed to encode image to {format_ext} buffer.")
    return base64.b64encode(buffer).decode("utf-8")
