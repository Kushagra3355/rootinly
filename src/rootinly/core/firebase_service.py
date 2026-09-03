"""Firebase Admin integration service for storing feedback and comparison images."""
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from src.rootinly.config import settings
from src.rootinly.logger import logger

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
    HAS_FIREBASE_ADMIN = True
except ImportError:
    HAS_FIREBASE_ADMIN = False


class FirebaseService:
    """Manages Firestore database and Firebase Storage operations."""

    def __init__(self):
        self.db = None
        self.bucket = None
        self.is_initialized = False
        self.initialize()

    def initialize(self) -> bool:
        """Initializes Firebase Admin SDK if credentials exist."""
        if not HAS_FIREBASE_ADMIN:
            logger.warning("[Firebase] firebase-admin package is not installed.")
            return False

        if self.is_initialized and self.db is not None:
            return True

        candidates = [
            Path(settings.FIREBASE_CREDENTIALS_PATH),
            settings.BASE_DIR / "firebaseServiceAccount.json",
            settings.BASE_DIR / "firebase-credentials.json",
        ]

        cred_path = None
        for cand in candidates:
            p = cand if cand.is_absolute() else settings.BASE_DIR / cand
            if p.exists():
                cred_path = p
                break

        if not cred_path:
            logger.info(
                "[Firebase] No credentials file found (checked firebaseServiceAccount.json, firebase-credentials.json). "
                "Firebase features will be disabled until credentials are provided."
            )
            return False


        try:
            storage_bucket = settings.FIREBASE_STORAGE_BUCKET
            if not firebase_admin._apps:
                cred = credentials.Certificate(str(cred_path))
                app_options = {}
                if storage_bucket:
                    app_options["storageBucket"] = storage_bucket
                elif hasattr(cred, "project_id") and cred.project_id:
                    app_options["storageBucket"] = f"{cred.project_id}.appspot.com"
                firebase_admin.initialize_app(cred, app_options if app_options else None)

            self.db = firestore.client()
            try:
                self.bucket = storage.bucket()
            except Exception as st_err:
                logger.warning(f"[Firebase] Storage bucket initialization warning: {st_err}")
                self.bucket = None

            self.is_initialized = True
            logger.info("[Firebase] Firestore and Firebase Storage connected successfully.")
            return True

        except Exception as e:
            logger.error(f"[Firebase] Failed to initialize Firebase: {e}")
            self.is_initialized = False
            return False

    def upload_image_bytes(
        self,
        image_bytes: bytes,
        destination_path: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Uploads image bytes to Firebase Storage and returns public or signed URL."""
        if not self.is_initialized or not self.bucket:
            raise RuntimeError("Firebase Storage is not initialized or configured.")

        blob = self.bucket.blob(destination_path)
        blob.upload_from_string(image_bytes, content_type=content_type)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.name

    def save_crown_feedback(
        self,
        feedback_data: Dict[str, Any],
        baseline_bytes: Optional[bytes] = None,
        today_bytes: Optional[bytes] = None,
    ) -> str:
        """Saves user questionnaire feedback along with visit images to Firebase."""
        # Attempt late initialization if credentials were newly placed
        if not self.is_initialized:
            self.initialize()

        if not self.is_initialized or not self.db:
            raise RuntimeError(
                "Firebase is not configured yet. Please add 'firebase-credentials.json' in the project root "
                "or set FIREBASE_CREDENTIALS_PATH in your .env file."
            )

        feedback_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        image_urls = {}

        # Upload baseline image if present
        if baseline_bytes and self.bucket:
            try:
                dest = f"crown_comparisons/{feedback_id}/baseline.jpg"
                image_urls["baseline_image_url"] = self.upload_image_bytes(baseline_bytes, dest)
            except Exception as e:
                logger.warning(f"[Firebase] Failed to upload baseline image to storage: {e}")

        # Upload today/follow-up image if present
        if today_bytes and self.bucket:
            try:
                dest = f"crown_comparisons/{feedback_id}/today.jpg"
                image_urls["today_image_url"] = self.upload_image_bytes(today_bytes, dest)
            except Exception as e:
                logger.warning(f"[Firebase] Failed to upload follow-up image to storage: {e}")

        doc_payload = {
            "feedback_id": feedback_id,
            "created_at": created_at,
            "patient_name": feedback_data.get("patient_name", ""),
            "time_since_treatment": feedback_data.get("time_since_treatment", ""),
            "questions": {
                "patient_name": feedback_data.get("patient_name", ""),
                "time_since_treatment": feedback_data.get("time_since_treatment", ""),
                "is_prev_masking_correct": feedback_data.get("is_prev_masking_correct"),
                "prev_masking_accuracy_pct": feedback_data.get("prev_masking_pct"),
                "is_curr_masking_correct": feedback_data.get("is_curr_masking_correct"),
                "curr_masking_accuracy_pct": feedback_data.get("curr_masking_pct"),
                "is_prev_classification_correct": feedback_data.get("is_prev_classification_correct"),
                "prev_classification_accuracy_pct": feedback_data.get("prev_classification_pct"),
                "is_curr_classification_correct": feedback_data.get("is_curr_classification_correct"),
                "curr_classification_accuracy_pct": feedback_data.get("curr_classification_pct"),
                "is_result_valid": feedback_data.get("is_result_valid"),
                "overall_validity_pct": feedback_data.get("overall_validity_pct"),
            },
            "images": image_urls,
            "metrics": feedback_data.get("metrics", {}),
            "notes": feedback_data.get("notes", ""),
        }



        self.db.collection("hair_growth_feedback").document(feedback_id).set(doc_payload)
        logger.info(f"[Firebase] Stored hair growth feedback with ID: {feedback_id}")
        return feedback_id


firebase_service = FirebaseService()
