"""Google Cloud Vision OCR wrapper."""

import os
from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class GoogleVision(BaseOCRModel):
    @property
    def name(self): return "google_vision"
    @property
    def display_name(self): return "Google Cloud Vision"
    @property
    def model_type(self): return "cloud_api"

    def setup(self):
        from google.cloud import vision
        cfg = self.config.get("google", {})
        creds_path = cfg.get("credentials_path", "")
        if creds_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        self._client = vision.ImageAnnotatorClient()
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        from google.cloud import vision

        with open(image_path, "rb") as f:
            content = f.read()

        image = vision.Image(content=content)

        # Use document_text_detection for better layout handling
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            return OCRResult(error=response.error.message)

        full_text = ""
        if response.full_text_annotation:
            full_text = response.full_text_annotation.text

        # Extract per-page confidence
        confidences = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    confidences.append(block.confidence)

        avg_conf = sum(confidences) / len(confidences) if confidences else None

        return OCRResult(
            raw_text=full_text,
            confidence=avg_conf,
            metadata={"num_blocks": len(confidences)},
        )

    def estimate_cost(self, num_pages: int) -> float:
        # Google: 1000 free/month, then ~$1.50/1000
        free_pages = 1000
        billable = max(0, num_pages - free_pages)
        return billable * 0.0015
