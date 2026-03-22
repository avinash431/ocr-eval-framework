"""Sarvam Vision OCR wrapper — India-focused multilingual OCR."""

import base64
from pathlib import Path
from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class SarvamOCR(BaseOCRModel):
    @property
    def name(self): return "sarvam_ocr"
    @property
    def display_name(self): return "Sarvam Vision OCR"
    @property
    def model_type(self): return "cloud_api"

    def setup(self):
        import requests
        self._requests = requests
        cfg = self.config.get("sarvam", {})
        if not cfg.get("api_key"):
            raise ValueError("Sarvam api_key required in config.")
        self._api_key = cfg["api_key"]
        self._endpoint = cfg.get("endpoint", "https://api.sarvam.ai/v1/ocr")
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower().lstrip(".")
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png", "pdf": "application/pdf"}.get(ext, "image/jpeg")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "image": f"data:{media};base64,{img_b64}",
        }

        resp = self._requests.post(self._endpoint, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Adapt based on actual Sarvam API response structure
        # Check their docs for the exact response format
        text = data.get("text", "") or data.get("result", {}).get("text", "")
        if not text and "pages" in data:
            text = "\n".join(p.get("text", "") for p in data["pages"])

        return OCRResult(
            raw_text=text,
            metadata={"endpoint": self._endpoint},
        )
