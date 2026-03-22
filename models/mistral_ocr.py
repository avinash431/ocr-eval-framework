"""Mistral OCR wrapper."""

import base64
from pathlib import Path
from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class MistralOCR(BaseOCRModel):
    @property
    def name(self): return "mistral_ocr"
    @property
    def display_name(self): return "Mistral OCR"
    @property
    def model_type(self): return "cloud_api"

    def setup(self):
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral
        cfg = self.config.get("mistral", {})
        if not cfg.get("api_key"):
            raise ValueError("Mistral api_key required in config.")
        self._client = Mistral(api_key=cfg["api_key"])
        self._model = cfg.get("model", "mistral-ocr-latest")
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        ext = Path(image_path).suffix.lower()

        if ext == ".pdf":
            # PDF: upload as document
            with open(image_path, "rb") as f:
                uploaded = self._client.files.upload(file={
                    "file_name": Path(image_path).name,
                    "content": f.read(),
                })
            signed_url = self._client.files.get_signed_url(file_id=uploaded.id)
            response = self._client.ocr.process(
                model=self._model,
                document={"type": "document_url", "document_url": signed_url.url},
            )
        else:
            # Image: encode as base64 data URI
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            media = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                     "png": "image/png"}.get(ext.lstrip("."), "image/jpeg")
            data_uri = f"data:{media};base64,{img_b64}"
            response = self._client.ocr.process(
                model=self._model,
                document={"type": "image_url", "image_url": data_uri},
            )

        # Extract text from pages
        all_text = []
        for page in response.pages:
            all_text.append(page.markdown)

        raw_text = "\n\n".join(all_text)

        return OCRResult(
            raw_text=raw_text,
            metadata={"pages": len(response.pages), "model": self._model},
        )

    def estimate_cost(self, num_pages: int) -> float:
        # Mistral: ~$1 per 1000 pages
        return num_pages * 0.001
