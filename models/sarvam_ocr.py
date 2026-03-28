"""Sarvam Vision OCR wrapper — India-focused multilingual OCR."""

import base64
import json
import os
import tempfile
import zipfile
from pathlib import Path
from models import register_model
from models.base import BaseOCRModel, OCRResult

from sarvamai import SarvamAI


@register_model
class SarvamOCR(BaseOCRModel):
    @property
    def name(self):
        return "sarvam_ocr"

    @property
    def display_name(self):
        return "Sarvam Vision OCR"

    @property
    def model_type(self):
        return "cloud_api"

    def setup(self):
        cfg = self.config.get("sarvam") or {}
        self._api_key = cfg.get("api_key") or cfg.get("SARVAM_OCR_KEY")
        if not self._api_key:
            raise ValueError("Sarvam api_key required in config.")

        self._endpoint = cfg.get("endpoint", "https://api.sarvam.ai/v1/ocr")
        self._language = cfg.get("language", "en-IN")
        self._output_format = cfg.get("output_format", "md")
        self._use_sdk = cfg.get("use_sdk", True)
        if self._use_sdk and self._output_format not in {"html", "md"}:
            raise ValueError("sarvam.output_format must be 'html' or 'md' when use_sdk is true.")
        self._is_setup = True

    def _extract_strings_from_json(self, value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from self._extract_strings_from_json(v)
        elif isinstance(value, list):
            for item in value:
                yield from self._extract_strings_from_json(item)

    def _extract_text_from_json(self, data: dict) -> str:
        if not isinstance(data, dict):
            return ""

        if "blocks" in data and isinstance(data["blocks"], list):
            blocks = sorted(data["blocks"], key=lambda b: b.get("reading_order", 0))
            return " ".join(str(b.get("text", "")).strip() for b in blocks).strip()

        if "pages" in data and isinstance(data["pages"], list):
            return "\n".join(str(page.get("text", "")).strip() for page in data["pages"]).strip()

        if "text" in data and isinstance(data["text"], str):
            return data["text"].strip()

        if "result" in data and isinstance(data["result"], dict) and "text" in data["result"]:
            return str(data["result"]["text"]).strip()

        return " ".join(
            str(v).strip()
            for v in self._extract_strings_from_json(data)
            if str(v).strip()
        ).strip()

    def _load_prediction_file(self, path: Path) -> str:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._extract_text_from_json(data)
        return path.read_text(encoding="utf-8").strip()

    def _parse_zip_output(self, zip_path: Path) -> str:
        if not zip_path.exists():
            return ""

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            extracted = []
            for root, _, files in os.walk(tmpdir):
                for name in sorted(files):
                    file_path = Path(root) / name
                    if file_path.suffix.lower() in {".json", ".md", ".txt", ".html"}:
                        extracted.append(file_path)

            if not extracted:
                return ""

            text_parts = []
            for file_path in extracted:
                if file_path.suffix.lower() == ".json":
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    text_parts.append(self._extract_text_from_json(data))
                else:
                    text_parts.append(file_path.read_text(encoding="utf-8").strip())
            return "\n".join(part for part in text_parts if part).strip()

    def _ocr_impl(self, image_path: str) -> OCRResult:
        if self._use_sdk:
            if SarvamAI is None:
                raise ImportError(
                    "Sarvam SDK is not installed. Install it with `pip install sarvamai` "
                    "or set sarvam.use_sdk to false if you want fallback endpoint mode."
                )

            client = SarvamAI(api_subscription_key=self._api_key)
            job = client.document_intelligence.create_job(
                language=self._language,
                output_format=self._output_format,
            )
            job.upload_file(image_path)
            job.start()
            status = job.wait_until_complete()
            job_state = getattr(status, "job_state", None) or getattr(status, "state", None) or str(status)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "sarvam_output.zip"
                job.download_output(str(output_path))
                raw_text = self._parse_zip_output(output_path)

            return OCRResult(
                raw_text=raw_text,
                structured_data={"job_state": job_state},
                metadata={"output_format": self._output_format},
            )

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        ext = Path(image_path).suffix.lower().lstrip(".")
        media = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "pdf": "application/pdf",
        }.get(ext, "image/jpeg")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "image": f"data:{media};base64,{img_b64}",
        }

        try:
            import requests
        except ImportError as e:
            raise ImportError("requests is required for Sarvam OCR fallback mode") from e

        resp = requests.post(self._endpoint, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        raw_text = self._extract_text_from_json(data)

        return OCRResult(
            raw_text=raw_text,
            metadata={"endpoint": self._endpoint},
        )
