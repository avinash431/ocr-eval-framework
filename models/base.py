"""Base OCR model interface. All wrappers inherit from this."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCRResult:
    """Standardized result from any OCR model."""
    model_name: str = ""
    document_path: str = ""
    raw_text: str = ""
    structured_data: Optional[dict] = None
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and len(self.raw_text.strip()) > 0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name, "document_path": self.document_path,
            "raw_text": self.raw_text, "structured_data": self.structured_data,
            "confidence": self.confidence, "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 6), "error": self.error,
            "success": self.success, "metadata": self.metadata,
        }


class BaseOCRModel(ABC):
    """
    Base class. Subclasses must implement `name` and `_ocr_impl`.
    The base class handles latency capture, retries, and error handling.
    """

    def __init__(self, config: dict):
        self.config = config
        self._is_setup = False

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def model_type(self) -> str:
        return "open_source"

    def setup(self):
        self._is_setup = True

    def teardown(self):
        self._is_setup = False

    @abstractmethod
    def _ocr_impl(self, image_path: str) -> OCRResult:
        pass

    def ocr(self, image_path: str) -> OCRResult:
        if not self._is_setup:
            self.setup()

        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            return OCRResult(model_name=self.name, document_path=image_path,
                             error=f"File not found: {image_path}")

        max_retries = self.config.get("execution", {}).get("max_retries", 2)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                start = time.perf_counter()
                result = self._ocr_impl(image_path)
                result.latency_ms = (time.perf_counter() - start) * 1000
                result.document_path = image_path
                result.model_name = self.name
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))

        return OCRResult(model_name=self.name, document_path=image_path,
                         error=f"Failed after {max_retries + 1} attempts: {last_error}")

    def ocr_batch(self, image_paths: list) -> list:
        return [self.ocr(p) for p in image_paths]

    def estimate_cost(self, num_pages: int) -> float:
        return 0.0
