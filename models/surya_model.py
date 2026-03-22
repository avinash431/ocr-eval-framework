"""Surya OCR wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class SuryaModel(BaseOCRModel):
    @property
    def name(self): return "surya"
    @property
    def display_name(self): return "Surya OCR"

    def setup(self):
        from surya.ocr import run_ocr
        from surya.model.detection.model import load_model as load_det_model
        from surya.model.detection.model import load_processor as load_det_proc
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_proc

        self._run_ocr = run_ocr
        self._det_model = load_det_model()
        self._det_proc = load_det_proc()
        self._rec_model = load_rec_model()
        self._rec_proc = load_rec_proc()
        self._langs = self.config.get("surya", {}).get("langs", ["en"])
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        from PIL import Image
        img = Image.open(image_path)
        results = self._run_ocr(
            [img], [self._langs],
            self._det_model, self._det_proc,
            self._rec_model, self._rec_proc,
        )
        lines = []
        confidences = []
        if results:
            for line in results[0].text_lines:
                lines.append(line.text)
                confidences.append(line.confidence)

        raw_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return OCRResult(raw_text=raw_text, confidence=avg_conf,
                         metadata={"lines": len(lines), "langs": self._langs})
