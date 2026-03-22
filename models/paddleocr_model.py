"""PaddleOCR wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class PaddleOCRModel(BaseOCRModel):
    @property
    def name(self): return "paddleocr"
    @property
    def display_name(self): return "PaddleOCR"

    def setup(self):
        from paddleocr import PaddleOCR
        cfg = self.config.get("paddleocr", {})
        self._ocr = PaddleOCR(
            use_angle_cls=cfg.get("use_angle_cls", True),
            lang=cfg.get("lang", "en"),
            use_gpu=cfg.get("use_gpu", True),
            show_log=False,
        )
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        result = self._ocr.ocr(image_path, cls=True)
        lines, confidences = [], []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                lines.append(text)
                confidences.append(conf)

        raw_text = "\n".join(lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        return OCRResult(raw_text=raw_text, confidence=avg_conf,
                         metadata={"lines": len(lines)})
