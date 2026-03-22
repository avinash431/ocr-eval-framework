"""Tesseract OCR — Baseline model."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class TesseractOCR(BaseOCRModel):
    @property
    def name(self): return "tesseract"
    @property
    def display_name(self): return "Tesseract (Baseline)"

    def setup(self):
        import pytesseract
        self._tess = pytesseract
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "Tesseract not installed.\n"
                "  macOS:  brew install tesseract tesseract-lang\n"
                "  Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel"
            )
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        from PIL import Image
        cfg = self.config.get("tesseract", {})
        lang = cfg.get("lang", "eng")
        custom = f"--psm {cfg.get('psm', 3)} --oem {cfg.get('oem', 3)}"

        img = Image.open(image_path)
        text = self._tess.image_to_string(img, lang=lang, config=custom)
        data = self._tess.image_to_data(img, lang=lang, config=custom,
                                         output_type=self._tess.Output.DICT)
        confs = [int(c) for c in data["conf"] if int(c) > 0]
        avg = sum(confs) / len(confs) / 100 if confs else 0
        return OCRResult(raw_text=text, confidence=avg,
                         metadata={"lang": lang, "words": len(confs)})
