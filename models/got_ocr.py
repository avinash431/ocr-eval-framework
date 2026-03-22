"""GOT-OCR 2.0 wrapper — General OCR Transformer."""

from models import register_model
from models.base import BaseOCRModel, OCRResult
from utils.helpers import get_device


@register_model
class GOTOCR(BaseOCRModel):
    @property
    def name(self): return "got_ocr"
    @property
    def display_name(self): return "GOT-OCR 2.0"

    def setup(self):
        from transformers import AutoModel, AutoTokenizer
        import torch

        cfg = self.config.get("got_ocr", {})
        model_name = cfg.get("model_name", "stepfun-ai/GOT-OCR-2.0-hf")
        self._device = get_device(self.config)

        print(f"  Loading GOT-OCR 2.0 on {self._device}...")
        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype,
        ).to(self._device).eval()
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        import torch
        # GOT-OCR has its own chat interface
        with torch.no_grad():
            text = self._model.chat(
                self._tokenizer, image_path, ocr_type="ocr",
            )
        return OCRResult(raw_text=text, metadata={"device": str(self._device)})
