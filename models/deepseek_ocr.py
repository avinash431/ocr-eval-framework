"""DeepSeek OCR wrapper — VLM-based."""

from models import register_model
from models.base import BaseOCRModel, OCRResult
from utils.helpers import get_device


@register_model
class DeepSeekOCR(BaseOCRModel):
    @property
    def name(self): return "deepseek_ocr"
    @property
    def display_name(self): return "DeepSeek OCR"

    def setup(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        cfg = self.config.get("deepseek_ocr", {})
        model_path = cfg.get("model_path", "deepseek-ai/DeepSeek-OCR")
        self._device = get_device(self.config)

        print(f"  Loading DeepSeek OCR on {self._device} (this may take a few minutes)...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype,
        ).to(self._device).eval()
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")

        # DeepSeek OCR uses a specific prompt format
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "OCR this document. Extract all text preserving layout."},
        ]}]

        inputs = self._tokenizer.apply_chat_template(messages, return_tensors="pt",
                                                      add_generation_prompt=True)
        inputs = inputs.to(self._device)

        with __import__("torch").no_grad():
            outputs = self._model.generate(inputs, max_new_tokens=4096)

        text = self._tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return OCRResult(raw_text=text, metadata={"device": str(self._device)})
