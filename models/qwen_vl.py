"""Qwen2.5-VL wrapper — Alibaba VLM for OCR."""

from models import register_model
from models.base import BaseOCRModel, OCRResult
from utils.helpers import get_device


@register_model
class QwenVL(BaseOCRModel):
    @property
    def name(self): return "qwen_vl"
    @property
    def display_name(self): return "Qwen2.5-VL"

    def setup(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        cfg = self.config.get("qwen_vl", {})
        model_name = cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self._device = get_device(self.config)

        print(f"  Loading Qwen2.5-VL on {self._device}...")
        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype,
        ).to(self._device).eval()
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        from PIL import Image
        import torch

        img = Image.open(image_path).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "OCR this document. Extract all text preserving layout. Use markdown tables for any tabular data."},
        ]}]

        text_input = self._processor.apply_chat_template(messages, tokenize=False,
                                                          add_generation_prompt=True)
        inputs = self._processor(text=[text_input], images=[img], return_tensors="pt",
                                  padding=True).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=4096)

        output_ids = output_ids[:, inputs.input_ids.shape[1]:]
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return OCRResult(raw_text=text, metadata={"device": str(self._device)})
