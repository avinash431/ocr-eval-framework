"""Amazon Bedrock Data Automation (BDAn) wrapper."""

from models import register_model
from models.base import BaseOCRModel, OCRResult


@register_model
class AmazonBDAn(BaseOCRModel):
    @property
    def name(self): return "amazon_bdan"
    @property
    def display_name(self): return "Amazon Bedrock Data Automation"
    @property
    def model_type(self): return "cloud_api"

    def setup(self):
        import boto3
        cfg = self.config.get("aws", {})
        kwargs = {"region_name": cfg.get("region", "us-east-1")}
        if cfg.get("access_key_id"):
            kwargs["aws_access_key_id"] = cfg["access_key_id"]
            kwargs["aws_secret_access_key"] = cfg["secret_access_key"]
        self._bedrock = boto3.client("bedrock-runtime", **kwargs)
        self._is_setup = True

    def _ocr_impl(self, image_path: str) -> OCRResult:
        import base64, json
        from pathlib import Path

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        ext = Path(image_path).suffix.lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "pdf": "application/pdf"}.get(ext, "image/jpeg")

        # Use Claude on Bedrock for document extraction
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": media_type, "data": img_b64}},
                    {"type": "text", "text": (
                        "Extract all text from this document. Preserve the layout, "
                        "tables (as markdown), and reading order. Output only the "
                        "extracted content, no commentary."
                    )},
                ],
            }],
        })

        response = self._bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body=body,
            contentType="application/json",
        )
        resp_body = json.loads(response["body"].read())
        text = resp_body.get("content", [{}])[0].get("text", "")

        return OCRResult(
            raw_text=text,
            metadata={"model_id": "claude-3.5-sonnet-v2", "via": "bedrock"},
        )

    def estimate_cost(self, num_pages: int) -> float:
        # Bedrock Claude pricing varies; rough estimate ~$0.01-0.03 per page
        return num_pages * 0.02
